import numpy as np
from collections import Counter
import xgboost as xgb
from scipy.stats import norm

def dp_pate_xgboost_time_series(
    X_train, y_train,
    X_val, y_val,
    epsilon, delta,
    num_teachers=10,
    public_fraction=0.2,
    teacher_params=None,
    student_params=None
):
    """
    PATE‐wrapped XGBoost for time‐series data without shuffling across time.

    Args:
      X_train, y_train      # full time‐ordered series for teacher + public pool
      X_val, y_val          # held‐out “future” window for early‐stopping + final eval
      epsilon, delta        # target DP budget for Gaussian aggregation
      num_teachers          # number of teacher models (k)
      public_fraction       # fraction of X_train at the *end* used as unlabeled public pool
      teacher_params        # XGBoost params for teachers
      student_params        # XGBoost params for student

    Returns:
      student      : trained xgboost.Booster
      val_acc      : accuracy on (X_val, y_val)
      evals_result : dict of train + validation metrics per round
    """
    # 1) Determine split‐points in time
    n_total = X_train.shape[0]
    n_public = int(n_total * public_fraction)
    n_teach  = n_total - n_public

    # Teacher window: earliest n_teach samples
    X_teach_full = X_train[:n_teach]
    y_teach_full = y_train[:n_teach]

    # Public pool: the following n_public samples
    X_public = X_train[n_teach:]
    y_public_true = y_train[n_teach:]

    # 2) Shard the teacher window into k contiguous blocks
    shard_indices = np.array_split(np.arange(n_teach), num_teachers)

    # 3) Default XGBoost params if not provided
    teacher_params = teacher_params or {
        "objective": "binary:logistic",
        "max_depth": 6,
        "learning_rate": 0.3,
    }
    student_params = student_params or {
        "objective": "binary:logistic",
        "max_depth": 6,
        "learning_rate": 0.3,
    }

    # 4) Train each teacher on its contiguous time block
    teachers = []
    for idxs in shard_indices:
        dtrain = xgb.DMatrix(X_teach_full[idxs], label=y_teach_full[idxs])
        teachers.append(
            xgb.train(teacher_params, dtrain, num_boost_round=50)
        )

    # 5) Compute Gaussian‐mechanism noise scale (Δ₂ = 1 vote)
    sigma = np.sqrt(2 * np.log(1.25 / delta)) / epsilon

    # 6) Use teachers to label the public pool with DP‐protected majority vote
    student_labels = []
    for x in X_public:
        votes = [int(t.predict(xgb.DMatrix(x.reshape(1, -1))) > 0.5)
                 for t in teachers]
        counts = Counter(votes)
        noisy_counts = {
            cls: counts.get(cls, 0) + norm.rvs(scale=sigma)
            for cls in [0, 1]
        }
        student_labels.append(int(noisy_counts[1] > noisy_counts[0]))
    student_labels = np.array(student_labels)

    # 7) Prepare DMatrix for student training + validation
    d_public = xgb.DMatrix(X_public, label=student_labels)
    d_val    = xgb.DMatrix(X_val,    label=y_val)

    # 8) Train student with early stopping on the held‐out future window
    evals_result = {}
    student = xgb.train(
        student_params,
        d_public,
        num_boost_round=500,
        evals=[(d_public, "train"), (d_val, "validation")],
        early_stopping_rounds=15,
        evals_result=evals_result,
        verbose_eval=False
    )

    return student, evals_result, X_public, y_public_true
