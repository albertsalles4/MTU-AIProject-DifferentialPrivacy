from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
import time
from xgboost import XGBClassifier
import random
import numpy as np

def prepare_flattened_data(X, y, apply_smote=True, smote_strategy=0.25, scaler=None, fit_scaler=False):
    if apply_smote:
        smote = SMOTE(random_state=42, sampling_strategy=smote_strategy)
        X, y = smote.fit_resample(X, y)

    if scaler is None:
        scaler = StandardScaler()

    if fit_scaler:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    return X_scaled, y, scaler



def run_classical_model_tuning(model_class, param_grid, X_train, y_train, X_val, y_val, X_test, y_test):
    X_train_proc, y_train_proc, scaler = prepare_flattened_data(X_train, y_train, fit_scaler=True)

    X_val_proc, y_val_proc, _ = prepare_flattened_data(X_val, y_val, apply_smote=False, scaler=scaler)
    X_test_proc, y_test_proc, _ = prepare_flattened_data(X_test, y_test, apply_smote=False, scaler=scaler)


    best_model = None
    best_f1 = 0
    best_params = None
    all_results = []

    print("Hyperparameter tuning...\n") 

    for params in tqdm(param_grid, desc="Tuning Progress", ncols=100):
        start_time = time.time()
        model = model_class(**params)
        model.fit(X_train_proc, y_train_proc)
        elapsed_time = time.time() - start_time

        # Predictions
        y_pred_val = model.predict(X_val_proc)
        y_proba_val = model.predict_proba(X_val_proc)[:, 1]

        # Metrics
        val_f1 = f1_score(y_val_proc, y_pred_val)
        val_precision = precision_score(y_val_proc, y_pred_val)
        val_recall = recall_score(y_val_proc, y_pred_val)
        val_auc = roc_auc_score(y_val_proc, y_proba_val)

        # Log
        tqdm.write(f"Params: {params} | F1: {val_f1:.4f} | Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | AUC: {val_auc:.4f} | Time: {elapsed_time:.2f}s")

        all_results.append({
            'params': params,
            'f1': val_f1,
            'precision': val_precision,
            'recall': val_recall,
            'auc': val_auc,
            'runtime_sec': elapsed_time
        })

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model = model
            best_params = params

    print("\nBest Parameters Found:")
    print(best_params)

    # Final test set evaluation
    y_pred_test = best_model.predict(X_test_proc)
    y_proba_test = best_model.predict_proba(X_test_proc)[:, 1]

    print("\nTest Set Evaluation:")
    print(classification_report(y_test_proc, y_pred_test, digits=4))
    print("Test AUC:", roc_auc_score(y_test_proc, y_proba_test))

    return all_results, best_model, best_params


def run_xgboost_random_tuning(X_train, y_train, X_val, y_val, X_test, y_test, param_grid, n_iter=10, random_seed=42):

    random.seed(random_seed)
    np.random.seed(random_seed)

    # Preprocess: scale once
    X_train_base, y_train_base, scaler = prepare_flattened_data(X_train, y_train, apply_smote=False, fit_scaler=True)
    X_train_smote, y_train_smote, _ = prepare_flattened_data(X_train, y_train, apply_smote=True, scaler=scaler)
    X_val_proc, y_val_proc, _ = prepare_flattened_data(X_val, y_val, apply_smote=False, scaler=scaler)
    X_test_proc, y_test_proc, _ = prepare_flattened_data(X_test, y_test, apply_smote=False, scaler=scaler)

    all_results = []
    best_model = None
    best_f1 = 0
    best_params = None

    print(f"Random hyperparameter search for XGBoost ({n_iter} iterations)...\n")

    for iter in range(n_iter):
        # Randomly sample one configuration
        print(f"Iteration {iter+1}/{n_iter}")
        params = {
            'n_estimators': random.choice(param_grid['n_estimators']),
            'max_depth': random.choice(param_grid['max_depth']),
            'learning_rate': random.choice(param_grid['learning_rate']),
            'subsample': random.choice(param_grid['subsample']),
            'colsample_bytree': random.choice(param_grid['colsample_bytree']),
            'gamma': random.choice(param_grid['gamma']),
            'use_smote': random.choice(param_grid['use_smote']),
            'eval_metric': random.choice(param_grid['eval_metric'])
        }

        # Only use scale_pos_weight if SMOTE is not used
        if not params['use_smote']:
            params['scale_pos_weight'] = random.choice(param_grid['scale_pos_weight'])

        train_X = X_train_smote if params['use_smote'] else X_train_base
        train_y = y_train_smote if params['use_smote'] else y_train_base

        model_params = params.copy()
        del model_params['use_smote']  # not needed for model

        start_time = time.time()
        model = XGBClassifier(early_stopping_rounds=15, **model_params)
        model.fit(
            train_X, train_y,
            eval_set=[(X_val_proc, y_val_proc)],
            verbose=False
        )

        elapsed_time = time.time() - start_time

        y_pred_val = model.predict(X_val_proc)
        y_proba_val = model.predict_proba(X_val_proc)[:, 1]

        val_f1 = f1_score(y_val_proc, y_pred_val)
        val_precision = precision_score(y_val_proc, y_pred_val)
        val_recall = recall_score(y_val_proc, y_pred_val)
        val_auc = roc_auc_score(y_val_proc, y_proba_val)

        tqdm.write(f"Params: {params} | F1: {val_f1:.4f} | Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | AUC: {val_auc:.4f} | Time: {elapsed_time:.2f}s")

        all_results.append({
            'params': params,
            'f1': val_f1,
            'precision': val_precision,
            'recall': val_recall,
            'auc': val_auc,
            'runtime_sec': elapsed_time,
            'best_n_rounds': model.best_iteration + 1
        })

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model = model
            best_params = params

    # Final evaluation on test set
    y_pred_test = best_model.predict(X_test_proc)
    y_proba_test = best_model.predict_proba(X_test_proc)[:, 1]

    print("\nBest Parameters Found:")
    print(best_params)

    print("\nTest Set Evaluation:")
    print(classification_report(y_test_proc, y_pred_test, digits=4))
    print("Test AUC:", roc_auc_score(y_test_proc, y_proba_test))

    return all_results, best_model, best_params