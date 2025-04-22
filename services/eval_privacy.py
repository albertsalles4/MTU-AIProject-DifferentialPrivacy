import numpy as np
import tensorflow as tf
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import membership_inference_attack as mia
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackInputData
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import SlicingSpec
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackType
import tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.plotting as plotting


def compute_mia_attributes(model, X_train, X_test, y_train, y_test):
    y_train_pred_prob = np.asarray(model.predict(X_train)).reshape(-1, 1)
    y_test_pred_prob = np.asarray(model.predict(X_test)).reshape(-1, 1)

    eps = 1e-6
    y_train_pred_prob_clipped = np.clip(y_train_pred_prob, eps, 1 - eps)
    y_test_pred_prob_clipped = np.clip(y_test_pred_prob, eps, 1 - eps)

    # Compute the logits
    logits_train = np.log(y_train_pred_prob_clipped / (1 - y_train_pred_prob_clipped))
    logits_test = np.log(y_test_pred_prob_clipped / (1 - y_test_pred_prob_clipped))

    # Compute the binary cross-entropy loss
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
    loss_train = bce(y_train.reshape(-1, 1), y_train_pred_prob_clipped).numpy().reshape(-1)
    loss_test = bce(y_test.reshape(-1, 1), y_test_pred_prob_clipped).numpy().reshape(-1)

    labels_train = y_train.reshape(-1)
    labels_test = y_test.reshape(-1)

    return logits_train, logits_test, loss_train, loss_test, labels_train, labels_test


def compute_mia_attacks(model, X_train, X_test, y_train, y_test):
    logits_train, logits_test, loss_train, loss_test, labels_train, labels_test = compute_mia_attributes(
        model, X_train, X_test, y_train, y_test
    )
    # Create an AttackInputData object
    attack_input = AttackInputData(
        logits_train=logits_train,
        logits_test=logits_test,
        loss_train=loss_train,
        loss_test=loss_test,
        labels_train=labels_train,
        labels_test=labels_test
    )

    slicing_spec = SlicingSpec(
        entire_dataset=True,
        by_class=True,
        by_percentiles=True,
        by_classification_correctness=True
    )

    attacks_result = mia.run_attacks(
        attack_input,
        slicing_spec,
        attack_types=[AttackType.THRESHOLD_ATTACK, AttackType.LOGISTIC_REGRESSION]
    )

    max_auc_attacker = attacks_result.get_result_with_max_auc()

    figure = plotting.plot_roc_curve(max_auc_attacker.roc_curve)

    return attacks_result, figure


def compute_mia_attacks_with_preds(model, X_train, X_test, y_train, y_test):
    train_preds = model.predict_proba(X_train)[:, 1]
    test_preds = model.predict_proba(X_test)[:, 1]

    train_probs = np.stack([1 - train_preds, train_preds], axis=1)
    test_probs = np.stack([1 - test_preds, test_preds], axis=1)

    labels_train = y_train.astype(int)
    labels_test = y_test.astype(int)

    attack_input = AttackInputData(
        probs_train=train_probs,
        probs_test=test_probs,
        labels_train=labels_train,
        labels_test=labels_test
    )

    slicing_spec = SlicingSpec(
        entire_dataset=True,
        by_class=True,
        by_percentiles=True,
        by_classification_correctness=True
    )

    attacks_result = mia.run_attacks(
        attack_input,
        slicing_spec,
        attack_types=[AttackType.THRESHOLD_ATTACK, AttackType.LOGISTIC_REGRESSION]
    )

    max_auc_attacker = attacks_result.get_result_with_max_auc()

    figure = plotting.plot_roc_curve(max_auc_attacker.roc_curve)

    return attacks_result, figure
