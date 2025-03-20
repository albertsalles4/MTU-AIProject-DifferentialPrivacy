import numpy as np
from sklearn.model_selection import train_test_split
from .load_dataset import train_test_split_sorted
from imblearn.over_sampling import SMOTE
import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report



# --------------------------------------#
# 2. Function to Create Sequences       #
# --------------------------------------#
def create_sequences(data, labels, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length + 1):
        X.append(data[i:i+seq_length])
        # Label a sequence as fraudulent if any transaction in it is fraud
        y.append(1 if labels[i:i+seq_length].sum() > 0 else 0)
    return np.array(X), np.array(y)


def apply_smote(X_seq, y_seq, sampling_strategy=0.5):
    seq_length = X_seq.shape[1]
    # Apply SMOTE on the training set
    n_samples, seq_len, n_features_local = X_seq.shape
    X_train_flat = X_seq.reshape(n_samples, seq_len * n_features_local)
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_flat, y_seq)
    X_train_res = X_train_res.reshape(-1, seq_length, n_features_local)

    return X_train_res, y_train_res


# 3. Custom HyperModel that Tunes Sequence Length     #
#    and Other Hyperparameters                        #
class LSTMHyperModel(kt.HyperModel):


    def __init__(self, X_train, labels):
        super().__init__()
        self.X_train = X_train
        self.labels = labels

    def build(self, hp):
        # Tune the sequence length: choose from 5, 10, or 20
        seq_length = hp.Choice('sequence_length', [5, 10, 20])

        model = Sequential()
        # First LSTM layer
        units1 = hp.Int('units_lstm1', min_value=32, max_value=128, step=32)
        model.add(LSTM(units1, input_shape=(seq_length, self.X_train.shape[-1]), return_sequences=True))
        dropout1 = hp.Float('dropout1', min_value=0.1, max_value=0.5, step=0.1)
        model.add(Dropout(dropout1))

        # Second LSTM layer
        units2 = hp.Int('units_lstm2', min_value=16, max_value=64, step=16)
        model.add(LSTM(units2))
        dropout2 = hp.Float('dropout2', min_value=0.1, max_value=0.5, step=0.1)
        model.add(Dropout(dropout2))

        # Dense output layer
        model.add(Dense(1, activation='sigmoid'))

        # Tune learning rate
        learning_rate = hp.Float('learning_rate', min_value=1e-7, max_value=1e-3, sampling='LOG')
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_focal_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        return model

    def fit(self, hp, model, *args, **kwargs):
        seq_length = hp.get('sequence_length')
        X_seq, y_seq = create_sequences(self.X_train, self.labels, seq_length)
        X_train_fit, X_val_fit, y_train_fit, y_val_fit = train_test_split_sorted(
            X_seq, y_seq, test_size=0.1
        )
        
        # Apply SMOTE on the training set
        n_samples, seq_len, n_features_local = X_train_fit.shape
        X_train_flat = X_train_fit.reshape(n_samples, seq_len * n_features_local)
        smote = SMOTE(random_state=42, sampling_strategy=0.25)
        X_train_res, y_train_res = smote.fit_resample(X_train_flat, y_train_fit)
        X_train_res = X_train_res.reshape(-1, seq_length, n_features_local)

        return model.fit(
            X_train_res, y_train_res,
            validation_data=(X_val_fit, y_val_fit),
            epochs=10,
            batch_size=32,
            **kwargs
        )
    

def run_tuner(X_train, y_train, X_test, y_test):
    # 4. Set Up and Run the Tuner  #
    hypermodel = LSTMHyperModel(X_train=X_train, labels=y_train)

    tuner = kt.RandomSearch(
        hypermodel,
        objective=kt.Objective('val_precision', direction='max'),
        max_trials=10,         # Increase for a more thorough search
        executions_per_trial=1,
        directory='hyperparam_tuning',
        project_name='credit_card_fraud_lstm'
    )

    # Since our HyperModel.fit() method handles data generation,
    # we don't need to pass x and y to tuner.search().
    tuner.search()

    # -----------------------------#
    # 5. Evaluate the Best Model   #
    # -----------------------------#
    # Get the best hyperparameters and model
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_seq_length = best_hp.get('sequence_length')

    # For final evaluation, generate a test set using the best sequence length.
    X_train_final, y_train_final = create_sequences(X_train, y_train, best_seq_length)
    X_test_final, y_test_final = create_sequences(X_test, y_test, best_seq_length)

    # Apply SMOTE on the training set
    n_samples, seq_len, n_features_local = X_train_final.shape
    X_train_final_flat = X_train_final.reshape(n_samples, seq_len * n_features_local)
    smote = SMOTE(random_state=42, sampling_strategy=0.25)
    X_train_final_res, y_train_final_res = smote.fit_resample(X_train_final_flat, y_train_final)
    X_train_final_res = X_train_final_res.reshape(-1, best_seq_length, n_features_local)

    # Retrieve the best model and evaluate on the test set
    best_model = tuner.get_best_models(num_models=1)[0]
    loss, accuracy, precision, recall, auc_metric = best_model.evaluate(X_test_final, y_test_final, verbose=0)
    print("Best Model Evaluation on Test Set:")
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"AUC: {auc_metric:.4f}")


    # -----------------------------------------#
    # 5. Print All Trials and Best Parameters  #
    # -----------------------------------------#

    print("\nAll Trial Results:")
    for trial in tuner.oracle.trials.values():
        print(f"Trial ID: {trial.trial_id}, Score: {trial.score}, Hyperparameters: {trial.hyperparameters.values}")

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("\nBest Hyperparameters:")
    print(best_hp.values)

    return best_model, best_hp, X_train_final_res, y_train_final_res, X_test_final, y_test_final


def train_and_plot_results(X_train, y_train, X_test, y_test, best_hp, X_val=None, y_val=None, validation_split=None):
    if validation_split is None and (X_val is None or y_val is None):
        raise ValueError("Either provide a validation split or a validation set.")

    if validation_split is not None:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=validation_split, random_state=42, stratify=y_train
        )
    # Train the model with the best hyperparameters using more epochs and plot the learning curve
    hypermodel = LSTMHyperModel(X_train=X_train, labels=y_train)
    model = hypermodel.build(best_hp)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=32
    )

    # Plot the learning curve
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.title('Loss')
    plt.legend()
    plt.grid(True)
    plt.subplot(2, 2, 2)
    plt.plot(history.history['auc'], label='train')
    plt.plot(history.history['val_auc'], label='validation')
    plt.title('AUC')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.subplot(2, 2, 3)
    plt.plot(history.history['precision'], label='train')
    plt.plot(history.history['val_precision'], label='validation')
    plt.title('Precision')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.subplot(2, 2, 4)
    plt.plot(history.history['recall'], label='train')
    plt.plot(history.history['val_recall'], label='validation')
    plt.title('Recall')
    plt.show()


    # Predict the test set
    y_pred = model.predict(X_test)

    # Print a classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred > 0.5))

    # Create a confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred > 0.5)
    print("Confusion Matrix:")
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Normal', 'Fraud'])
    disp.plot()
    plt.show()

    return model, history
    