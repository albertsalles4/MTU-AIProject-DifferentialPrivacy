import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow_privacy import compute_dp_sgd_privacy_statement
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer
from imblearn.over_sampling import SMOTE
from .modeling import create_sequences
import re

class PrivacyCalculator:
    @staticmethod
    def parse_privacy_statement(statement):
        """Convert the text statement into a structured dict"""
        result = {
            'parameters': {},
            'guarantees': {}
        }
        
        # Extract parameters
        param_patterns = {
            'examples': r"over (\d+) examples",
            'batch_size': r"(\d+) examples per iteration",
            'noise_multiplier': r"noise multiplier (\d+\.?\d*)",
            'epochs': r"for (\d+) epochs",
            'microbatching': r"(with|without) microbatching"
        }
        
        for key, pattern in param_patterns.items():
            match = re.search(pattern, statement)
            if match:
                value = match.group(1)
                if key in ['examples', 'batch_size', 'epochs']:
                    value = int(value)
                elif key == 'noise_multiplier':
                    value = float(value)
                elif key == 'microbatching':
                    value = match.group(1) == 'with'
                result['parameters'][key] = value
        
        # Extract guarantees
        guarantee_patterns = {
            'delta': r"delta = (\d+\.?\d*e?-?\d*)",
            'epsilon_conservative': r"Epsilon with each example occurring once per epoch:\s+([\d.]+)",
            'epsilon_poisson': r"Epsilon assuming Poisson sampling \(\*\):\s+([\d.]+)"
        }
        
        for key, pattern in guarantee_patterns.items():
            match = re.search(pattern, statement)
            if match:
                value = float(match.group(1))
                result['guarantees'][key] = value
        
        return result

    @classmethod
    def compute_privacy_metrics(cls, n, batch_size, noise_multiplier, epochs, delta, microbatching=True):
        """Get structured privacy metrics"""
        statement = compute_dp_sgd_privacy_statement(
            number_of_examples=n,
            batch_size=batch_size,
            noise_multiplier=noise_multiplier,
            num_epochs=epochs,
            delta=delta,
            used_microbatching=microbatching
        )
        return cls.parse_privacy_statement(statement)


def calculate_noise_multiplier(delta, epochs, batch_size, dataset_size, target_epsilon, microbatching):
    """
    Calculate required noise multiplier to achieve target epsilon for given delta
    
    Returns:
        noise_multiplier, actual_epsilon
    """
    low, high = 0.001, 1000.0
    tolerance = 0.01
    
    while high - low > tolerance:
        mid = (low + high) / 2
        metrics = PrivacyCalculator.compute_privacy_metrics(
            n=dataset_size,
            batch_size=batch_size,
            noise_multiplier=mid,
            epochs=epochs,
            delta=delta,
            microbatching=microbatching
        )
        
        epsilon = metrics['guarantees']['epsilon_conservative']
        
        if epsilon > target_epsilon:
            low = mid
        else:
            high = mid
    
    final_metrics = PrivacyCalculator.compute_privacy_metrics(
        n=dataset_size,
        batch_size=batch_size,
        noise_multiplier=(low + high)/2,
        epochs=epochs,
        delta=delta,
        microbatching=microbatching
    )
    
    return {
        'noise_multiplier': (low + high)/2,
        'privacy_metrics': final_metrics,
        'epsilon': final_metrics['guarantees']['epsilon_conservative']
    }

class DPLSTMModel:
    def __init__(self, X_train, y_train, X_val, y_val, hp):
        """
        Initialize the DP LSTM model with training data and hyperparameters
        
        Args:
            X_train: Input features
            y_train: Target labels
            hp: Dictionary of hyperparameters with keys:
                - sequence_length
                - units_lstm1
                - dropout1
                - units_lstm2
                - dropout2
                - learning_rate
        """
        self.X_train, self.y_train = create_sequences(X_train, y_train, hp['sequence_length'])
        self.X_val, self.y_val = create_sequences(X_val, y_val, hp['sequence_length'])
        self.best_hp = hp
        self.model = None

    def build_model(self, noise_multiplier=1.0):
        """Build the LSTM model with DP optimizer"""
        model = Sequential([
            LSTM(self.best_hp['units_lstm1'],
                 input_shape=(self.best_hp['sequence_length'], self.X_train.shape[-1]),
                 return_sequences=True),
            Dropout(self.best_hp['dropout1']),
            LSTM(self.best_hp['units_lstm2']),
            Dropout(self.best_hp['dropout2']),
            Dense(1, activation='sigmoid')
        ])
        
        optimizer = DPKerasAdamOptimizer(
            l2_norm_clip=self.best_hp['l2_norm_clip'],
            noise_multiplier=noise_multiplier,
            num_microbatches=self.best_hp['num_microbatches'],
            learning_rate=self.best_hp['learning_rate']
        )
        
        model.compile(
            optimizer=optimizer,
            loss='binary_focal_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        
        self.model = model
        return model
        
    def train(self, epochs=10, batch_size=32):
        """
        Train model with DP using pre-existing validation data
        
        Args:
            epochs: Number of training epochs
            batch_size: Must be divisible by num_microbatches (16)
        """
        # Verify batch size compatibility
        if batch_size % self.best_hp['num_microbatches'] != 0:
            raise ValueError(f"Batch size {batch_size} must be divisible by {self.best_hp['num_microbatches']}")
        
        # Apply SMOTE only to training data
        n_samples, seq_len, n_features = self.X_train.shape
        X_train_flat = self.X_train.reshape(n_samples, seq_len * n_features)
        smote = SMOTE(random_state=42, sampling_strategy=0.25)
        X_train_res, y_train_res = smote.fit_resample(X_train_flat, self.y_train)
        X_train_res = X_train_res.reshape(-1, self.best_hp['sequence_length'], n_features)
        
        # Train with DP
        history = self.model.fit(
            X_train_res, y_train_res,
            validation_data=(self.X_val, self.y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        return history
    

def train_model_with_delta(delta, X_train, y_train, X_val, y_val, best_hp, target_epsilon=1.0, epochs=10, batch_size=32):
    dp_analyzer = DPLSTMModel(X_train, y_train, X_val, y_val, best_hp)

    dataset_size = len(dp_analyzer.X_train)
    privacy_results = calculate_noise_multiplier(delta=delta, epochs=epochs, batch_size=batch_size, dataset_size=dataset_size, target_epsilon=target_epsilon, microbatching=(best_hp['num_microbatches'] > 1))
    noise_multiplier = privacy_results['noise_multiplier']
    actual_epsilon = privacy_results['epsilon']

    print(f"δ={delta:.1e}: Using noise_multiplier={noise_multiplier:.3f} (Achieved ε={actual_epsilon:.2f})")

    dp_analyzer.build_model(noise_multiplier=noise_multiplier)
    history = dp_analyzer.train(epochs=epochs, batch_size=batch_size)

    return {
        'model': dp_analyzer.model,
        'history': history,
        'epsilon': actual_epsilon,
        'noise_multiplier': noise_multiplier
    }
