import numpy as np
from sklearn.preprocessing import StandardScaler

def apply_dp_to_data(X, epsilon=1.0, delta=1e-5, sensitivity=1.0, clip_value=3.0, seed=42):
    """
    Applies (epsilon, delta)-Differential Privacy to the input dataset X using the Gaussian mechanism.
    
    Steps:
    - Standardizes the features using z-score
    - Clips the features to a fixed L2-bound
    - Adds Gaussian noise calibrated to epsilon and delta

    Args:
        X (np.ndarray)         : Input feature matrix
        epsilon (float)        : Target epsilon (lower = more private)
        delta (float)          : Target delta (usually ~1e-5 or lower)
        sensitivity (float)    : Sensitivity of each feature (after clipping, usually 1.0)
        clip_value (float)     : Value to clip standardized features to (e.g., [-3, 3])
        seed (int)             : Random seed for reproducibility

    Returns:
        X_dp (np.ndarray)      : Differentially private version of X
        scaler (StandardScaler): Fitted scaler (in case you need to transform test set the same way)
    """
    np.random.seed(seed)

    # 1. Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. Clip to bounded range
    X_clipped = np.clip(X_scaled, -clip_value, clip_value)

    # 3. Calculate Gaussian noise std (σ)
    sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
    print(f"[DP Noise] ε = {epsilon}, δ = {delta}, σ = {sigma:.4f}")

    # 4. Add Gaussian noise
    noise = np.random.normal(loc=0.0, scale=sigma, size=X_clipped.shape)
    X_dp = X_clipped + noise

    return X_dp, scaler
