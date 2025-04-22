import numpy as np
import xgboost as xgb

class XGBProbaWrapper:
    def __init__(self, booster: xgb.Booster):
        """
        Wraps an xgboost.Booster to give it a sklearn‐like predict_proba.
        If you trained with early_stopping_rounds, booster.best_iteration
        will be set; otherwise we fall back to using all trees.
        """
        self.booster = booster
        # XGBoost core‐API sets best_iteration if early stopping occurred:
        self.best_iteration = getattr(booster, "best_iteration", None)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Returns an (n_samples,) array of predictions.
        honoring early stopping if available.
        """
        dm = xgb.DMatrix(X)
        if self.best_iteration is not None:
            # use only up to the best iteration + 1
            return self.booster.predict(dm, iteration_range=(0, self.best_iteration + 1))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns an (n_samples, 2) array [[P(y=0), P(y=1)], …]
        honoring early stopping if available.
        """
        dm = xgb.DMatrix(X)
        if self.best_iteration is not None:
            # use only up to the best iteration + 1
            p1 = self.booster.predict(dm, iteration_range=(0, self.best_iteration + 1))
        else:
            # no early stopping—use all trees
            p1 = self.booster.predict(dm)
        return np.vstack([1 - p1, p1]).T
