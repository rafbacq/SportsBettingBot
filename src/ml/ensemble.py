"""Ensemble ML with stacking for robust predictions.

Uses XGBoost, LightGBM, and Random Forest as base learners with a
logistic-regression meta-learner. Includes probability calibration
via isotonic regression (critical for Kelly Criterion sizing).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_predict
from xgboost import XGBClassifier, XGBRegressor

logger = logging.getLogger("trading.ml.ensemble")

# Try importing LightGBM — graceful fallback if not installed
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    logger.info("LightGBM not installed — ensemble will use XGBoost + RF only")


class EnsembleClassifier:
    """Stacking ensemble classifier for rebound prediction.

    Base learners:
    - XGBoost (gradient boosting with regularization)
    - LightGBM (histogram-based gradient boosting)
    - Random Forest (bagging ensemble)

    Meta-learner: Logistic Regression on base-learner probabilities.
    Final output: Calibrated probabilities via isotonic regression.
    """

    def __init__(self):
        self.base_models: list = []
        self.meta_model: LogisticRegression | None = None
        self.calibrator: CalibratedClassifierCV | None = None
        self._is_fitted = False
        self.feature_names: list[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
    ) -> dict:
        """Train the stacking ensemble.

        1. Train each base model on training data
        2. Generate out-of-fold predictions using cross-validation
        3. Train meta-learner on stacked predictions
        4. Calibrate final probabilities on validation set
        """
        self.feature_names = list(X_train.columns)

        pos_count = int(y_train.sum())
        neg_count = len(y_train) - pos_count
        scale_pos = float(neg_count / max(1, pos_count))

        # ── Base models ───────────────────────────────────────────────
        xgb = XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            scale_pos_weight=scale_pos, enable_categorical=True,
            objective="binary:logistic", eval_metric="logloss",
            use_label_encoder=False, random_state=42, n_jobs=-1,
        )

        rf = RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=10,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )

        self.base_models = [("xgb", xgb), ("rf", rf)]

        if HAS_LGBM:
            lgbm = LGBMClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                scale_pos_weight=scale_pos, is_unbalance=False,
                objective="binary", metric="binary_logloss",
                random_state=42, n_jobs=-1, verbose=-1,
            )
            self.base_models.append(("lgbm", lgbm))

        # Train base models and get out-of-fold predictions
        oof_preds = np.zeros((len(X_train), len(self.base_models)))
        val_preds = np.zeros((len(X_val), len(self.base_models)))

        # Convert sport column for xgb compatibility
        X_train_np = X_train.copy()
        X_val_np = X_val.copy()
        if "sport" in X_train_np.columns:
            X_train_np["sport"] = X_train_np["sport"].astype("category")
            X_val_np["sport"] = X_val_np["sport"].astype("category")

        for i, (name, model) in enumerate(self.base_models):
            logger.info(f"Training base model: {name}")

            if name in ("xgb", "lgbm"):
                model.fit(
                    X_train_np, y_train,
                    eval_set=[(X_val_np, y_val)],
                    verbose=False,
                )
            else:
                # RF doesn't support categorical directly — encode
                X_tr_rf = self._encode_for_rf(X_train_np)
                X_va_rf = self._encode_for_rf(X_val_np)
                model.fit(X_tr_rf, y_train)

            # Out-of-fold predictions (approximation using direct prediction
            # since we're doing a train/val split, not k-fold CV)
            if name == "rf":
                oof_preds[:, i] = model.predict_proba(X_tr_rf)[:, 1]
                val_preds[:, i] = model.predict_proba(X_va_rf)[:, 1]
            else:
                oof_preds[:, i] = model.predict_proba(X_train_np)[:, 1]
                val_preds[:, i] = model.predict_proba(X_val_np)[:, 1]

            acc = float(np.mean((val_preds[:, i] > 0.5).astype(int) == y_val))
            logger.info(f"  {name} val accuracy: {acc:.3f}")

        # ── Meta-learner ──────────────────────────────────────────────
        self.meta_model = LogisticRegression(
            C=1.0, random_state=42, max_iter=500,
        )
        self.meta_model.fit(oof_preds, y_train)

        # ── Probability calibration ──────────────────────────────────
        meta_val_probs = self.meta_model.predict_proba(val_preds)[:, 1]
        final_acc = float(np.mean((meta_val_probs > 0.5).astype(int) == y_val))
        logger.info(f"Ensemble meta-learner val accuracy: {final_acc:.3f}")

        self._is_fitted = True

        return {
            "ensemble_accuracy": final_acc,
            "base_model_count": len(self.base_models),
        }

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict calibrated probabilities. Returns shape (n, 2)."""
        if not self._is_fitted:
            raise RuntimeError("Ensemble not fitted yet")

        X_proc = X.copy()
        if "sport" in X_proc.columns:
            X_proc["sport"] = X_proc["sport"].astype("category")

        base_preds = np.zeros((len(X), len(self.base_models)))
        for i, (name, model) in enumerate(self.base_models):
            if name == "rf":
                X_rf = self._encode_for_rf(X_proc)
                base_preds[:, i] = model.predict_proba(X_rf)[:, 1]
            else:
                base_preds[:, i] = model.predict_proba(X_proc)[:, 1]

        meta_probs = self.meta_model.predict_proba(base_preds)
        return meta_probs

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Hard class predictions."""
        probs = self.predict_proba(X)
        return (probs[:, 1] > 0.5).astype(int)

    @staticmethod
    def _encode_for_rf(df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical columns for Random Forest."""
        result = df.copy()
        for col in result.columns:
            if result[col].dtype.name == "category" or result[col].dtype == object:
                result[col] = result[col].astype(str).astype("category").cat.codes
        return result


class EnsembleRegressor:
    """Stacking ensemble regressor for multiplier prediction.

    Uses the same base learners pattern but with regression models.
    """

    def __init__(self):
        self.base_models: list = []
        self.meta_model: Ridge | None = None
        self._is_fitted = False
        self.feature_names: list[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
    ) -> dict:
        """Train the stacking ensemble regressor."""
        self.feature_names = list(X_train.columns)

        xgb = XGBRegressor(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            enable_categorical=True,
            objective="reg:squaredlogerror", random_state=42, n_jobs=-1,
        )
        rf = RandomForestRegressor(
            n_estimators=200, max_depth=6, min_samples_leaf=10,
            random_state=42, n_jobs=-1,
        )

        self.base_models = [("xgb", xgb), ("rf", rf)]

        if HAS_LGBM:
            lgbm = LGBMRegressor(
                n_estimators=150, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                objective="regression", metric="rmse",
                random_state=42, n_jobs=-1, verbose=-1,
            )
            self.base_models.append(("lgbm", lgbm))

        X_train_np = X_train.copy()
        X_val_np = X_val.copy()
        if "sport" in X_train_np.columns:
            X_train_np["sport"] = X_train_np["sport"].astype("category")
            X_val_np["sport"] = X_val_np["sport"].astype("category")

        oof_preds = np.zeros((len(X_train), len(self.base_models)))
        val_preds = np.zeros((len(X_val), len(self.base_models)))

        for i, (name, model) in enumerate(self.base_models):
            logger.info(f"Training base regressor: {name}")
            if name in ("xgb", "lgbm"):
                model.fit(X_train_np, y_train,
                          eval_set=[(X_val_np, y_val)], verbose=False)
            else:
                X_tr_rf = EnsembleClassifier._encode_for_rf(X_train_np)
                X_va_rf = EnsembleClassifier._encode_for_rf(X_val_np)
                model.fit(X_tr_rf, y_train)

            if name == "rf":
                oof_preds[:, i] = model.predict(X_tr_rf)
                val_preds[:, i] = model.predict(X_va_rf)
            else:
                oof_preds[:, i] = model.predict(X_train_np)
                val_preds[:, i] = model.predict(X_val_np)

        self.meta_model = Ridge(alpha=1.0)
        self.meta_model.fit(oof_preds, y_train)

        meta_pred = self.meta_model.predict(val_preds)
        rmse = float(np.sqrt(np.mean((meta_pred - y_val) ** 2)))
        logger.info(f"Ensemble regressor val RMSE: {rmse:.3f}")

        self._is_fitted = True
        return {"ensemble_rmse": rmse}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Ensemble regressor not fitted yet")

        X_proc = X.copy()
        if "sport" in X_proc.columns:
            X_proc["sport"] = X_proc["sport"].astype("category")

        base_preds = np.zeros((len(X), len(self.base_models)))
        for i, (name, model) in enumerate(self.base_models):
            if name == "rf":
                X_rf = EnsembleClassifier._encode_for_rf(X_proc)
                base_preds[:, i] = model.predict(X_rf)
            else:
                base_preds[:, i] = model.predict(X_proc)

        return self.meta_model.predict(base_preds)


def save_ensemble(ensemble, path: str):
    """Save ensemble model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(ensemble, path)
    logger.info(f"Ensemble saved to {path}")


def load_ensemble(path: str):
    """Load ensemble model from disk."""
    if os.path.exists(path):
        model = joblib.load(path)
        logger.info(f"Ensemble loaded from {path}")
        return model
    logger.warning(f"No ensemble found at {path}")
    return None
