"""Regime classifier — Cross vs Non-Cross.

XGBoost binary classifier that determines whether the current game state
represents a Cross regime (strong team collapse → full recovery possible)
or a Non-Cross regime (weak team collapse → partial rebound only).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from xgboost import XGBClassifier

from src.features.engine import FeatureVector

logger = logging.getLogger("trading.regime_classifier")


class RegimeClassifier:
    """XGBoost-based regime classifier: Cross (1) vs Non-Cross (0)."""

    def __init__(self, model_path: str | None = None):
        self.model: XGBClassifier | None = None
        self.model_path = model_path
        self.feature_names = FeatureVector.feature_names()

        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_cols: list[str] | None = None,
        target_col: str = "regime",
    ) -> dict:
        """Train the regime classifier.

        Args:
            train_df: Training data with features and target column.
            val_df: Validation data.
            feature_cols: Feature column names. Defaults to FeatureVector names.
            target_col: Target column name (should contain 'cross' / 'non_cross').

        Returns:
            Dictionary of evaluation metrics.
        """
        if feature_cols is None:
            feature_cols = [c for c in self.feature_names if c in train_df.columns]

        # Encode target: cross=1, non_cross=0
        y_train = (train_df[target_col] == "cross").astype(int)
        y_val = (val_df[target_col] == "cross").astype(int)
        
        X_train = train_df[feature_cols].copy()
        if "sport" in feature_cols:
            X_train["sport"] = X_train["sport"].astype("category")
        X_val = val_df[feature_cols].copy()
        if "sport" in feature_cols:
            X_val["sport"] = X_val["sport"].astype("category")

        pos_count = y_train.sum()
        neg_count = len(y_train) - pos_count
        scale_pos = float(neg_count / max(1, pos_count))

        logger.info(
            f"Training regime classifier: {len(X_train)} train, {len(X_val)} val | "
            f"Cross rate: {y_train.mean():.1%} (train), {y_val.mean():.1%} (val)"
        )

        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos,
            enable_categorical=True,
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1,
        )

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Evaluate
        y_pred = self.model.predict(X_val)
        y_prob = self.model.predict_proba(X_val)[:, 1]

        accuracy = accuracy_score(y_val, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, y_pred, average="binary"
        )

        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

        logger.info(
            f"Regime classifier results: "
            f"accuracy={accuracy:.3f}, precision={precision:.3f}, "
            f"recall={recall:.3f}, f1={f1:.3f}"
        )

        # Feature importance
        importances = self.model.feature_importances_
        feat_imp = sorted(
            zip(feature_cols, importances),
            key=lambda x: x[1],
            reverse=True,
        )
        logger.info(
            "Top features: " +
            ", ".join(f"{n}={v:.3f}" for n, v in feat_imp[:5])
        )

        return metrics

    def predict(self, features: FeatureVector) -> dict:
        """Predict regime for a single feature vector.

        Returns:
            {"regime": "cross"|"non_cross", "confidence": float}
        """
        if self.model is None:
            # Fallback: use heuristic
            return self._heuristic_predict(features)

        df_feat = pd.DataFrame([features.to_dict()])
        if "sport" in df_feat.columns:
            df_feat["sport"] = df_feat["sport"].astype("category")
        X = df_feat[self.feature_names]

        prob = float(self.model.predict_proba(X)[0, 1])
        regime = "cross" if prob >= 0.5 else "non_cross"

        return {"regime": regime, "confidence": prob if regime == "cross" else 1 - prob}

    def _heuristic_predict(self, features: FeatureVector) -> dict:
        """Rule-based fallback when no model is loaded."""
        # Strong team has collapsed significantly
        if features.s_value > 2.0 and features.prob_a_initial > 0.55:
            return {"regime": "cross", "confidence": 0.55}

        # Weak team at extreme low
        if features.op_value > 2.0 and features.prob_b_current < 0.20:
            return {"regime": "non_cross", "confidence": 0.55}

        return {"regime": "non_cross", "confidence": 0.50}

    def save(self, path: str | None = None):
        """Save trained model to disk."""
        path = path or self.model_path
        if path and self.model:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            joblib.dump(self.model, path)
            logger.info(f"Regime classifier saved to {path}")

    def load(self, path: str | None = None):
        """Load trained model from disk."""
        path = path or self.model_path
        if path and os.path.exists(path):
            self.model = joblib.load(path)
            logger.info(f"Regime classifier loaded from {path}")
        else:
            logger.warning(f"No model found at {path}")
