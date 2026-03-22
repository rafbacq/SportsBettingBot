"""ML-driven parameter optimizer for Non-Cross and Cross models.

Trains XGBoost ensembles (rebound classifier + multiplier regressor)
to learn optimal entry/exit parameters from historical trade data.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBClassifier, XGBRegressor

from src.data.models import CrossParams, ExitStrategy, NonCrossParams
from src.features.engine import FeatureVector

logger = logging.getLogger("trading.param_optimizer")


class NonCrossParamOptimizer:
    """Learns optimal Non-Cross trading parameters from historical data."""

    def __init__(self, model_path: str | None = None):
        self.rebound_model: XGBClassifier | None = None
        self.multiplier_model: XGBRegressor | None = None
        self.model_path = model_path
        self.feature_names = FeatureVector.feature_names()
        self._optimal_entry_low = 0.01
        self._optimal_entry_high = 0.05
        self._optimal_op_threshold = 5.0
        self._optimal_exit_mult = 6.0
        self._optimal_min_time = 0.20
        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
              feature_cols: list[str] | None = None) -> dict:
        train = train_df[train_df["regime"] == "non_cross"].copy()
        val = val_df[val_df["regime"] == "non_cross"].copy()
        if len(train) < 20:
            logger.warning(f"Too few Non-Cross samples ({len(train)})")
            return {"status": "skipped"}
        if feature_cols is None:
            feature_cols = [c for c in self.feature_names if c in train.columns]
        X_train, X_val = train[feature_cols].values, val[feature_cols].values

        # Rebound probability model
        y_rb_tr = train["did_rebound"].astype(int).values
        y_rb_va = val["did_rebound"].astype(int).values
        self.rebound_model = XGBClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            objective="binary:logistic", eval_metric="logloss",
            use_label_encoder=False, random_state=42, n_jobs=-1)
        self.rebound_model.fit(X_train, y_rb_tr, eval_set=[(X_val, y_rb_va)], verbose=False)
        rb_acc = float(np.mean(self.rebound_model.predict(X_val) == y_rb_va))
        logger.info(f"Non-Cross rebound accuracy: {rb_acc:.3f}")

        # Multiplier regressor
        y_m_tr = np.clip(train["max_rebound_multiplier"].values, 0, 50)
        y_m_va = np.clip(val["max_rebound_multiplier"].values, 0, 50)
        self.multiplier_model = XGBRegressor(
            n_estimators=150, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            objective="reg:squaredlogerror", random_state=42, n_jobs=-1)
        self.multiplier_model.fit(X_train, y_m_tr, eval_set=[(X_val, y_m_va)], verbose=False)
        m_pred = self.multiplier_model.predict(X_val)
        m_rmse = float(np.sqrt(mean_squared_error(y_m_va, m_pred)))
        logger.info(f"Non-Cross multiplier RMSE: {m_rmse:.3f}")

        # Derive optimal params
        prof = train[(train["did_rebound"]) & (train["max_rebound_multiplier"] > 2.0)]
        if len(prof) >= 5:
            self._optimal_entry_low = float(prof["prob_a_current"].quantile(0.10))
            self._optimal_entry_high = float(prof["prob_a_current"].quantile(0.90))
            self._optimal_op_threshold = float(prof["op_value"].quantile(0.25))
            self._optimal_exit_mult = float(prof["max_rebound_multiplier"].median())
            self._optimal_min_time = float(prof["time_remaining_frac"].quantile(0.10))

        return {"rebound_accuracy": rb_acc, "multiplier_rmse": m_rmse}

    def predict_params(self, features: FeatureVector) -> NonCrossParams:
        if self.rebound_model is None:
            return NonCrossParams()
        X = features.to_array().reshape(1, -1)
        est_m = float(self.multiplier_model.predict(X)[0])
        return NonCrossParams(
            entry_prob_low=self._optimal_entry_low,
            entry_prob_high=self._optimal_entry_high,
            op_threshold=self._optimal_op_threshold,
            exit_multiplier=min(max(est_m * 0.6, 4.0), 12.0),
            min_time_remaining_frac=self._optimal_min_time)

    def predict_ev(self, features: FeatureVector) -> float:
        if self.rebound_model is None:
            return 0.0
        X = features.to_array().reshape(1, -1)
        p = float(self.rebound_model.predict_proba(X)[0, 1])
        m = float(self.multiplier_model.predict(X)[0])
        return p * m - 1.0

    def save(self, path: str | None = None):
        path = path or self.model_path
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            joblib.dump({
                "rebound_model": self.rebound_model,
                "multiplier_model": self.multiplier_model,
                "optimal_entry_low": self._optimal_entry_low,
                "optimal_entry_high": self._optimal_entry_high,
                "optimal_op_threshold": self._optimal_op_threshold,
                "optimal_exit_mult": self._optimal_exit_mult,
                "optimal_min_time": self._optimal_min_time,
            }, path)
            logger.info(f"Non-Cross model saved to {path}")

    def load(self, path: str | None = None):
        path = path or self.model_path
        if path and os.path.exists(path):
            d = joblib.load(path)
            self.rebound_model = d["rebound_model"]
            self.multiplier_model = d["multiplier_model"]
            self._optimal_entry_low = d.get("optimal_entry_low", 0.01)
            self._optimal_entry_high = d.get("optimal_entry_high", 0.05)
            self._optimal_op_threshold = d.get("optimal_op_threshold", 5.0)
            self._optimal_exit_mult = d.get("optimal_exit_mult", 6.0)
            self._optimal_min_time = d.get("optimal_min_time", 0.20)
            logger.info(f"Non-Cross model loaded from {path}")


class CrossParamOptimizer:
    """Learns optimal Cross model trading parameters from historical data."""

    def __init__(self, model_path: str | None = None):
        self.rebound_model: XGBClassifier | None = None
        self.multiplier_model: XGBRegressor | None = None
        self.strategy_model: XGBClassifier | None = None
        self.model_path = model_path
        self.feature_names = FeatureVector.feature_names()
        self._optimal_start_low = 0.60
        self._optimal_start_high = 1.00
        self._optimal_collapse_low = 0.03
        self._optimal_collapse_high = 0.20
        self._optimal_s_threshold = 4.0
        self._optimal_exit_mult = 10.0
        self._optimal_min_time = 0.20
        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
              feature_cols: list[str] | None = None) -> dict:
        train = train_df[train_df["regime"] == "cross"].copy()
        val = val_df[val_df["regime"] == "cross"].copy()
        if len(train) < 20:
            logger.warning(f"Too few Cross samples ({len(train)})")
            return {"status": "skipped"}
        if feature_cols is None:
            feature_cols = [c for c in self.feature_names if c in train.columns]
        X_train, X_val = train[feature_cols].values, val[feature_cols].values

        # Recovery classifier
        y_rb_tr = train["did_rebound"].astype(int).values
        y_rb_va = val["did_rebound"].astype(int).values
        self.rebound_model = XGBClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            objective="binary:logistic", eval_metric="logloss",
            use_label_encoder=False, random_state=42, n_jobs=-1)
        self.rebound_model.fit(X_train, y_rb_tr, eval_set=[(X_val, y_rb_va)], verbose=False)
        rb_acc = float(np.mean(self.rebound_model.predict(X_val) == y_rb_va))
        logger.info(f"Cross recovery accuracy: {rb_acc:.3f}")

        # Multiplier regressor
        y_m_tr = np.clip(train["max_rebound_multiplier"].values, 0, 50)
        y_m_va = np.clip(val["max_rebound_multiplier"].values, 0, 50)
        self.multiplier_model = XGBRegressor(
            n_estimators=150, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            objective="reg:squaredlogerror", random_state=42, n_jobs=-1)
        self.multiplier_model.fit(X_train, y_m_tr, eval_set=[(X_val, y_m_va)], verbose=False)
        m_rmse = float(np.sqrt(mean_squared_error(
            y_m_va, self.multiplier_model.predict(X_val))))
        logger.info(f"Cross multiplier RMSE: {m_rmse:.3f}")

        # Exit strategy classifier
        train["best_exit"] = train.apply(self._label_exit, axis=1)
        val["best_exit"] = val.apply(self._label_exit, axis=1)
        smap = {"full_hold": 0, "multiplier": 1, "dynamic": 2}
        self.strategy_model = XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            objective="multi:softmax", num_class=3, eval_metric="mlogloss",
            use_label_encoder=False, random_state=42, n_jobs=-1)
        self.strategy_model.fit(
            X_train, train["best_exit"].map(smap).values,
            eval_set=[(X_val, val["best_exit"].map(smap).values)],
            verbose=False)

        # Derive optimal params
        prof = train[(train["did_rebound"]) & (train["max_rebound_multiplier"] > 2.0)]
        if len(prof) >= 5:
            fav_probs = prof["prob_a_initial"].apply(lambda x: max(x, 1-x))
            self._optimal_start_low = float(fav_probs.quantile(0.10))
            self._optimal_collapse_low = float(prof["prob_a_current"].quantile(0.10))
            self._optimal_collapse_high = float(prof["prob_a_current"].quantile(0.90))
            self._optimal_s_threshold = float(prof["s_value"].quantile(0.25))
            self._optimal_exit_mult = float(prof["max_rebound_multiplier"].median())
            self._optimal_min_time = float(prof["time_remaining_frac"].quantile(0.10))

        return {"recovery_accuracy": rb_acc, "multiplier_rmse": m_rmse}

    @staticmethod
    def _label_exit(row) -> str:
        ep = row.get("exit_prob_at_max", 0.0)
        m = row.get("max_rebound_multiplier", 1.0)
        if ep > 0.60:
            return "full_hold"
        elif m > 8:
            return "multiplier"
        return "dynamic"

    def predict_params(self, features: FeatureVector) -> CrossParams:
        if self.rebound_model is None:
            return CrossParams()
        X = features.to_array().reshape(1, -1)
        est_m = float(self.multiplier_model.predict(X)[0])
        smap_inv = {0: ExitStrategy.FULL_HOLD, 1: ExitStrategy.MULTIPLIER, 2: ExitStrategy.DYNAMIC}
        strat = ExitStrategy.MULTIPLIER
        if self.strategy_model is not None:
            strat = smap_inv.get(int(self.strategy_model.predict(X)[0]), ExitStrategy.MULTIPLIER)
        return CrossParams(
            start_prob_low=self._optimal_start_low,
            start_prob_high=self._optimal_start_high,
            collapse_prob_low=self._optimal_collapse_low,
            collapse_prob_high=self._optimal_collapse_high,
            s_threshold=self._optimal_s_threshold,
            exit_multiplier=min(max(est_m * 0.6, 5.0), 20.0),
            exit_strategy=strat,
            min_time_remaining_frac=self._optimal_min_time)

    def predict_ev(self, features: FeatureVector) -> float:
        if self.rebound_model is None:
            return 0.0
        X = features.to_array().reshape(1, -1)
        p = float(self.rebound_model.predict_proba(X)[0, 1])
        m = float(self.multiplier_model.predict(X)[0])
        return p * m - 1.0

    def save(self, path: str | None = None):
        path = path or self.model_path
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            joblib.dump({
                "rebound_model": self.rebound_model,
                "multiplier_model": self.multiplier_model,
                "strategy_model": self.strategy_model,
                "optimal_start_low": self._optimal_start_low,
                "optimal_start_high": self._optimal_start_high,
                "optimal_collapse_low": self._optimal_collapse_low,
                "optimal_collapse_high": self._optimal_collapse_high,
                "optimal_s_threshold": self._optimal_s_threshold,
                "optimal_exit_mult": self._optimal_exit_mult,
                "optimal_min_time": self._optimal_min_time,
            }, path)
            logger.info(f"Cross model saved to {path}")

    def load(self, path: str | None = None):
        path = path or self.model_path
        if path and os.path.exists(path):
            d = joblib.load(path)
            self.rebound_model = d["rebound_model"]
            self.multiplier_model = d["multiplier_model"]
            self.strategy_model = d.get("strategy_model")
            self._optimal_start_low = d.get("optimal_start_low", 0.60)
            self._optimal_start_high = d.get("optimal_start_high", 1.00)
            self._optimal_collapse_low = d.get("optimal_collapse_low", 0.03)
            self._optimal_collapse_high = d.get("optimal_collapse_high", 0.20)
            self._optimal_s_threshold = d.get("optimal_s_threshold", 4.0)
            self._optimal_exit_mult = d.get("optimal_exit_mult", 10.0)
            self._optimal_min_time = d.get("optimal_min_time", 0.20)
            logger.info(f"Cross model loaded from {path}")
