"""Regime router — dispatches game states to the correct strategy.

Routes each game observation through:
1. Feature computation
2. Regime classification (Cross vs Non-Cross)
3. ML parameter optimization
4. Strategy evaluation
"""

from __future__ import annotations

import logging
from typing import Optional

from src.data.models import (
    CrossParams,
    GameState,
    NonCrossParams,
    Regime,
    TradeSignal,
)
from src.features.engine import FeatureEngine, FeatureVector
from src.ml.param_optimizer import CrossParamOptimizer, NonCrossParamOptimizer
from src.ml.regime_classifier import RegimeClassifier
from src.strategy.cross import CrossStrategy
from src.strategy.non_cross import NonCrossStrategy
from src.utils.logging_config import load_config

logger = logging.getLogger("trading.strategy.router")


class RegimeRouter:
    """Routes game states to the appropriate regime strategy."""

    def __init__(
        self,
        config: dict | None = None,
        regime_classifier: RegimeClassifier | None = None,
        nc_optimizer: NonCrossParamOptimizer | None = None,
        cr_optimizer: CrossParamOptimizer | None = None,
    ):
        self.config = config or load_config()

        # ML models
        self.regime_clf = regime_classifier or RegimeClassifier()
        self.nc_optimizer = nc_optimizer or NonCrossParamOptimizer()
        self.cr_optimizer = cr_optimizer or CrossParamOptimizer()

        # Feature engine
        self.feature_engine = FeatureEngine()

        # Strategies
        self.non_cross_strategy = NonCrossStrategy()
        self.cross_strategy = CrossStrategy()

        # Default params (used when ML models aren't loaded)
        nc_cfg = self.config.get("non_cross", {})
        self.default_nc_params = NonCrossParams(
            entry_prob_low=nc_cfg.get("entry_prob_low", 0.01),
            entry_prob_high=nc_cfg.get("entry_prob_high", 0.05),
            op_threshold=nc_cfg.get("op_threshold", 5.0),
            exit_multiplier=nc_cfg.get("exit_multiplier", 6.0),
            min_time_remaining_frac=nc_cfg.get("min_time_remaining_frac", 0.20),
        )

        cr_cfg = self.config.get("cross", {})
        from src.data.models import ExitStrategy
        self.default_cr_params = CrossParams(
            start_prob_low=cr_cfg.get("start_prob_low", 0.60),
            start_prob_high=cr_cfg.get("start_prob_high", 1.00),
            collapse_prob_low=cr_cfg.get("collapse_prob_low", 0.03),
            collapse_prob_high=cr_cfg.get("collapse_prob_high", 0.20),
            s_threshold=cr_cfg.get("s_threshold", 4.0),
            exit_multiplier=cr_cfg.get("exit_multiplier", 10.0),
            exit_strategy=ExitStrategy(cr_cfg.get("exit_strategy", "multiplier")),
            min_time_remaining_frac=cr_cfg.get("min_time_remaining_frac", 0.20),
        )

    def evaluate(self, game: GameState) -> TradeSignal | None:
        """Evaluate a game state for trade signals.

        Pipeline:
        1. Compute features
        2. Classify regime
        3. Get ML-optimized parameters
        4. Evaluate strategy entry conditions
        """
        # Step 1: Features
        features = self.feature_engine.compute(game)

        # Step 2: Classify regime
        regime_result = self.regime_clf.predict(features)
        regime = regime_result["regime"]
        confidence = regime_result["confidence"]

        logger.debug(
            f"[{game.game_id}] regime={regime} (conf={confidence:.2f}) | "
            f"OP={features.op_value:.2f} S={features.s_value:.2f} | "
            f"time_rem={features.time_remaining_frac:.2f}"
        )

        # Step 3 & 4: Route to strategy
        if regime == "non_cross":
            return self._evaluate_non_cross(game, features)
        else:
            return self._evaluate_cross(game, features)

    def _evaluate_non_cross(
        self, game: GameState, features: FeatureVector
    ) -> TradeSignal | None:
        """Evaluate Non-Cross strategy with ML parameters."""
        # Get optimized params
        if self.nc_optimizer.rebound_model is not None:
            params = self.nc_optimizer.predict_params(features)
            ev = self.nc_optimizer.predict_ev(features)
            if ev < 0:
                logger.debug(f"[{game.game_id}] Non-Cross EV={ev:.3f} < 0, skipping")
                return None
        else:
            params = self.default_nc_params

        return self.non_cross_strategy.evaluate_entry(game, features, params)

    def _evaluate_cross(
        self, game: GameState, features: FeatureVector
    ) -> TradeSignal | None:
        """Evaluate Cross strategy with ML parameters."""
        if self.cr_optimizer.rebound_model is not None:
            params = self.cr_optimizer.predict_params(features)
            ev = self.cr_optimizer.predict_ev(features)
            if ev < 0:
                logger.debug(f"[{game.game_id}] Cross EV={ev:.3f} < 0, skipping")
                return None
        else:
            params = self.default_cr_params

        return self.cross_strategy.evaluate_entry(game, features, params)

    def load_models(self, models_dir: str | None = None):
        """Load all ML models from disk."""
        ml_cfg = self.config.get("ml", {})
        base = models_dir or ml_cfg.get("models_dir", "src/ml/models")

        import os
        rc_path = os.path.join(base, ml_cfg.get("regime_classifier", "regime_classifier.joblib"))
        nc_path = os.path.join(base, ml_cfg.get("non_cross_model", "non_cross_params.joblib"))
        cr_path = os.path.join(base, ml_cfg.get("cross_model", "cross_params.joblib"))

        self.regime_clf.load(rc_path)
        self.nc_optimizer.load(nc_path)
        self.cr_optimizer.load(cr_path)
        logger.info("All ML models loaded")
