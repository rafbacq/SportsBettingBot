"""Feature engineering engine.

Computes all features from a GameState needed by the ML models and
trading strategies:
- OP value (Non-Cross regime)
- S value (Cross regime)
- Probability derivatives (rate of change, acceleration)
- Time features (remaining fraction, game phase)
- Momentum indicators (rolling trend)
- Volatility (std dev of recent changes)
- Strength ratio (initial prob disparity)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.data.models import GameState, ProbabilityCurve

logger = logging.getLogger("trading.features")


@dataclass
class FeatureVector:
    """Complete feature set for one game state observation."""
    # Identifiers
    game_id: str = ""
    snapshot_idx: int = 0

    # Raw probabilities
    prob_a_initial: float = 0.5
    prob_b_initial: float = 0.5
    prob_a_current: float = 0.5
    prob_b_current: float = 0.5

    # OP / S values
    op_value: float = 1.0           # P0_weak / Pt_weak  (Non-Cross)
    s_value: float = 1.0            # P0_strong / Pt_strong  (Cross)

    # Probability change features
    prob_change_abs: float = 0.0    # |P0 - Pt| for the collapsing team
    prob_change_pct: float = 0.0    # (P0 - Pt) / P0

    # Derivatives
    prob_derivative: float = 0.0    # rate of change (last 5 snapshots)
    prob_acceleration: float = 0.0  # change in rate of change

    # Time features
    time_remaining_frac: float = 1.0
    game_phase: int = 0             # 0=early, 1=mid, 2=late

    # Momentum
    momentum_5: float = 0.0        # trend over last 5 snapshots
    momentum_10: float = 0.0       # trend over last 10 snapshots
    momentum_20: float = 0.0       # trend over last 20 snapshots

    # Volatility
    volatility_5: float = 0.0      # std dev of last 5 changes
    volatility_10: float = 0.0     # std dev of last 10 changes

    # Strength
    strength_ratio: float = 1.0    # initial_prob_a / initial_prob_b
    is_team_a_favorite: bool = True

    # Sport encoding (one-hot style)
    sport_ncaab: float = 0.0
    sport_atp: float = 0.0
    sport_other: float = 0.0

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for ML model input."""
        return np.array([
            self.prob_a_initial, self.prob_b_initial,
            self.prob_a_current, self.prob_b_current,
            self.op_value, self.s_value,
            self.prob_change_abs, self.prob_change_pct,
            self.prob_derivative, self.prob_acceleration,
            self.time_remaining_frac, float(self.game_phase),
            self.momentum_5, self.momentum_10, self.momentum_20,
            self.volatility_5, self.volatility_10,
            self.strength_ratio, float(self.is_team_a_favorite),
            self.sport_ncaab, self.sport_atp, self.sport_other,
        ], dtype=np.float64)

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "prob_a_initial", "prob_b_initial",
            "prob_a_current", "prob_b_current",
            "op_value", "s_value",
            "prob_change_abs", "prob_change_pct",
            "prob_derivative", "prob_acceleration",
            "time_remaining_frac", "game_phase",
            "momentum_5", "momentum_10", "momentum_20",
            "volatility_5", "volatility_10",
            "strength_ratio", "is_team_a_favorite",
            "sport_ncaab", "sport_atp", "sport_other",
        ]

    def to_dict(self) -> dict:
        return dict(zip(self.feature_names(), self.to_array()))


class FeatureEngine:
    """Computes features from GameState objects."""

    def compute(
        self,
        game: GameState,
        snapshot_idx: int | None = None,
    ) -> FeatureVector:
        """Compute full feature vector for a game at a given snapshot index.
        
        If snapshot_idx is None, uses the latest snapshot.
        """
        curve = game.curve
        if not curve.snapshots:
            logger.warning(f"No snapshots for game {game.game_id}")
            return FeatureVector(game_id=game.game_id)

        if snapshot_idx is None:
            snapshot_idx = len(curve.snapshots) - 1

        snap = curve.snapshots[snapshot_idx]
        snap0 = curve.snapshots[0]

        fv = FeatureVector(
            game_id=game.game_id,
            snapshot_idx=snapshot_idx,
        )

        # ── Raw probabilities ─────────────────────────────────────────
        fv.prob_a_initial = snap0.prob_a
        fv.prob_b_initial = snap0.prob_b
        fv.prob_a_current = snap.prob_a
        fv.prob_b_current = snap.prob_b

        # ── Identify favorite / underdog ──────────────────────────────
        fv.is_team_a_favorite = snap0.prob_a >= snap0.prob_b

        if fv.is_team_a_favorite:
            p0_strong = snap0.prob_a
            pt_strong = snap.prob_a
            p0_weak = snap0.prob_b
            pt_weak = snap.prob_b
        else:
            p0_strong = snap0.prob_b
            pt_strong = snap.prob_b
            p0_weak = snap0.prob_a
            pt_weak = snap.prob_a

        # ── OP value (Non-Cross: weak team collapse) ──────────────────
        fv.op_value = p0_weak / max(pt_weak, 1e-6)

        # ── S value (Cross: strong team collapse) ─────────────────────
        fv.s_value = p0_strong / max(pt_strong, 1e-6)

        # ── Probability change ────────────────────────────────────────
        # For OP/Non-Cross: look at weak team
        fv.prob_change_abs = abs(p0_weak - pt_weak)
        fv.prob_change_pct = (p0_weak - pt_weak) / max(p0_weak, 1e-6)

        # ── Derivatives (using collapsing team's probability) ─────────
        probs = self._extract_prob_series(curve, snapshot_idx, fv.is_team_a_favorite)
        fv.prob_derivative = self._compute_derivative(probs, window=5)
        fv.prob_acceleration = self._compute_acceleration(probs, window=5)

        # ── Time features ─────────────────────────────────────────────
        if game.total_duration_est > 0:
            elapsed = snap.timestamp - snap0.timestamp
            fv.time_remaining_frac = max(0.0, 1.0 - elapsed / game.total_duration_est)
        else:
            fv.time_remaining_frac = 1.0

        if fv.time_remaining_frac > 0.66:
            fv.game_phase = 0   # early
        elif fv.time_remaining_frac > 0.33:
            fv.game_phase = 1   # mid
        else:
            fv.game_phase = 2   # late

        # ── Momentum ──────────────────────────────────────────────────
        fv.momentum_5 = self._compute_momentum(probs, window=5)
        fv.momentum_10 = self._compute_momentum(probs, window=10)
        fv.momentum_20 = self._compute_momentum(probs, window=20)

        # ── Volatility ────────────────────────────────────────────────
        fv.volatility_5 = self._compute_volatility(probs, window=5)
        fv.volatility_10 = self._compute_volatility(probs, window=10)

        # ── Strength ratio ────────────────────────────────────────────
        fv.strength_ratio = snap0.prob_a / max(snap0.prob_b, 1e-6)

        # ── Sport encoding ────────────────────────────────────────────
        sport = game.sport.upper()
        fv.sport_ncaab = 1.0 if sport == "NCAAB" else 0.0
        fv.sport_atp = 1.0 if sport == "ATP" else 0.0
        fv.sport_other = 1.0 if sport not in ("NCAAB", "ATP") else 0.0

        return fv

    # ── Helpers ───────────────────────────────────────────────────────

    def _extract_prob_series(
        self,
        curve: ProbabilityCurve,
        up_to_idx: int,
        is_team_a_favorite: bool,
    ) -> list[float]:
        """Extract the relevant team's probability series up to given index."""
        series = []
        for i in range(up_to_idx + 1):
            snap = curve.snapshots[i]
            # For features, we track the team that is collapsing.
            # For OP features, that's the weak team; for S that's the strong.
            # We track both by just returning one series — caller picks.
            series.append(snap.prob_a if is_team_a_favorite else snap.prob_b)
        return series

    @staticmethod
    def _compute_derivative(series: list[float], window: int = 5) -> float:
        """Average rate of change over last `window` points."""
        if len(series) < 2:
            return 0.0
        w = min(window, len(series))
        recent = series[-w:]
        diffs = [recent[i] - recent[i - 1] for i in range(1, len(recent))]
        return float(np.mean(diffs)) if diffs else 0.0

    @staticmethod
    def _compute_acceleration(series: list[float], window: int = 5) -> float:
        """Change in derivative (second derivative)."""
        if len(series) < 3:
            return 0.0
        w = min(window, len(series))
        recent = series[-w:]
        diffs = [recent[i] - recent[i - 1] for i in range(1, len(recent))]
        if len(diffs) < 2:
            return 0.0
        diffs2 = [diffs[i] - diffs[i - 1] for i in range(1, len(diffs))]
        return float(np.mean(diffs2)) if diffs2 else 0.0

    @staticmethod
    def _compute_momentum(series: list[float], window: int = 5) -> float:
        """Simple linear trend over last `window` points.
        
        Returns slope of best-fit line (positive = rising).
        """
        if len(series) < 2:
            return 0.0
        w = min(window, len(series))
        recent = series[-w:]
        x = np.arange(len(recent), dtype=np.float64)
        y = np.array(recent, dtype=np.float64)
        if np.std(x) == 0:
            return 0.0
        slope = float(np.polyfit(x, y, 1)[0])
        return slope

    @staticmethod
    def _compute_volatility(series: list[float], window: int = 5) -> float:
        """Standard deviation of price changes over last `window` points."""
        if len(series) < 2:
            return 0.0
        w = min(window, len(series))
        recent = series[-w:]
        diffs = [recent[i] - recent[i - 1] for i in range(1, len(recent))]
        return float(np.std(diffs)) if diffs else 0.0
