"""Tests for trading strategies."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.models import (
    CrossParams, ExitStrategy, GameState, NonCrossParams, Regime,
)
from src.features.engine import FeatureEngine
from src.strategy.non_cross import NonCrossStrategy
from src.strategy.cross import CrossStrategy


def make_game(probs: list[float], sport: str = "NCAAB") -> GameState:
    game = GameState(
        game_id="TEST-001", sport=sport,
        team_a="Favorites", team_b="Underdogs",
        start_time=0, total_duration_est=len(probs) * 24.0,
        kalshi_ticker="TEST-001",
    )
    for i, p in enumerate(probs):
        game.add_probability(float(i * 24), p)
    return game


class TestNonCrossStrategy:

    def test_entry_signal_generated(self):
        """Should generate signal when weak team prob drops into entry range."""
        # Favorite starts at 80%, climbs to 97% (underdog at 3%)
        probs = [0.80, 0.82, 0.85, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97]
        game = make_game(probs)
        engine = FeatureEngine()
        features = engine.compute(game)

        params = NonCrossParams(
            entry_prob_low=0.01, entry_prob_high=0.10,
            op_threshold=2.0, exit_multiplier=4.0,
            min_time_remaining_frac=0.0,
        )

        strategy = NonCrossStrategy()
        signal = strategy.evaluate_entry(game, features, params)

        # Underdog is at 3%, which is in [1%, 10%]
        assert signal is not None
        assert signal.regime == Regime.NON_CROSS

    def test_no_entry_when_prob_too_high(self):
        """Should NOT generate signal when prob is above entry range."""
        probs = [0.60, 0.62, 0.65, 0.63, 0.64, 0.62, 0.61, 0.60, 0.59, 0.58]
        game = make_game(probs)
        engine = FeatureEngine()
        features = engine.compute(game)

        params = NonCrossParams(
            entry_prob_low=0.01, entry_prob_high=0.05,
            op_threshold=5.0, exit_multiplier=6.0,
            min_time_remaining_frac=0.20,
        )

        strategy = NonCrossStrategy()
        signal = strategy.evaluate_entry(game, features, params)
        assert signal is None

    def test_exit_below_50(self):
        """Exit price must remain below 50%."""
        strategy = NonCrossStrategy()
        params = NonCrossParams(exit_multiplier=20.0)
        exit_p = strategy.compute_exit_price(0.03, params)
        assert exit_p < 0.50

    def test_should_exit_at_target(self):
        """Should exit when target is hit."""
        strategy = NonCrossStrategy()
        params = NonCrossParams(exit_multiplier=4.0)
        # Entry at 3%, target at 12%
        assert strategy.should_exit(0.12, 0.03, params)
        # Not yet
        assert not strategy.should_exit(0.08, 0.03, params)


class TestCrossStrategy:

    def test_entry_signal_generated(self):
        """Should generate signal when strong team collapses."""
        # Strong team starts at 75%, drops to 15%
        probs = [0.75, 0.65, 0.50, 0.40, 0.30, 0.25, 0.20, 0.18, 0.16, 0.15]
        game = make_game(probs)
        engine = FeatureEngine()
        features = engine.compute(game)

        params = CrossParams(
            start_prob_low=0.60, start_prob_high=1.00,
            collapse_prob_low=0.03, collapse_prob_high=0.20,
            s_threshold=2.0, exit_multiplier=5.0,
            min_time_remaining_frac=0.0,
        )

        strategy = CrossStrategy()
        signal = strategy.evaluate_entry(game, features, params)
        assert signal is not None
        assert signal.regime == Regime.CROSS

    def test_full_hold_exits_only_on_game_end(self):
        """Full hold should not exit until game ends."""
        strategy = CrossStrategy()
        params = CrossParams(exit_strategy=ExitStrategy.FULL_HOLD)

        assert not strategy.should_exit(0.80, 0.10, params, game_ended=False)
        assert strategy.should_exit(0.80, 0.10, params, game_ended=True)

    def test_multiplier_exit(self):
        """Multiplier exit should trigger at m * entry."""
        strategy = CrossStrategy()
        params = CrossParams(exit_multiplier=5.0, exit_strategy=ExitStrategy.MULTIPLIER)

        # Entry at 10%, target at 50%
        assert strategy.should_exit(0.50, 0.10, params)
        assert not strategy.should_exit(0.30, 0.10, params)

    def test_stop_loss(self):
        """Stop loss should trigger when prob drops below 50% of entry."""
        strategy = CrossStrategy()
        params = CrossParams(exit_multiplier=5.0, exit_strategy=ExitStrategy.MULTIPLIER)

        # Entry at 10%, stop at 5%
        assert strategy.should_exit(0.04, 0.10, params)
