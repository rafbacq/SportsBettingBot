"""Tests for data layer."""

import sys
import os
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.models import (
    GameState, ProbabilityCurve, ProbabilitySnapshot,
    TradeRecord, TradeStatus, Regime, ExitStrategy,
    NonCrossParams, CrossParams,
)
from src.data.storage import TradingDatabase


class TestGameState:

    def test_add_probability(self):
        game = GameState(
            game_id="G1", sport="NCAAB",
            team_a="A", team_b="B",
            start_time=0, total_duration_est=3600,
            kalshi_ticker="G1",
        )
        game.add_probability(0, 0.60)
        game.add_probability(60, 0.55)

        assert len(game.curve.snapshots) == 2
        assert game.initial_prob_a == 0.60
        assert game.current_prob_a == 0.55
        assert abs(game.current_prob_b - 0.45) < 1e-10

    def test_time_remaining(self):
        game = GameState(
            game_id="G1", sport="ATP",
            team_a="A", team_b="B",
            start_time=0, total_duration_est=1000,
            kalshi_ticker="G1",
        )
        game.add_probability(0, 0.50)
        game.add_probability(500, 0.60)

        assert abs(game.time_remaining_frac - 0.50) < 0.01


class TestNonCrossParams:

    def test_to_dict(self):
        p = NonCrossParams(entry_prob_low=0.02, exit_multiplier=8.0)
        d = p.to_dict()
        assert d["entry_prob_low"] == 0.02
        assert d["exit_multiplier"] == 8.0


class TestStorage:

    def test_store_and_retrieve_trade(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = TradingDatabase(os.path.join(tmpdir, "test.db"))

            trade = TradeRecord(
                trade_id="T1", game_id="G1",
                regime=Regime.NON_CROSS,
                status=TradeStatus.OPEN,
                entry_prob=0.03, stake_usd=1.0,
            )
            db.store_trade(trade)

            trades = db.get_open_trades()
            assert len(trades) == 1
            assert trades[0]["trade_id"] == "T1"
            assert trades[0]["regime"] == "non_cross"

            db.close()
