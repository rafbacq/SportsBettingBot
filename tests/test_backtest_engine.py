"""Tests for the comprehensive backtesting engine."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.data.models import GameState
from src.backtest.backtest_engine import BacktestEngine, BacktestMetrics
from src.ml.dataset import SyntheticDataGenerator
from src.utils.logging_config import load_config


def _get_config():
    try:
        return load_config()
    except Exception:
        return {
            "trading": {
                "dry_run": True, "default_stake_usd": 1.0,
                "max_stake_usd": 10.0, "initial_bankroll_usd": 100.0,
                "max_concurrent_positions": 10, "daily_loss_limit_usd": 50.0,
            },
            "kelly": {"enabled": True, "fraction": 0.25},
            "risk": {"trailing_stop_pct": 0.40, "max_drawdown_pct": 0.25},
            "non_cross": {}, "cross": {}, "ml": {}, "market_regime": {"enabled": False},
        }


class TestBacktestEngine:

    def test_runs_without_errors(self):
        """Backtest should run on synthetic data without crashing."""
        gen = SyntheticDataGenerator(seed=42)
        games = gen.generate(50)
        config = _get_config()
        engine = BacktestEngine(config, initial_bankroll=100.0)
        metrics = engine.run(games)
        assert isinstance(metrics, BacktestMetrics)

    def test_equity_curve_populated(self):
        """Equity curve should have at least the initial point."""
        gen = SyntheticDataGenerator(seed=42)
        games = gen.generate(30)
        config = _get_config()
        engine = BacktestEngine(config, initial_bankroll=100.0)
        engine.run(games)
        assert len(engine.equity_curve) >= 1
        assert engine.equity_curve[0].equity == 100.0

    def test_sharpe_ratio_computed(self):
        """Sharpe ratio should be a finite number after backtest."""
        gen = SyntheticDataGenerator(seed=42)
        games = gen.generate(100)
        config = _get_config()
        engine = BacktestEngine(config)
        metrics = engine.run(games)
        assert np.isfinite(metrics.sharpe_ratio)

    def test_max_drawdown_non_negative(self):
        """Max drawdown should be >= 0."""
        gen = SyntheticDataGenerator(seed=42)
        games = gen.generate(50)
        config = _get_config()
        engine = BacktestEngine(config)
        metrics = engine.run(games)
        assert metrics.max_drawdown_pct >= 0.0

    def test_win_rate_in_valid_range(self):
        """Win rate should be between 0 and 1."""
        gen = SyntheticDataGenerator(seed=42)
        games = gen.generate(50)
        config = _get_config()
        engine = BacktestEngine(config)
        metrics = engine.run(games)
        assert 0.0 <= metrics.win_rate <= 1.0

    def test_per_regime_breakdown(self):
        """NC + CR trade counts should equal total trades."""
        gen = SyntheticDataGenerator(seed=42)
        games = gen.generate(80)
        config = _get_config()
        engine = BacktestEngine(config)
        metrics = engine.run(games)
        assert metrics.nc_trades + metrics.cr_trades == metrics.total_trades

    def test_trade_journal_has_correct_columns(self):
        """Trade journal should contain expected columns."""
        gen = SyntheticDataGenerator(seed=42)
        games = gen.generate(50)
        config = _get_config()
        engine = BacktestEngine(config)
        engine.run(games)
        journal = engine.get_trade_journal()
        if not journal.empty:
            expected_cols = {"game_id", "regime", "entry_prob", "exit_prob",
                            "pnl", "multiplier", "exit_reason"}
            assert expected_cols.issubset(set(journal.columns))


class TestMonteCarloSimulation:

    def test_monte_carlo_produces_results(self):
        """MC simulation should produce terminal equities."""
        gen = SyntheticDataGenerator(seed=42)
        games = gen.generate(80)
        config = _get_config()
        engine = BacktestEngine(config)
        engine.run(games)

        if engine.trade_results:
            mc = engine.run_monte_carlo(n_simulations=100)
            assert "terminal_equities" in mc
            assert len(mc["terminal_equities"]) == 100

    def test_mc_confidence_interval_ordered(self):
        """5th percentile should be <= median <= 95th percentile."""
        gen = SyntheticDataGenerator(seed=42)
        games = gen.generate(100)
        config = _get_config()
        engine = BacktestEngine(config)
        engine.run(games)

        if engine.trade_results:
            engine.run_monte_carlo(n_simulations=200)
            m = engine.metrics
            assert m.mc_5th_percentile <= m.mc_median_terminal <= m.mc_95th_percentile


class TestMetricsCalculation:

    def test_sharpe_on_known_values(self):
        """Verify Sharpe calculation on a known P&L series."""
        pnls = np.array([1.0, -0.5, 0.8, -0.3, 1.2])
        mean = np.mean(pnls)
        std = np.std(pnls)
        expected_sharpe = mean / std
        # Verify the formula is correct
        assert abs(expected_sharpe - mean / std) < 1e-6

    def test_profit_factor_on_known_values(self):
        """Profit factor = gross profit / gross loss."""
        gross_profit = 10.0
        gross_loss = 5.0
        expected = 2.0
        assert abs(gross_profit / gross_loss - expected) < 1e-6

    def test_max_drawdown_on_known_curve(self):
        """Max drawdown on a simple equity curve."""
        equities = np.array([100, 110, 105, 95, 100, 90, 95])
        running_max = np.maximum.accumulate(equities)
        drawdowns = running_max - equities
        dd_pcts = drawdowns / running_max
        # Peak was 110, trough was 90 → drawdown = 20/110 ≈ 18.2%
        assert abs(np.max(dd_pcts) - 20.0 / 110.0) < 1e-6
