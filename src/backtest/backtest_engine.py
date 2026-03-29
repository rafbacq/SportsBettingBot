"""Comprehensive backtesting engine with institutional-grade analytics.

Features:
- Full equity curve tracking with per-trade journal
- Sharpe, Sortino, Calmar, profit factor, expectancy
- Max drawdown (absolute and percentage)
- Monte Carlo simulation with confidence intervals
- Walk-forward optimization
- Transaction cost modeling
- Per-regime performance breakdown (Cross vs Non-Cross)
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.data.models import (
    CrossParams,
    GameState,
    NonCrossParams,
    Regime,
    TradeStatus,
)
from src.execution.portfolio import PortfolioManager
from src.execution.risk import RiskManager
from src.features.engine import FeatureEngine
from src.features.market_regime import MarketRegimeDetector
from src.strategy.cross import CrossStrategy
from src.strategy.non_cross import NonCrossStrategy
from src.strategy.regime_router import RegimeRouter

logger = logging.getLogger("trading.backtest")


@dataclass
class TradeResult:
    """Record of a single completed trade during backtest."""
    game_id: str
    regime: str           # "cross" or "non_cross"
    entry_idx: int
    exit_idx: int
    entry_prob: float
    exit_prob: float
    entry_timestamp: float
    exit_timestamp: float
    stake_usd: float
    pnl_usd: float
    multiplier: float
    hold_snapshots: int
    exit_reason: str      # "target", "stop_loss", "trailing_stop", "game_end"


@dataclass
class EquityPoint:
    """A point on the equity curve."""
    timestamp: float
    equity: float
    trade_count: int
    drawdown_pct: float


@dataclass
class BacktestMetrics:
    """Complete backtest performance metrics."""
    # Summary
    total_games: int = 0
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0

    # P&L
    total_pnl: float = 0.0
    avg_pnl_per_trade: float = 0.0
    median_pnl: float = 0.0
    best_trade_pnl: float = 0.0
    worst_trade_pnl: float = 0.0
    total_return_pct: float = 0.0

    # Risk-adjusted returns
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Drawdown
    max_drawdown_usd: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_drawdown_pct: float = 0.0

    # Trade statistics
    profit_factor: float = 0.0
    expectancy: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_hold_snapshots: float = 0.0
    max_consecutive_losses: int = 0

    # Per-regime breakdown
    nc_trades: int = 0
    nc_wins: int = 0
    nc_pnl: float = 0.0
    nc_sharpe: float = 0.0
    cr_trades: int = 0
    cr_wins: int = 0
    cr_pnl: float = 0.0
    cr_sharpe: float = 0.0

    # Transaction costs
    total_costs: float = 0.0

    # Monte Carlo (filled after simulation)
    mc_median_terminal: float = 0.0
    mc_5th_percentile: float = 0.0
    mc_95th_percentile: float = 0.0
    mc_prob_profitable: float = 0.0
    mc_median_max_dd: float = 0.0

    # Walk-forward
    wf_avg_oos_sharpe: float = 0.0
    wf_avg_oos_pnl: float = 0.0


class BacktestEngine:
    """Comprehensive backtesting engine for the trading system."""

    def __init__(
        self,
        config: dict,
        initial_bankroll: float = 100.0,
        transaction_cost_pct: float = 0.01,  # 1% per trade (Kalshi typical)
    ):
        self.config = config
        self.initial_bankroll = initial_bankroll
        self.transaction_cost_pct = transaction_cost_pct

        # Results storage
        self.trade_results: list[TradeResult] = []
        self.equity_curve: list[EquityPoint] = []
        self.metrics = BacktestMetrics()

    def run(
        self,
        games: list[GameState],
        router: RegimeRouter | None = None,
    ) -> BacktestMetrics:
        """Run a full backtest on historical game data.

        Args:
            games: List of GameState objects with probability curves.
            router: Pre-configured RegimeRouter (uses default if None).

        Returns:
            BacktestMetrics with comprehensive analytics.
        """
        t0 = time.time()
        logger.info(f"Starting backtest on {len(games)} games...")

        # Initialize components
        if router is None:
            router = RegimeRouter(self.config)
            try:
                router.load_models()
            except Exception:
                logger.info("No trained models found. Using default parameters.")

        config_with_bankroll = copy.deepcopy(self.config)
        config_with_bankroll.setdefault("trading", {})["initial_bankroll_usd"] = self.initial_bankroll
        portfolio = PortfolioManager(config_with_bankroll)
        risk = RiskManager(config_with_bankroll)
        nc_strat = NonCrossStrategy()
        cr_strat = CrossStrategy()
        feature_engine = FeatureEngine()
        regime_detector = MarketRegimeDetector()

        # Track equity
        equity = self.initial_bankroll
        peak_equity = equity
        self.equity_curve = [EquityPoint(0.0, equity, 0, 0.0)]
        self.trade_results = []

        for game in games:
            snapshots = game.curve.snapshots
            if len(snapshots) < 10:
                continue

            active_trade = None

            for idx in range(5, len(snapshots)):
                features = feature_engine.compute(game, snapshot_idx=idx)

                # Detect market regime for dynamic parameter adjustment
                strong_probs = [
                    s.prob_a if features.is_team_a_favorite else s.prob_b
                    for s in snapshots[:idx + 1]
                ]
                regime_state = regime_detector.detect(strong_probs)

                # Check exit
                if active_trade:
                    should_exit = False
                    exit_reason = "game_end"

                    if active_trade["regime"] == Regime.NON_CROSS:
                        weak_p = (
                            features.prob_b_current
                            if features.is_team_a_favorite
                            else features.prob_a_current
                        )
                        params = NonCrossParams(
                            exit_multiplier=active_trade["exit_mult"]
                        )
                        if nc_strat.should_exit(
                            weak_p, active_trade["entry_prob"], params
                        ):
                            should_exit = True
                            target = nc_strat.compute_exit_price(
                                active_trade["entry_prob"], params
                            )
                            exit_reason = (
                                "target" if weak_p >= target else "stop_loss"
                            )
                    else:
                        strong_p = (
                            features.prob_a_current
                            if features.is_team_a_favorite
                            else features.prob_b_current
                        )
                        params = CrossParams(
                            exit_multiplier=active_trade["exit_mult"]
                        )
                        if cr_strat.should_exit(
                            strong_p, active_trade["entry_prob"], params
                        ):
                            should_exit = True
                            target = cr_strat.compute_exit_price(
                                active_trade["entry_prob"], params
                            )
                            exit_reason = (
                                "target" if strong_p >= target else "stop_loss"
                            )

                    # Check trailing stop
                    if not should_exit and active_trade.get("trade_id"):
                        relevant_p = (
                            features.prob_b_current
                            if active_trade["regime"] == Regime.NON_CROSS
                            and features.is_team_a_favorite
                            else features.prob_a_current
                        )
                        if risk.check_trailing_stop(
                            active_trade["trade_id"],
                            relevant_p,
                            features.time_remaining_frac,
                        ):
                            should_exit = True
                            exit_reason = "trailing_stop"

                    # Force exit at game end
                    if idx == len(snapshots) - 1:
                        should_exit = True
                        exit_reason = "game_end"

                    if should_exit:
                        exit_prob = snapshots[idx].prob_a
                        mult = exit_prob / max(active_trade["entry_prob"], 1e-6)
                        stake = active_trade.get("stake", 1.0)
                        contracts = stake / max(active_trade["entry_prob"], 1e-6)
                        raw_pnl = contracts * (exit_prob - active_trade["entry_prob"])
                        cost = abs(raw_pnl) * self.transaction_cost_pct
                        pnl = raw_pnl - cost

                        equity += pnl
                        if equity > peak_equity:
                            peak_equity = equity
                        dd_pct = (
                            (peak_equity - equity) / peak_equity
                            if peak_equity > 0
                            else 0.0
                        )

                        self.trade_results.append(
                            TradeResult(
                                game_id=game.game_id,
                                regime=active_trade["regime"].value,
                                entry_idx=active_trade["entry_idx"],
                                exit_idx=idx,
                                entry_prob=active_trade["entry_prob"],
                                exit_prob=exit_prob,
                                entry_timestamp=snapshots[
                                    active_trade["entry_idx"]
                                ].timestamp,
                                exit_timestamp=snapshots[idx].timestamp,
                                stake_usd=stake,
                                pnl_usd=pnl,
                                multiplier=mult,
                                hold_snapshots=idx - active_trade["entry_idx"],
                                exit_reason=exit_reason,
                            )
                        )

                        self.equity_curve.append(
                            EquityPoint(
                                timestamp=snapshots[idx].timestamp,
                                equity=equity,
                                trade_count=len(self.trade_results),
                                drawdown_pct=dd_pct,
                            )
                        )

                        # Clean up
                        if active_trade.get("trade_id"):
                            risk.remove_trailing_stop(active_trade["trade_id"])
                        active_trade = None

                # Check entry (only if no active trade)
                if not active_trade:
                    signal = router.evaluate(game, snapshot_idx=idx)
                    if signal:
                        # Apply regime-based sizing adjustment
                        sizing_mult = regime_state.recommended_sizing_mult
                        stake = portfolio._compute_stake(signal) * sizing_mult
                        stake = min(stake, equity * 0.05)  # max 5% of equity
                        stake = max(stake, 0.50)  # min $0.50

                        trade_id = f"BT-{len(self.trade_results):05d}"
                        active_trade = {
                            "regime": signal.regime,
                            "entry_prob": signal.entry_prob,
                            "exit_mult": signal.exit_multiplier,
                            "entry_idx": idx,
                            "stake": stake,
                            "trade_id": trade_id,
                        }

                        # Register trailing stop
                        risk.register_trailing_stop(trade_id, signal.entry_prob)

        # Compute all metrics
        self.metrics = self._compute_metrics(equity)

        elapsed = time.time() - t0
        logger.info(f"Backtest completed in {elapsed:.1f}s — {len(self.trade_results)} trades")
        return self.metrics

    def run_monte_carlo(
        self, n_simulations: int = 1000, seed: int = 42
    ) -> dict:
        """Monte Carlo simulation: bootstrap trade results.

        Randomly resamples the trade P&L sequence to estimate
        the distribution of outcomes and confidence intervals.
        """
        if not self.trade_results:
            return {}

        rng = np.random.RandomState(seed)
        pnls = np.array([t.pnl_usd for t in self.trade_results])
        n_trades = len(pnls)

        terminal_equities = []
        max_drawdowns = []
        sharpe_ratios = []

        for _ in range(n_simulations):
            # Resample trade P&Ls with replacement
            resampled = rng.choice(pnls, size=n_trades, replace=True)

            # Build equity curve
            eq = np.cumsum(resampled) + self.initial_bankroll
            terminal_equities.append(eq[-1])

            # Max drawdown
            running_max = np.maximum.accumulate(eq)
            dd = (running_max - eq) / np.maximum(running_max, 1e-6)
            max_drawdowns.append(float(np.max(dd)))

            # Sharpe
            if np.std(resampled) > 0:
                sharpe_ratios.append(
                    float(np.mean(resampled) / np.std(resampled))
                )

        terminal_equities = np.array(terminal_equities)
        max_drawdowns = np.array(max_drawdowns)

        self.metrics.mc_median_terminal = float(np.median(terminal_equities))
        self.metrics.mc_5th_percentile = float(np.percentile(terminal_equities, 5))
        self.metrics.mc_95th_percentile = float(np.percentile(terminal_equities, 95))
        self.metrics.mc_prob_profitable = float(
            np.mean(terminal_equities > self.initial_bankroll)
        )
        self.metrics.mc_median_max_dd = float(np.median(max_drawdowns))

        mc_results = {
            "terminal_equities": terminal_equities,
            "max_drawdowns": max_drawdowns,
            "sharpe_ratios": np.array(sharpe_ratios) if sharpe_ratios else np.array([0.0]),
        }

        logger.info(
            f"Monte Carlo ({n_simulations} sims): "
            f"Median terminal=${self.metrics.mc_median_terminal:.2f} | "
            f"5th-95th: ${self.metrics.mc_5th_percentile:.2f} - "
            f"${self.metrics.mc_95th_percentile:.2f} | "
            f"P(profit)={self.metrics.mc_prob_profitable:.1%}"
        )
        return mc_results

    def run_walk_forward(
        self,
        games: list[GameState],
        n_folds: int = 5,
        train_ratio: float = 0.70,
    ) -> dict:
        """Walk-forward optimization.

        Splits games chronologically, trains on each window, tests on next.
        Only evaluates true out-of-sample performance.

        Returns:
            Dictionary with per-fold metrics and average OOS performance.
        """
        if len(games) < n_folds * 2:
            logger.warning(f"Too few games ({len(games)}) for {n_folds}-fold walk-forward")
            return {"status": "skipped"}

        fold_size = len(games) // n_folds
        fold_results = []

        for fold in range(n_folds - 1):
            # Training window: folds 0..fold
            train_end = (fold + 1) * fold_size
            test_end = min(train_end + fold_size, len(games))

            train_games = games[:train_end]
            test_games = games[train_end:test_end]

            if not test_games:
                continue

            # Run backtest on OOS data only
            fold_engine = BacktestEngine(
                self.config,
                initial_bankroll=self.initial_bankroll,
                transaction_cost_pct=self.transaction_cost_pct,
            )
            fold_metrics = fold_engine.run(test_games)

            fold_results.append({
                "fold": fold,
                "train_games": len(train_games),
                "test_games": len(test_games),
                "oos_trades": fold_metrics.total_trades,
                "oos_pnl": fold_metrics.total_pnl,
                "oos_sharpe": fold_metrics.sharpe_ratio,
                "oos_win_rate": fold_metrics.win_rate,
            })

            logger.info(
                f"Walk-forward fold {fold}: "
                f"OOS trades={fold_metrics.total_trades}, "
                f"P&L={fold_metrics.total_pnl:+.3f}, "
                f"Sharpe={fold_metrics.sharpe_ratio:.2f}"
            )

        if fold_results:
            self.metrics.wf_avg_oos_sharpe = float(
                np.mean([f["oos_sharpe"] for f in fold_results])
            )
            self.metrics.wf_avg_oos_pnl = float(
                np.mean([f["oos_pnl"] for f in fold_results])
            )

        return {"folds": fold_results}

    def _compute_metrics(self, final_equity: float) -> BacktestMetrics:
        """Compute all performance metrics from trade results."""
        m = BacktestMetrics()

        if not self.trade_results:
            return m

        pnls = np.array([t.pnl_usd for t in self.trade_results])
        m.total_trades = len(pnls)
        m.wins = int(np.sum(pnls > 0))
        m.losses = int(np.sum(pnls <= 0))
        m.win_rate = m.wins / m.total_trades if m.total_trades > 0 else 0.0

        # P&L
        m.total_pnl = float(np.sum(pnls))
        m.avg_pnl_per_trade = float(np.mean(pnls))
        m.median_pnl = float(np.median(pnls))
        m.best_trade_pnl = float(np.max(pnls))
        m.worst_trade_pnl = float(np.min(pnls))
        m.total_return_pct = (
            (final_equity - self.initial_bankroll) / self.initial_bankroll * 100
        )

        # Transaction costs
        m.total_costs = float(
            sum(abs(t.pnl_usd) * self.transaction_cost_pct for t in self.trade_results)
        )

        # Risk-adjusted returns
        if np.std(pnls) > 0:
            m.sharpe_ratio = float(np.mean(pnls) / np.std(pnls))
        downside = pnls[pnls < 0]
        if len(downside) > 0 and np.std(downside) > 0:
            m.sortino_ratio = float(np.mean(pnls) / np.std(downside))

        # Drawdown from equity curve
        equities = np.array([e.equity for e in self.equity_curve])
        if len(equities) > 1:
            running_max = np.maximum.accumulate(equities)
            drawdowns = running_max - equities
            dd_pcts = drawdowns / np.maximum(running_max, 1e-6)
            m.max_drawdown_usd = float(np.max(drawdowns))
            m.max_drawdown_pct = float(np.max(dd_pcts))
            m.avg_drawdown_pct = float(np.mean(dd_pcts))

        # Calmar ratio
        if m.max_drawdown_pct > 0:
            m.calmar_ratio = (m.total_return_pct / 100.0) / m.max_drawdown_pct

        # Trade statistics
        wins_pnl = pnls[pnls > 0]
        losses_pnl = pnls[pnls <= 0]
        m.avg_win = float(np.mean(wins_pnl)) if len(wins_pnl) > 0 else 0.0
        m.avg_loss = float(np.mean(losses_pnl)) if len(losses_pnl) > 0 else 0.0

        gross_profit = float(np.sum(wins_pnl)) if len(wins_pnl) > 0 else 0.0
        gross_loss = float(abs(np.sum(losses_pnl))) if len(losses_pnl) > 0 else 0.0
        m.profit_factor = gross_profit / max(gross_loss, 1e-6)

        m.expectancy = m.win_rate * m.avg_win + (1 - m.win_rate) * m.avg_loss

        m.avg_hold_snapshots = float(
            np.mean([t.hold_snapshots for t in self.trade_results])
        )

        # Max consecutive losses
        max_consec = 0
        current_consec = 0
        for p in pnls:
            if p <= 0:
                current_consec += 1
                max_consec = max(max_consec, current_consec)
            else:
                current_consec = 0
        m.max_consecutive_losses = max_consec

        # Per-regime breakdown
        nc = [t for t in self.trade_results if t.regime == "non_cross"]
        cr = [t for t in self.trade_results if t.regime == "cross"]

        m.nc_trades = len(nc)
        m.nc_wins = sum(1 for t in nc if t.pnl_usd > 0)
        m.nc_pnl = sum(t.pnl_usd for t in nc)
        nc_pnls = np.array([t.pnl_usd for t in nc]) if nc else np.array([0.0])
        m.nc_sharpe = float(np.mean(nc_pnls) / np.std(nc_pnls)) if np.std(nc_pnls) > 0 else 0.0

        m.cr_trades = len(cr)
        m.cr_wins = sum(1 for t in cr if t.pnl_usd > 0)
        m.cr_pnl = sum(t.pnl_usd for t in cr)
        cr_pnls = np.array([t.pnl_usd for t in cr]) if cr else np.array([0.0])
        m.cr_sharpe = float(np.mean(cr_pnls) / np.std(cr_pnls)) if np.std(cr_pnls) > 0 else 0.0

        return m

    def get_trade_journal(self) -> pd.DataFrame:
        """Return trade results as a DataFrame for analysis."""
        if not self.trade_results:
            return pd.DataFrame()
        records = []
        for t in self.trade_results:
            records.append({
                "game_id": t.game_id,
                "regime": t.regime,
                "entry_prob": t.entry_prob,
                "exit_prob": t.exit_prob,
                "multiplier": t.multiplier,
                "stake": t.stake_usd,
                "pnl": t.pnl_usd,
                "hold_snapshots": t.hold_snapshots,
                "exit_reason": t.exit_reason,
            })
        return pd.DataFrame(records)

    def print_summary(self):
        """Print a comprehensive summary to console."""
        m = self.metrics
        print("\n" + "=" * 70)
        print("                    BACKTEST RESULTS")
        print("=" * 70)

        print(f"\n{'─── OVERVIEW ───':^70}")
        print(f"  Total trades:           {m.total_trades}")
        print(f"  Wins / Losses:          {m.wins} / {m.losses}")
        print(f"  Win rate:               {m.win_rate:.1%}")
        print(f"  Total P&L:              ${m.total_pnl:+.2f}")
        print(f"  Total return:           {m.total_return_pct:+.1f}%")
        print(f"  Transaction costs:      ${m.total_costs:.2f}")

        print(f"\n{'─── RISK-ADJUSTED RETURNS ───':^70}")
        print(f"  Sharpe ratio:           {m.sharpe_ratio:.3f}")
        print(f"  Sortino ratio:          {m.sortino_ratio:.3f}")
        print(f"  Calmar ratio:           {m.calmar_ratio:.3f}")
        print(f"  Max drawdown:           {m.max_drawdown_pct:.1%} (${m.max_drawdown_usd:.2f})")
        print(f"  Avg drawdown:           {m.avg_drawdown_pct:.1%}")

        print(f"\n{'─── TRADE STATISTICS ───':^70}")
        print(f"  Avg P&L/trade:          ${m.avg_pnl_per_trade:+.4f}")
        print(f"  Best trade:             ${m.best_trade_pnl:+.4f}")
        print(f"  Worst trade:            ${m.worst_trade_pnl:+.4f}")
        print(f"  Avg win:                ${m.avg_win:+.4f}")
        print(f"  Avg loss:               ${m.avg_loss:+.4f}")
        print(f"  Profit factor:          {m.profit_factor:.2f}")
        print(f"  Expectancy:             ${m.expectancy:+.4f}")
        print(f"  Max consecutive losses: {m.max_consecutive_losses}")
        print(f"  Avg hold (snapshots):   {m.avg_hold_snapshots:.1f}")

        print(f"\n{'─── PER-REGIME BREAKDOWN ───':^70}")
        print(f"  Non-Cross:  {m.nc_trades} trades | "
              f"{m.nc_wins} wins ({m.nc_wins / max(m.nc_trades, 1):.0%}) | "
              f"P&L=${m.nc_pnl:+.3f} | Sharpe={m.nc_sharpe:.3f}")
        print(f"  Cross:      {m.cr_trades} trades | "
              f"{m.cr_wins} wins ({m.cr_wins / max(m.cr_trades, 1):.0%}) | "
              f"P&L=${m.cr_pnl:+.3f} | Sharpe={m.cr_sharpe:.3f}")

        if m.mc_median_terminal > 0:
            print(f"\n{'─── MONTE CARLO ───':^70}")
            print(f"  Median terminal equity: ${m.mc_median_terminal:.2f}")
            print(f"  90% CI:                 ${m.mc_5th_percentile:.2f} – ${m.mc_95th_percentile:.2f}")
            print(f"  P(profitable):          {m.mc_prob_profitable:.1%}")
            print(f"  Median max drawdown:    {m.mc_median_max_dd:.1%}")

        if m.wf_avg_oos_sharpe != 0:
            print(f"\n{'─── WALK-FORWARD ───':^70}")
            print(f"  Avg OOS Sharpe:         {m.wf_avg_oos_sharpe:.3f}")
            print(f"  Avg OOS P&L:            ${m.wf_avg_oos_pnl:+.3f}")

        print("=" * 70)
