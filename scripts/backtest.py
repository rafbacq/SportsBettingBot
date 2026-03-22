#!/usr/bin/env python3
"""Backtest the trading system on historical data.

Replays historical probability curves through the full pipeline
and reports performance metrics.

Usage:
    python scripts/backtest.py --data-dir data/
    python scripts/backtest.py --synthetic --max-samples 500
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logging_config import load_config, setup_logging
from src.data.csv_loader import load_games_from_csv
from src.data.models import Regime, TradeStatus
from src.execution.portfolio import PortfolioManager
from src.execution.risk import RiskManager
from src.features.engine import FeatureEngine
from src.ml.dataset import SyntheticDataGenerator
from src.strategy.regime_router import RegimeRouter
from src.strategy.non_cross import NonCrossStrategy
from src.strategy.cross import CrossStrategy
from src.data.models import NonCrossParams, CrossParams


def main():
    parser = argparse.ArgumentParser(description="Backtest trading system")
    parser.add_argument("--data-dir", default="data", help="Directory with CSV data")
    parser.add_argument("--model-dir", default=None, help="Model directory")
    parser.add_argument("--config", default="config/settings.yaml", help="Config file")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--max-samples", type=int, default=500)
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config)

    # Load data
    if args.synthetic:
        gen = SyntheticDataGenerator(seed=42)
        games = gen.generate(args.max_samples)
    else:
        games = load_games_from_csv(args.data_dir)
        if not games:
            print("No data found. Use --synthetic for testing.")
            return

    # Initialize components
    router = RegimeRouter(config)
    if args.model_dir:
        router.load_models(args.model_dir)
    else:
        try:
            router.load_models()
        except Exception:
            print("No trained models found. Using default parameters.")

    portfolio = PortfolioManager(config)
    risk = RiskManager(config)
    nc_strat = NonCrossStrategy()
    cr_strat = CrossStrategy()
    feature_engine = FeatureEngine()

    # Backtest metrics
    total_trades = 0
    wins = 0
    total_pnl = 0.0
    trade_log = []

    print(f"\nBacktesting on {len(games)} games...")
    print("-" * 50)

    for game in games:
        snapshots = game.curve.snapshots
        if len(snapshots) < 10:
            continue

        active_trade = None

        for idx in range(5, len(snapshots)):
            # Simulate: set curve to only include snapshots up to idx
            from src.data.models import GameState, ProbabilityCurve, ProbabilitySnapshot
            sim_game = GameState(
                game_id=game.game_id, sport=game.sport,
                team_a=game.team_a, team_b=game.team_b,
                start_time=game.start_time,
                total_duration_est=game.total_duration_est,
                kalshi_ticker=game.kalshi_ticker,
            )
            for s in snapshots[:idx + 1]:
                sim_game.add_probability(s.timestamp, s.prob_a)

            current_prob = snapshots[idx].prob_a
            features = feature_engine.compute(sim_game)

            # Check exit
            if active_trade:
                should_exit = False
                if active_trade["regime"] == Regime.NON_CROSS:
                    weak_p = features.prob_b_current if features.is_team_a_favorite else features.prob_a_current
                    params = NonCrossParams(exit_multiplier=active_trade["exit_mult"])
                    should_exit = nc_strat.should_exit(weak_p, active_trade["entry_prob"], params)
                else:
                    strong_p = features.prob_a_current if features.is_team_a_favorite else features.prob_b_current
                    params = CrossParams(exit_multiplier=active_trade["exit_mult"])
                    should_exit = cr_strat.should_exit(strong_p, active_trade["entry_prob"], params)

                if should_exit or idx == len(snapshots) - 1:
                    exit_prob = current_prob
                    mult = exit_prob / max(active_trade["entry_prob"], 1e-6)
                    pnl = mult - 1.0
                    total_pnl += pnl
                    total_trades += 1
                    if pnl > 0:
                        wins += 1
                    trade_log.append({
                        "game": game.game_id, "regime": active_trade["regime"].value,
                        "entry": active_trade["entry_prob"], "exit": exit_prob,
                        "mult": mult, "pnl": pnl,
                    })
                    active_trade = None

            # Check entry (only if no active trade)
            if not active_trade:
                signal = router.evaluate(sim_game)
                if signal:
                    active_trade = {
                        "regime": signal.regime,
                        "entry_prob": signal.entry_prob,
                        "exit_mult": signal.exit_multiplier,
                        "entry_idx": idx,
                    }

    # Results
    print("\n" + "=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)
    print(f"Games tested:     {len(games)}")
    print(f"Total trades:     {total_trades}")
    print(f"Wins:             {wins}")
    print(f"Win rate:         {wins / max(total_trades, 1):.1%}")
    print(f"Total P&L:        {total_pnl:+.3f}")
    print(f"Avg P&L/trade:    {total_pnl / max(total_trades, 1):+.4f}")

    nc_trades = [t for t in trade_log if t["regime"] == "non_cross"]
    cr_trades = [t for t in trade_log if t["regime"] == "cross"]
    print(f"\nNon-Cross trades: {len(nc_trades)}")
    if nc_trades:
        nc_pnl = sum(t["pnl"] for t in nc_trades)
        nc_wins = sum(1 for t in nc_trades if t["pnl"] > 0)
        print(f"  Win rate: {nc_wins / len(nc_trades):.1%} | P&L: {nc_pnl:+.3f}")

    print(f"Cross trades:     {len(cr_trades)}")
    if cr_trades:
        cr_pnl = sum(t["pnl"] for t in cr_trades)
        cr_wins = sum(1 for t in cr_trades if t["pnl"] > 0)
        print(f"  Win rate: {cr_wins / len(cr_trades):.1%} | P&L: {cr_pnl:+.3f}")


if __name__ == "__main__":
    main()
