#!/usr/bin/env python3
"""
Analyze CSV — Feed in live Kalshi data, get trade recommendations
==================================================================

Accepts CSV files in the Kalshi scraper format (candlesticks.csv or
decision_features.csv) and outputs clear BUY / NO TRADE for each market.

Usage:
    python scripts/analyze_csv.py path/to/candlesticks.csv
    python scripts/analyze_csv.py path/to/decision_features.csv
    python scripts/analyze_csv.py path/to/data_folder/

    # Show top N markets only
    python scripts/analyze_csv.py data.csv --top 5

    # Stricter entry filters
    python scripts/analyze_csv.py data.csv --min-ev 1.0 --min-rebound 0.6
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ════════════════════════════════════════════════════════════════════════
# Formatting
# ════════════════════════════════════════════════════════════════════════

B = "\033[1m"
G = "\033[92m"
R = "\033[91m"
Y = "\033[93m"
C = "\033[96m"
M = "\033[95m"
D = "\033[2m"
X = "\033[0m"
LINE = "─" * 72


def color_pnl(v):
    return G if v > 0 else R if v < 0 else ""


# ════════════════════════════════════════════════════════════════════════
# Load models
# ════════════════════════════════════════════════════════════════════════

def load_models(models_dir: str = "src/ml/models") -> dict:
    models = {}
    names = ["regime_classifier", "rebound_classifier", "multiplier_regressor", "ev_estimator"]
    for name in names:
        path = os.path.join(models_dir, f"{name}.joblib")
        if os.path.exists(path):
            models[name] = joblib.load(path)
        else:
            print(f"  {R}✗ Missing model: {path}{X}")
            sys.exit(1)
    return models


# ════════════════════════════════════════════════════════════════════════
# CSV Ingestion — auto-detect format
# ════════════════════════════════════════════════════════════════════════

def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV or an entire directory of CSVs. Auto-detects format."""

    if os.path.isdir(path):
        # Try decision_features first, then candlesticks
        for fname in ["decision_features.csv", "candlesticks.csv"]:
            fpath = os.path.join(path, fname)
            if os.path.exists(fpath):
                df = pd.read_csv(fpath)
                print(f"  Loaded {fpath} — {len(df):,} rows, {df['ticker'].nunique()} markets")
                return df
        # Fall back to any CSV in the directory
        for f in sorted(os.listdir(path)):
            if f.endswith(".csv"):
                df = pd.read_csv(os.path.join(path, f))
                if "ticker" in df.columns:
                    print(f"  Loaded {os.path.join(path, f)} — {len(df):,} rows")
                    return df
        print(f"  {R}No usable CSV found in {path}{X}")
        sys.exit(1)
    else:
        df = pd.read_csv(path)
        print(f"  Loaded {path} — {len(df):,} rows, {df['ticker'].nunique()} markets")
        return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute model features regardless of input CSV format.

    Handles both candlesticks.csv and decision_features.csv.
    """
    df = df.copy()

    # Ensure we have implied_prob
    if "implied_prob" not in df.columns:
        if "price_close" in df.columns:
            df["implied_prob"] = df["price_close"]
        elif "yes_price" in df.columns:
            df["implied_prob"] = df["yes_price"]
        else:
            print(f"  {R}Cannot find probability column in CSV{X}")
            sys.exit(1)

    df = df.dropna(subset=["implied_prob"])
    df = df[df["implied_prob"] > 0].copy()

    # Sort by time within each ticker
    ts_col = "end_period_ts"
    if ts_col not in df.columns:
        # Try other timestamp columns
        for candidate in ["created_time", "timestamp"]:
            if candidate in df.columns:
                ts_col = candidate
                break

    df = df.sort_values(["ticker", ts_col]).reset_index(drop=True)

    # ── Compute initial probability per market ──
    initial = (
        df.groupby("ticker")["implied_prob"]
        .first()
        .rename("initial_prob")
    )
    df = df.merge(initial, on="ticker", how="left")

    # ── Market progress ──
    if "market_progress" not in df.columns:
        # Estimate from position within each ticker's time series
        df["market_progress"] = df.groupby("ticker").cumcount() / df.groupby("ticker")["ticker"].transform("count")

    # ── Determine favorite side ──
    df["is_favorite"] = df["initial_prob"] >= 0.50

    # Weak / strong team probabilities
    df["prob_weak"] = np.where(df["is_favorite"], 1 - df["implied_prob"], df["implied_prob"])
    df["prob_strong"] = np.where(df["is_favorite"], df["implied_prob"], 1 - df["implied_prob"])
    df["initial_prob_weak"] = np.where(df["is_favorite"], 1 - df["initial_prob"], df["initial_prob"])
    df["initial_prob_strong"] = np.where(df["is_favorite"], df["initial_prob"], 1 - df["initial_prob"])

    # ── Regime features ──
    df["op_value"] = df["initial_prob_weak"] / df["prob_weak"].clip(lower=0.001)
    df["s_value"] = df["initial_prob_strong"] / df["prob_strong"].clip(lower=0.001)
    df["strength_ratio"] = df["initial_prob"].clip(lower=0.01) / (1 - df["initial_prob"]).clip(lower=0.01)
    df["prob_drop"] = df["initial_prob"] - df["implied_prob"]
    df["prob_drop_pct"] = df["prob_drop"] / df["initial_prob"].clip(lower=0.001)

    # ── Candle returns (if not already present) ──
    if "candle_return_1" not in df.columns:
        df["candle_return_1"] = df.groupby("ticker")["implied_prob"].diff(1).fillna(0)
    if "candle_return_3" not in df.columns:
        df["candle_return_3"] = df.groupby("ticker")["implied_prob"].diff(3).fillna(0)
    if "candle_return_5" not in df.columns:
        df["candle_return_5"] = df.groupby("ticker")["implied_prob"].diff(5).fillna(0)

    # ── Rolling volatility ──
    if "rolling_volatility_5" not in df.columns:
        df["rolling_volatility_5"] = (
            df.groupby("ticker")["implied_prob"]
            .transform(lambda x: x.diff().rolling(5, min_periods=1).std())
            .fillna(0)
        )

    # ── Volume features (0 if not present) ──
    for col in ["volume", "volume_vs_rolling_mean_5", "spread", "relative_spread",
                 "trade_imbalance", "trade_vwap_edge"]:
        if col not in df.columns:
            df[col] = 0.0

    df = df.fillna(0).replace([np.inf, -np.inf], 0)
    return df


# ════════════════════════════════════════════════════════════════════════
# Model prediction
# ════════════════════════════════════════════════════════════════════════

MODEL_FEATURES = [
    "implied_prob", "initial_prob", "market_progress",
    "candle_return_1", "candle_return_3", "candle_return_5",
    "rolling_volatility_5", "volume", "volume_vs_rolling_mean_5",
    "spread", "relative_spread",
    "trade_imbalance", "trade_vwap_edge",
    "op_value", "s_value", "strength_ratio",
    "prob_weak", "prob_strong",
    "prob_drop", "prob_drop_pct",
]


def score_markets(df: pd.DataFrame, models: dict) -> pd.DataFrame:
    """Score every row in the dataframe with all 4 models.

    Returns a per-market summary scored at the LATEST candle.
    """
    feature_names = models["regime_classifier"]["feature_names"]

    # Get the last row per ticker (= current state)
    latest = df.groupby("ticker").tail(1).copy()

    # Build feature matrix
    avail = [c for c in feature_names if c in latest.columns]
    missing = [c for c in feature_names if c not in latest.columns]
    for c in missing:
        latest[c] = 0.0

    X = latest[feature_names].values

    # Regime
    regime_model = models["regime_classifier"]["model"]
    regime_pred = regime_model.predict(X)
    regime_proba = regime_model.predict_proba(X)
    latest["regime"] = np.where(regime_pred == 1, "CROSS", "NON-CROSS")
    latest["regime_confidence"] = np.max(regime_proba, axis=1)

    # Rebound
    rebound_model = models["rebound_classifier"]["model"]
    rebound_pred = rebound_model.predict(X)
    rebound_proba = rebound_model.predict_proba(X)[:, 1]
    latest["rebound_prob"] = rebound_proba
    latest["will_rebound"] = rebound_pred == 1

    # Multiplier
    mult_model = models["multiplier_regressor"]["model"]
    latest["pred_multiplier"] = mult_model.predict(X)

    # EV
    ev_model = models["ev_estimator"]["model"]
    latest["pred_ev"] = ev_model.predict(X)

    # Compute derived trade fields
    latest["target_exit"] = (latest["implied_prob"] * latest["pred_multiplier"]).clip(upper=0.95)
    latest["stop_loss"] = (latest["implied_prob"] * 0.5).clip(lower=0.01)

    return latest


# ════════════════════════════════════════════════════════════════════════
# Output — Clear, actionable results
# ════════════════════════════════════════════════════════════════════════

def print_results(scored: pd.DataFrame, min_ev: float, min_rebound: float, top_n: int):
    """Print clean, ranked trade recommendations."""

    # Split into actionable and not
    buys = scored[
        (scored["pred_ev"] > min_ev) &
        (scored["rebound_prob"] > min_rebound)
    ].sort_values("pred_ev", ascending=False)

    no_trades = scored[
        ~scored.index.isin(buys.index)
    ].sort_values("pred_ev", ascending=False)

    # ── Header ──
    print(f"\n{B}{C}{'═' * 72}")
    print(f"  TRADE RECOMMENDATIONS — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═' * 72}{X}")
    print(f"  {D}Markets analyzed: {len(scored)}  |  Filters: EV > {min_ev:+.1f}, Rebound > {min_rebound:.0%}{X}")

    # ── BUY signals ──
    if len(buys) == 0:
        print(f"\n  {Y}No markets currently meet entry criteria.{X}")
        print(f"  {D}Try lowering thresholds with --min-ev or --min-rebound{X}")
    else:
        buy_count = min(len(buys), top_n) if top_n else len(buys)
        print(f"\n  {B}{G}⚡ {buy_count} TRADE SIGNAL{'S' if buy_count > 1 else ''} FOUND{X}")
        print(f"  {LINE}")

        for rank, (_, row) in enumerate(buys.head(buy_count).iterrows(), 1):
            ticker = row["ticker"]
            league = row.get("league", "")
            title = row.get("title", ticker)
            if pd.isna(title) or title == "0":
                title = ticker

            current = row["implied_prob"]
            initial = row["initial_prob"]
            regime = row["regime"]
            reb_p = row["rebound_prob"]
            mult = row["pred_multiplier"]
            ev = row["pred_ev"]
            target = row["target_exit"]
            stop = row["stop_loss"]
            op = row["op_value"]
            s = row["s_value"]
            drop_pct = row["prob_drop_pct"]

            # Determine which side to buy
            if row["is_favorite"]:
                buy_side = "YES (favorite side)"
                if current < initial:
                    buy_side = "YES (favorite collapsed — rebound expected)"
            else:
                buy_side = "YES (underdog side)"
                if current < initial:
                    buy_side = "YES (underdog collapsed — rebound expected)"

            # If implied_prob < 0.50, we're buying the underdog/collapsed side
            if current < 0.50:
                buy_side = f"YES at {current*100:.0f}¢ (collapsed side → rebound)"
            else:
                buy_side = f"NO at {(1-current)*100:.0f}¢ (other side collapsed)"

            # Always recommend buying the collapsed side at low price
            if row["prob_weak"] < 0.30 and row["op_value"] > 2:
                entry_price = row["prob_weak"]
                buy_side = f"YES on weak side at {entry_price*100:.0f}¢"
                target = min(entry_price * mult, 0.95)
                stop = entry_price * 0.5

            print(f"\n  {G}┌{'─' * 68}┐{X}")
            print(f"  {G}│  {B}#{rank}  BUY — {ticker[:52]}{X}{' ' * max(0, 52 - len(ticker))}{G}     │{X}")
            print(f"  {G}│{X}  {D}{str(title)[:64]}{X}{' ' * max(0, 64 - len(str(title)[:64]))}{G}│{X}")
            print(f"  {G}├{'─' * 68}┤{X}")
            print(f"  {G}│{X}  {'Action:':<20s} {B}{buy_side}{X}{' ' * max(0, 44 - len(buy_side))}{G}│{X}")
            ev_str = f"{ev:+.2f} per unit"
            print(f"  {G}│{X}  {'Expected Value:':<20s} {G}{ev_str}{X}{' ' * max(0, 44 - len(ev_str))}{G}│{X}")
            reb_str = f"{reb_p:.0%}"
            print(f"  {G}│{X}  {'Rebound Prob:':<20s} {reb_str}{' ' * max(0, 44 - len(reb_str))}{G}│{X}")
            mult_str = f"{mult:.1f}x"
            print(f"  {G}│{X}  {'Pred. Multiplier:':<20s} {mult_str}{' ' * max(0, 44 - len(mult_str))}{G}│{X}")
            tgt_str = f"{target*100:.0f}¢ ({target:.0%})"
            print(f"  {G}│{X}  {'Target Exit:':<20s} {tgt_str}{' ' * max(0, 44 - len(tgt_str))}{G}│{X}")
            sl_str = f"{stop*100:.0f}¢ ({stop:.0%})"
            print(f"  {G}│{X}  {'Stop Loss:':<20s} {sl_str}{' ' * max(0, 44 - len(sl_str))}{G}│{X}")
            regime_str = f"{regime} (conf: {row['regime_confidence']:.0%})"
            print(f"  {G}│{X}  {'Regime:':<20s} {regime_str}{' ' * max(0, 44 - len(regime_str))}{G}│{X}")
            op_str = f"OP={op:.1f}  S={s:.1f}  Drop={drop_pct:.0%}"
            print(f"  {G}│{X}  {'Key Metrics:':<20s} {op_str}{' ' * max(0, 44 - len(op_str))}{G}│{X}")
            prog_str = f"{row['market_progress']:.0%}" if row["market_progress"] < 1 else "near end"
            print(f"  {G}│{X}  {'Market Progress:':<20s} {prog_str}{' ' * max(0, 44 - len(prog_str))}{G}│{X}")
            print(f"  {G}└{'─' * 68}┘{X}")

    # ── Markets NOT meeting criteria ──
    if len(no_trades) > 0:
        show_n = min(5, len(no_trades))
        print(f"\n  {B}Markets analyzed but NOT meeting criteria ({len(no_trades)} total):{X}")
        print(f"  {LINE}")
        print(f"  {'Ticker':<42s} {'Prob':>5s} {'EV':>7s} {'Reb%':>5s} {'Regime':<10s}")
        print(f"  {'─'*42} {'─'*5} {'─'*7} {'─'*5} {'─'*10}")

        for _, row in no_trades.head(show_n).iterrows():
            ticker = str(row["ticker"])[:40]
            prob = row["implied_prob"]
            ev = row["pred_ev"]
            reb = row["rebound_prob"]
            regime = row["regime"]
            ev_c = color_pnl(ev)
            print(f"  {ticker:<42s} {prob:>4.0%} {ev_c}{ev:>+6.2f}{X} {reb:>4.0%} {regime:<10s}")

        if len(no_trades) > show_n:
            print(f"  {D}  ... and {len(no_trades) - show_n} more{X}")

    # ── Final summary ──
    print(f"\n{B}{C}{'═' * 72}{X}")
    if len(buys) > 0:
        total_ev = buys["pred_ev"].sum()
        avg_reb = buys["rebound_prob"].mean()
        print(f"  {G}{B}ACTION: {len(buys)} trade(s) recommended{X}")
        print(f"  {D}Combined predicted EV: {total_ev:+.2f} | Avg rebound prob: {avg_reb:.0%}{X}")
    else:
        print(f"  {Y}{B}ACTION: HOLD — No trades meet entry criteria right now{X}")
        print(f"  {D}Wait for a probability collapse (sharp drop) to trigger entry{X}")
    print(f"{B}{C}{'═' * 72}{X}\n")


# ════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Analyze live Kalshi CSV data and output trade recommendations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/analyze_csv.py kalshi_data/candlesticks.csv
  python scripts/analyze_csv.py kalshi_data/
  python scripts/analyze_csv.py live_data.csv --top 3 --min-ev 1.0
        """,
    )
    parser.add_argument("csv_path", help="Path to CSV file or directory containing CSVs")
    parser.add_argument("--models-dir", default="src/ml/models", help="Trained model directory")
    parser.add_argument("--min-ev", type=float, default=0.5, help="Minimum predicted EV to trigger BUY (default: 0.5)")
    parser.add_argument("--min-rebound", type=float, default=0.45, help="Minimum rebound probability (default: 0.45)")
    parser.add_argument("--top", type=int, default=10, help="Show top N signals (default: 10)")
    args = parser.parse_args()

    print(f"\n{B}{C}{'═' * 72}")
    print(f"  AI DUAL-REGIME REBOUND SYSTEM — LIVE ANALYSIS")
    print(f"{'═' * 72}{X}")
    print(f"  {D}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{X}\n")

    # 1. Load models
    print(f"  Loading trained models...")
    models = load_models(args.models_dir)
    print(f"  {G}✓ 4 models loaded{X}\n")

    # 2. Load CSV data
    print(f"  Loading data...")
    df = load_csv(args.csv_path)

    # 3. Compute features
    print(f"  Computing features...")
    df = prepare_features(df)
    print(f"  {G}✓ Features ready — {df['ticker'].nunique()} markets{X}")

    # 4. Score with models
    print(f"  Scoring with ML models...")
    scored = score_markets(df, models)
    print(f"  {G}✓ All markets scored{X}")

    # 5. Print results
    print_results(scored, args.min_ev, args.min_rebound, args.top)


if __name__ == "__main__":
    main()
