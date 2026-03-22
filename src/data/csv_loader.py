"""CSV data loader for historical probability curve data.

Expected CSV format:
  game_id, timestamp, team_a, team_b, prob_a, prob_b, time_remaining, sport

Loads CSV files into ProbabilityCurve and GameState objects for ML training
and backtesting.
"""

from __future__ import annotations

import glob
import logging
import os
from typing import Optional

import pandas as pd

from src.data.models import GameState, ProbabilityCurve, ProbabilitySnapshot

logger = logging.getLogger("trading.csv_loader")


def load_csv(filepath: str) -> pd.DataFrame:
    """Load a single CSV file and validate columns."""
    required_cols = {"game_id", "timestamp", "team_a", "team_b", "prob_a", "prob_b"}
    df = pd.read_csv(filepath)

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV {filepath} missing columns: {missing}")

    # Ensure probabilities are in [0, 1]
    if (df["prob_a"] > 1).any():
        logger.info("Converting prob_a from percentage to decimal")
        df["prob_a"] = df["prob_a"] / 100.0
        df["prob_b"] = df["prob_b"] / 100.0

    df = df.sort_values(["game_id", "timestamp"]).reset_index(drop=True)
    return df


def load_csv_directory(data_dir: str, pattern: str = "*.csv") -> pd.DataFrame:
    """Load all CSVs from a directory and concatenate them."""
    files = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if not files:
        logger.warning(f"No CSV files found in {data_dir} matching {pattern}")
        return pd.DataFrame()

    logger.info(f"Loading {len(files)} CSV files from {data_dir}")
    dfs = [load_csv(f) for f in files]
    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(combined)} total rows across {combined['game_id'].nunique()} games")
    return combined


def dataframe_to_game_states(df: pd.DataFrame) -> list[GameState]:
    """Convert a DataFrame of probability data into GameState objects.

    Each unique game_id becomes one GameState with a full ProbabilityCurve.
    """
    games: list[GameState] = []

    for game_id, group in df.groupby("game_id"):
        group = group.sort_values("timestamp")
        row0 = group.iloc[0]

        sport = str(row0.get("sport", "UNKNOWN"))
        team_a = str(row0["team_a"])
        team_b = str(row0["team_b"])

        # Estimate total duration from data
        ts_min = group["timestamp"].min()
        ts_max = group["timestamp"].max()
        duration_est = float(ts_max - ts_min) if ts_max > ts_min else 3600.0

        time_remaining_col = "time_remaining" in group.columns

        curve = ProbabilityCurve(game_id=str(game_id))
        for _, r in group.iterrows():
            snap = ProbabilitySnapshot(
                timestamp=float(r["timestamp"]),
                prob_a=float(r["prob_a"]),
                prob_b=float(r["prob_b"]),
            )
            curve.add_snapshot(snap)

        gs = GameState(
            game_id=str(game_id),
            sport=sport,
            team_a=team_a,
            team_b=team_b,
            start_time=float(ts_min),
            total_duration_est=duration_est,
            kalshi_ticker=f"GAME-{game_id}",
            curve=curve,
            is_live=False,
        )
        games.append(gs)

    logger.info(f"Created {len(games)} GameState objects")
    return games


def load_games_from_csv(
    data_dir: str, pattern: str = "*.csv"
) -> list[GameState]:
    """Convenience: load CSVs → DataFrame → GameStates in one call."""
    df = load_csv_directory(data_dir, pattern)
    if df.empty:
        return []
    return dataframe_to_game_states(df)
