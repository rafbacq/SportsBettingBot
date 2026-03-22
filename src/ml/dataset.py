"""Training dataset builder.

Creates labeled training samples from historical probability curves:
1. Processes GameState objects into candidate entry points
2. Labels each sample: did_rebound, max_rebound_multiplier, regime
3. Generates feature vectors via FeatureEngine
4. Splits into train / validation / test

Also includes a synthetic data generator for development and testing.
"""

from __future__ import annotations

import logging
import random
from typing import Optional

import numpy as np
import pandas as pd

from src.data.models import GameState, ProbabilityCurve, ProbabilitySnapshot, Regime
from src.features.engine import FeatureEngine, FeatureVector

logger = logging.getLogger("trading.dataset")


class DatasetBuilder:
    """Builds training datasets from historical game data."""

    def __init__(self, feature_engine: FeatureEngine | None = None):
        self.engine = feature_engine or FeatureEngine()

    def build_from_games(
        self,
        games: list[GameState],
        min_collapse_pct: float = 0.30,
    ) -> pd.DataFrame:
        """Process games into labeled training samples.

        For each game, identifies all snapshots where a significant probability
        collapse has occurred. Each is a candidate entry point.

        Labels:
        - regime: "cross" if the collapsing team eventually crosses 50%,
                  "non_cross" otherwise
        - did_rebound: True if probability recovered by ≥2x from entry
        - max_rebound_multiplier: highest Pt_future / Pt_entry seen after entry
        - exit_prob_at_max: the probability at max rebound
        """
        all_samples: list[dict] = []

        for game in games:
            samples = self._process_game(game, min_collapse_pct)
            all_samples.extend(samples)

        if not all_samples:
            logger.warning("No training samples generated")
            return pd.DataFrame()

        df = pd.DataFrame(all_samples)
        logger.info(
            f"Built dataset: {len(df)} samples from {len(games)} games | "
            f"rebounds: {df['did_rebound'].sum()} ({df['did_rebound'].mean():.1%})"
        )
        return df

    def _process_game(
        self,
        game: GameState,
        min_collapse_pct: float,
    ) -> list[dict]:
        """Extract candidate entry points from a single game."""
        curve = game.curve
        if len(curve.snapshots) < 10:
            return []

        samples = []
        snap0 = curve.snapshots[0]

        # Determine favorite
        is_a_fav = snap0.prob_a >= snap0.prob_b

        for idx in range(5, len(curve.snapshots) - 5):
            snap = curve.snapshots[idx]

            # Check weak team collapse (Non-Cross candidate)
            if is_a_fav:
                p0_weak, pt_weak = snap0.prob_b, snap.prob_b
                p0_strong, pt_strong = snap0.prob_a, snap.prob_a
            else:
                p0_weak, pt_weak = snap0.prob_a, snap.prob_a
                p0_strong, pt_strong = snap0.prob_b, snap.prob_b

            # Check for significant collapse in weak team
            weak_collapse = (p0_weak - pt_weak) / max(p0_weak, 1e-6)
            strong_collapse = (p0_strong - pt_strong) / max(p0_strong, 1e-6)

            is_candidate = False
            candidate_type = None

            if weak_collapse >= min_collapse_pct and pt_weak < 0.15:
                is_candidate = True
                candidate_type = "weak_collapse"

            if strong_collapse >= min_collapse_pct and pt_strong < 0.30:
                is_candidate = True
                candidate_type = "strong_collapse"

            if not is_candidate:
                continue

            # Compute features
            fv = self.engine.compute(game, snapshot_idx=idx)

            # Look ahead for rebound
            future_probs_weak = [
                curve.snapshots[j].prob_b if is_a_fav else curve.snapshots[j].prob_a
                for j in range(idx, len(curve.snapshots))
            ]
            future_probs_strong = [
                curve.snapshots[j].prob_a if is_a_fav else curve.snapshots[j].prob_b
                for j in range(idx, len(curve.snapshots))
            ]

            # Non-Cross analysis (weak team rebound)
            max_rebound_weak = max(future_probs_weak) if future_probs_weak else pt_weak
            multiplier_weak = max_rebound_weak / max(pt_weak, 1e-6)
            did_rebound_weak = multiplier_weak >= 2.0
            crosses_50_weak = max_rebound_weak >= 0.50

            # Cross analysis (strong team rebound)
            max_rebound_strong = max(future_probs_strong) if future_probs_strong else pt_strong
            multiplier_strong = max_rebound_strong / max(pt_strong, 1e-6)
            did_rebound_strong = multiplier_strong >= 2.0

            # Determine regime label
            if candidate_type == "strong_collapse":
                regime = Regime.CROSS.value
                did_rebound = did_rebound_strong
                max_multiplier = multiplier_strong
                exit_prob_at_max = max_rebound_strong
            else:
                # If weak team crosses 50%, it could be Cross from other perspective
                regime = Regime.CROSS.value if crosses_50_weak else Regime.NON_CROSS.value
                did_rebound = did_rebound_weak
                max_multiplier = multiplier_weak
                exit_prob_at_max = max_rebound_weak

            sample = fv.to_dict()
            sample.update({
                "game_id": game.game_id,
                "sport": game.sport,
                "snapshot_idx": idx,
                "regime": regime,
                "did_rebound": did_rebound,
                "max_rebound_multiplier": min(max_multiplier, 50.0),  # cap outliers
                "exit_prob_at_max": exit_prob_at_max,
                "candidate_type": candidate_type,
            })
            samples.append(sample)

        return samples

    def split_dataset(
        self,
        df: pd.DataFrame,
        ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
        seed: int = 42,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split dataset into train / validation / test by game_id.

        Games are split (not individual samples) to avoid data leakage.
        """
        game_ids = df["game_id"].unique().tolist()
        rng = np.random.RandomState(seed)
        rng.shuffle(game_ids)

        n = len(game_ids)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])

        train_ids = set(game_ids[:n_train])
        val_ids = set(game_ids[n_train : n_train + n_val])
        test_ids = set(game_ids[n_train + n_val :])

        train_df = df[df["game_id"].isin(train_ids)].copy()
        val_df = df[df["game_id"].isin(val_ids)].copy()
        test_df = df[df["game_id"].isin(test_ids)].copy()

        logger.info(
            f"Split: train={len(train_df)} ({len(train_ids)} games), "
            f"val={len(val_df)} ({len(val_ids)} games), "
            f"test={len(test_df)} ({len(test_ids)} games)"
        )
        return train_df, val_df, test_df


# ── Synthetic Data Generator ─────────────────────────────────────────────

class SyntheticDataGenerator:
    """Generates synthetic probability curves for development/testing.

    Creates games with known rebound patterns so the ML pipeline can
    be tested end-to-end without real historical data.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def generate(self, n_games: int = 1000) -> list[GameState]:
        """Generate n synthetic games with realistic probability dynamics."""
        games = []
        sports = ["NCAAB", "ATP"]

        for i in range(n_games):
            sport = sports[i % len(sports)]
            game = self._generate_game(f"SYN-{i:05d}", sport)
            games.append(game)

        logger.info(f"Generated {len(games)} synthetic games")
        return games

    def _generate_game(self, game_id: str, sport: str) -> GameState:
        """Generate a single synthetic game with probability dynamics."""
        # Random initial probabilities
        p0_a = self.rng.uniform(0.20, 0.80)
        p0_b = 1.0 - p0_a

        # Duration: 100 snapshots (~40 min game with updates every 24 sec)
        n_snapshots = self.rng.randint(80, 200)
        duration = float(n_snapshots * 24)  # seconds

        # Generate probability path
        probs = self._random_walk_with_rebounds(p0_a, n_snapshots)

        game = GameState(
            game_id=game_id,
            sport=sport,
            team_a=f"Team-A-{game_id}",
            team_b=f"Team-B-{game_id}",
            start_time=0.0,
            total_duration_est=duration,
            kalshi_ticker=f"SYN-{game_id}",
        )

        for t, p in enumerate(probs):
            game.add_probability(float(t * 24), p)

        return game

    def _random_walk_with_rebounds(
        self, p0: float, n_steps: int
    ) -> list[float]:
        """Generate a probability path with potential collapses and rebounds."""
        probs = [p0]
        p = p0

        # Decide if this game will have a collapse event
        has_collapse = self.rng.random() < 0.40  # 40% of games
        collapse_start = self.rng.randint(n_steps // 5, 3 * n_steps // 5) if has_collapse else -1
        collapse_magnitude = self.rng.uniform(0.30, 0.70) if has_collapse else 0
        has_rebound = self.rng.random() < 0.35  # 35% of collapses rebound
        rebound_magnitude = self.rng.uniform(0.20, collapse_magnitude) if has_rebound else 0

        for t in range(1, n_steps):
            # Normal random walk
            noise = self.rng.normal(0, 0.015)

            # Add collapse dynamics
            if has_collapse and collapse_start <= t < collapse_start + 15:
                # Sharp collapse phase
                noise -= collapse_magnitude / 15.0

            elif has_collapse and has_rebound and collapse_start + 15 <= t < collapse_start + 40:
                # Rebound phase
                noise += rebound_magnitude / 25.0

            p = p + noise
            p = np.clip(p, 0.01, 0.99)
            probs.append(float(p))

        return probs
