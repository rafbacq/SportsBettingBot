"""SQLite storage for historical data, trades, and model metrics."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from typing import Optional

from src.data.models import (
    ExitStrategy,
    Regime,
    TradeRecord,
    TradeStatus,
)

logger = logging.getLogger("trading.storage")


class TradingDatabase:
    """SQLite-backed persistent storage for the trading system."""

    def __init__(self, db_path: str = "data/trading.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()
        logger.info(f"Database initialized at {db_path}")

    def _create_tables(self):
        cur = self._conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS games (
                game_id TEXT PRIMARY KEY,
                sport TEXT NOT NULL,
                team_a TEXT NOT NULL,
                team_b TEXT NOT NULL,
                start_time REAL,
                total_duration_est REAL,
                kalshi_ticker TEXT,
                initial_prob_a REAL,
                initial_prob_b REAL,
                probability_curve_json TEXT,
                created_at REAL DEFAULT (strftime('%s', 'now'))
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                game_id TEXT NOT NULL,
                regime TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                entry_price REAL,
                entry_prob REAL,
                entry_timestamp REAL,
                stake_usd REAL,
                exit_price REAL,
                exit_prob REAL,
                exit_timestamp REAL,
                exit_strategy TEXT,
                exit_multiplier REAL,
                op_or_s_value REAL,
                pnl_usd REAL DEFAULT 0.0,
                kalshi_order_id TEXT,
                created_at REAL DEFAULT (strftime('%s', 'now')),
                FOREIGN KEY (game_id) REFERENCES games(game_id)
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS model_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                dataset_split TEXT,
                timestamp REAL DEFAULT (strftime('%s', 'now')),
                metadata_json TEXT
            )
        """)

        self._conn.commit()

    # ── Games ─────────────────────────────────────────────────────────

    def store_game(
        self,
        game_id: str,
        sport: str,
        team_a: str,
        team_b: str,
        start_time: float,
        total_duration_est: float,
        kalshi_ticker: str,
        initial_prob_a: float,
        initial_prob_b: float,
        probability_curve_json: str = "[]",
    ):
        """Insert or replace a game record."""
        self._conn.execute(
            """INSERT OR REPLACE INTO games
               (game_id, sport, team_a, team_b, start_time,
                total_duration_est, kalshi_ticker,
                initial_prob_a, initial_prob_b, probability_curve_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                game_id, sport, team_a, team_b, start_time,
                total_duration_est, kalshi_ticker,
                initial_prob_a, initial_prob_b, probability_curve_json,
            ),
        )
        self._conn.commit()

    def get_games_by_sport(self, sport: str) -> list[dict]:
        """Retrieve all games for a given sport."""
        cur = self._conn.execute(
            "SELECT * FROM games WHERE sport = ? ORDER BY start_time DESC", (sport,)
        )
        return [dict(row) for row in cur.fetchall()]

    def get_game(self, game_id: str) -> Optional[dict]:
        cur = self._conn.execute("SELECT * FROM games WHERE game_id = ?", (game_id,))
        row = cur.fetchone()
        return dict(row) if row else None

    # ── Trades ────────────────────────────────────────────────────────

    def store_trade(self, trade: TradeRecord):
        """Insert or update a trade record."""
        self._conn.execute(
            """INSERT OR REPLACE INTO trades
               (trade_id, game_id, regime, status,
                entry_price, entry_prob, entry_timestamp, stake_usd,
                exit_price, exit_prob, exit_timestamp, exit_strategy,
                exit_multiplier, op_or_s_value, pnl_usd, kalshi_order_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                trade.trade_id, trade.game_id, trade.regime.value,
                trade.status.value,
                trade.entry_price, trade.entry_prob, trade.entry_timestamp,
                trade.stake_usd,
                trade.exit_price, trade.exit_prob, trade.exit_timestamp,
                trade.exit_strategy.value,
                trade.exit_multiplier, trade.op_or_s_value,
                trade.pnl_usd, trade.kalshi_order_id,
            ),
        )
        self._conn.commit()

    def get_trade_history(
        self,
        regime: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """Retrieve trade history with optional filters."""
        query = "SELECT * FROM trades WHERE 1=1"
        params: list = []
        if regime:
            query += " AND regime = ?"
            params.append(regime)
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY entry_timestamp DESC LIMIT ?"
        params.append(limit)

        cur = self._conn.execute(query, params)
        return [dict(row) for row in cur.fetchall()]

    def get_open_trades(self) -> list[dict]:
        return self.get_trade_history(status="open")

    # ── Model Metrics ─────────────────────────────────────────────────

    def store_metric(
        self,
        model_name: str,
        metric_name: str,
        metric_value: float,
        dataset_split: str = "test",
        metadata: Optional[dict] = None,
    ):
        self._conn.execute(
            """INSERT INTO model_metrics
               (model_name, metric_name, metric_value, dataset_split, metadata_json)
               VALUES (?, ?, ?, ?, ?)""",
            (
                model_name, metric_name, metric_value, dataset_split,
                json.dumps(metadata) if metadata else None,
            ),
        )
        self._conn.commit()

    def get_metrics(self, model_name: str) -> list[dict]:
        cur = self._conn.execute(
            "SELECT * FROM model_metrics WHERE model_name = ? ORDER BY timestamp DESC",
            (model_name,),
        )
        return [dict(row) for row in cur.fetchall()]

    # ── Cleanup ───────────────────────────────────────────────────────

    def close(self):
        self._conn.close()
        logger.info("Database connection closed")
