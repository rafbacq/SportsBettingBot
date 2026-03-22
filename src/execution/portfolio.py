"""Portfolio management — position tracking and sizing.

Tracks all open positions, computes P&L, and determines position sizes.
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from src.data.models import (
    ExitStrategy,
    Regime,
    TradeRecord,
    TradeSignal,
    TradeStatus,
)

logger = logging.getLogger("trading.execution.portfolio")


class PortfolioManager:
    """Manages open positions and handles sizing."""

    def __init__(self, config: dict):
        trading_cfg = config.get("trading", {})
        self.default_stake = trading_cfg.get("default_stake_usd", 1.00)
        self.max_stake = trading_cfg.get("max_stake_usd", 10.00)
        self.positions: dict[str, TradeRecord] = {}   # trade_id → TradeRecord

    def open_position(self, signal: TradeSignal, stake: float | None = None) -> TradeRecord:
        """Create a new open position from a trade signal."""
        trade_id = f"T-{uuid.uuid4().hex[:8]}"
        stake = stake or self._compute_stake(signal)

        trade = TradeRecord(
            trade_id=trade_id,
            game_id=signal.game_id,
            regime=signal.regime,
            status=TradeStatus.OPEN,
            entry_price=signal.entry_prob * 100,
            entry_prob=signal.entry_prob,
            entry_timestamp=signal.timestamp,
            stake_usd=stake,
            exit_multiplier=signal.exit_multiplier,
            op_or_s_value=signal.op_or_s_value,
            exit_strategy=signal.exit_strategy,
        )

        self.positions[trade_id] = trade
        logger.info(
            f"OPENED position {trade_id}: {signal.regime.value} | "
            f"entry={signal.entry_prob:.3f} | stake=${stake:.2f}"
        )
        return trade

    def close_position(
        self,
        trade_id: str,
        exit_prob: float,
        exit_timestamp: float,
    ) -> TradeRecord:
        """Close an open position and compute P&L."""
        trade = self.positions.get(trade_id)
        if not trade:
            raise ValueError(f"No open position with id {trade_id}")

        trade.exit_prob = exit_prob
        trade.exit_price = exit_prob * 100
        trade.exit_timestamp = exit_timestamp
        trade.status = TradeStatus.CLOSED

        # P&L: simplified binary contract model
        # Bought at entry_prob (cents), position value now exit_prob (cents)
        # Profit = (exit_price - entry_price) / 100 * contracts
        contracts = trade.stake_usd / (trade.entry_prob)
        trade.pnl_usd = contracts * (exit_prob - trade.entry_prob)

        logger.info(
            f"CLOSED position {trade_id}: "
            f"entry={trade.entry_prob:.3f} → exit={exit_prob:.3f} | "
            f"mult={trade.multiplier_achieved:.1f}x | "
            f"P&L=${trade.pnl_usd:.2f}"
        )

        del self.positions[trade_id]
        return trade

    def _compute_stake(self, signal: TradeSignal) -> float:
        """Determine position size based on signal confidence."""
        # Simple scaling: base stake * confidence, capped
        stake = self.default_stake * signal.confidence
        return min(max(stake, self.default_stake * 0.5), self.max_stake)

    @property
    def open_count(self) -> int:
        return len(self.positions)

    @property
    def total_exposure(self) -> float:
        return sum(t.stake_usd for t in self.positions.values())

    def get_position_for_game(self, game_id: str) -> TradeRecord | None:
        for t in self.positions.values():
            if t.game_id == game_id:
                return t
        return None

    def summary(self) -> str:
        if not self.positions:
            return "No open positions"
        lines = [f"Open positions ({len(self.positions)}):"]
        for tid, t in self.positions.items():
            lines.append(
                f"  {tid} | {t.regime.value:10s} | game={t.game_id} | "
                f"entry={t.entry_prob:.3f} | stake=${t.stake_usd:.2f}"
            )
        lines.append(f"Total exposure: ${self.total_exposure:.2f}")
        return "\n".join(lines)
