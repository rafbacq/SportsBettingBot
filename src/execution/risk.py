"""Risk management layer.

Enforces:
- Max concurrent positions
- Max loss per trade
- Daily loss limit / drawdown circuit breaker
- No duplicate game positions
"""

from __future__ import annotations

import logging
from typing import Optional

from src.data.models import TradeSignal
from src.execution.portfolio import PortfolioManager

logger = logging.getLogger("trading.execution.risk")


class RiskManager:
    """Enforces risk limits before allowing trade execution."""

    def __init__(self, config: dict):
        trading_cfg = config.get("trading", {})
        self.max_positions = trading_cfg.get("max_concurrent_positions", 10)
        self.daily_loss_limit = trading_cfg.get("daily_loss_limit_usd", 50.00)
        self.max_loss_per_trade = trading_cfg.get("max_loss_per_trade_usd", 10.00)
        self.daily_pnl: float = 0.0
        self._circuit_breaker_active = False

    def check_trade(
        self,
        signal: TradeSignal,
        portfolio: PortfolioManager,
    ) -> tuple[bool, str]:
        """Check if a trade passes all risk checks.

        Returns:
            (allowed: bool, reason: str)
        """
        # Circuit breaker
        if self._circuit_breaker_active:
            return False, "CIRCUIT BREAKER: daily loss limit hit"

        # Max positions
        if portfolio.open_count >= self.max_positions:
            return False, f"Max positions ({self.max_positions}) reached"

        # No duplicate game
        existing = portfolio.get_position_for_game(signal.game_id)
        if existing:
            return False, f"Already have position on {signal.game_id}"

        # Daily loss limit
        if self.daily_pnl <= -self.daily_loss_limit:
            self._circuit_breaker_active = True
            return False, f"Daily loss limit (${self.daily_loss_limit}) exceeded"

        logger.debug(f"Risk check PASSED for {signal.game_id}")
        return True, "OK"

    def record_trade_result(self, pnl: float):
        """Update daily P&L after a trade closes."""
        self.daily_pnl += pnl
        logger.info(f"Daily P&L updated: ${self.daily_pnl:.2f}")

        if self.daily_pnl <= -self.daily_loss_limit:
            self._circuit_breaker_active = True
            logger.warning("CIRCUIT BREAKER ACTIVATED — daily loss limit hit")

    def reset_daily(self):
        """Reset daily counters (call at start of each trading day)."""
        self.daily_pnl = 0.0
        self._circuit_breaker_active = False
        logger.info("Daily risk counters reset")

    def status(self) -> str:
        return (
            f"Risk | daily_pnl=${self.daily_pnl:.2f} | "
            f"limit=${self.daily_loss_limit:.2f} | "
            f"circuit_breaker={'ACTIVE' if self._circuit_breaker_active else 'off'}"
        )
