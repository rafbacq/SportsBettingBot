"""Advanced risk management layer.

Enforces:
- Max concurrent positions
- Max loss per trade
- Daily loss limit / drawdown circuit breaker
- No duplicate game positions
- Trailing stop-loss management
- Time-decay stops
- Max drawdown position scaling
- Volatility-adjusted sizing
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from src.data.models import TradeSignal
from src.execution.portfolio import PortfolioManager

logger = logging.getLogger("trading.execution.risk")


@dataclass
class TrailingStopState:
    """Tracks the trailing stop for an active position."""
    trade_id: str
    entry_prob: float
    highest_prob_seen: float = 0.0  # high-water mark
    trailing_pct: float = 0.40     # trail by 40% of gain
    time_decay_factor: float = 1.0  # tightens as time runs out
    current_stop: float = 0.0      # current stop-loss level

    def update(self, current_prob: float, time_remaining_frac: float) -> float:
        """Update trailing stop and return current stop level.

        The stop ratchets UP as the probability moves in our favor.
        As time runs out, the trail tightens (time-decay).
        """
        # Update high-water mark
        if current_prob > self.highest_prob_seen:
            self.highest_prob_seen = current_prob

        # Time-decay factor: trail tightens as time runs out
        # At 100% time remaining: normal trail (e.g. 40%)
        # At 0% time remaining: very tight trail (e.g. 10%)
        self.time_decay_factor = 0.10 + 0.90 * max(time_remaining_frac, 0.0)
        effective_trail = self.trailing_pct * self.time_decay_factor

        # Trailing stop = highest_seen - (highest_seen - entry) * trail
        gain = self.highest_prob_seen - self.entry_prob
        if gain > 0:
            self.current_stop = self.highest_prob_seen - gain * effective_trail
        else:
            # No gain yet — keep basic stop loss
            self.current_stop = self.entry_prob * 0.5

        # Never let stop go below the basic stop loss
        self.current_stop = max(self.current_stop, self.entry_prob * 0.4)

        return self.current_stop


class RiskManager:
    """Enforces risk limits before allowing trade execution."""

    def __init__(self, config: dict):
        trading_cfg = config.get("trading", {})
        self.max_positions = trading_cfg.get("max_concurrent_positions", 10)
        self.daily_loss_limit = trading_cfg.get("daily_loss_limit_usd", 50.00)
        self.max_loss_per_trade = trading_cfg.get("max_loss_per_trade_usd", 10.00)
        self.daily_pnl: float = 0.0
        self._circuit_breaker_active = False

        # Advanced risk settings
        risk_cfg = config.get("risk", {})
        self.trailing_stop_pct = risk_cfg.get("trailing_stop_pct", 0.40)
        self.max_drawdown_pct = risk_cfg.get("max_drawdown_pct", 0.25)
        self.drawdown_scale_factor = risk_cfg.get("drawdown_scale_factor", 0.50)
        self.vol_scale_enabled = risk_cfg.get("volatility_scaling", True)

        # Trailing stop tracking
        self.trailing_stops: dict[str, TrailingStopState] = {}

        # Drawdown tracking
        self.peak_equity: float = trading_cfg.get("initial_bankroll_usd", 100.00)
        self.current_drawdown_pct: float = 0.0

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

        # Max drawdown check
        if self.current_drawdown_pct >= self.max_drawdown_pct:
            logger.warning(
                f"Max drawdown ({self.current_drawdown_pct:.1%}) reached — "
                f"reducing position sizes"
            )
            # Don't block, but the position will be scaled down via get_sizing_multiplier

        logger.debug(f"Risk check PASSED for {signal.game_id}")
        return True, "OK"

    def get_sizing_multiplier(self, portfolio: PortfolioManager) -> float:
        """Get position sizing multiplier based on current risk state.

        Returns value in (0, 1] — multiply the Kelly-suggested stake by this.
        1.0 = normal sizing, 0.5 = half size (during drawdown), etc.
        """
        multiplier = 1.0

        # Scale down during drawdowns
        if self.current_drawdown_pct > self.max_drawdown_pct * 0.5:
            # Linear scale-down: 100% at half-max drawdown → scale_factor at max
            drawdown_ratio = min(
                self.current_drawdown_pct / max(self.max_drawdown_pct, 1e-6), 1.0
            )
            multiplier *= 1.0 - (1.0 - self.drawdown_scale_factor) * drawdown_ratio

        return max(multiplier, 0.1)  # never go below 10%

    def register_trailing_stop(
        self,
        trade_id: str,
        entry_prob: float,
    ) -> None:
        """Register a trailing stop for a new position."""
        self.trailing_stops[trade_id] = TrailingStopState(
            trade_id=trade_id,
            entry_prob=entry_prob,
            highest_prob_seen=entry_prob,
            trailing_pct=self.trailing_stop_pct,
        )

    def check_trailing_stop(
        self,
        trade_id: str,
        current_prob: float,
        time_remaining_frac: float,
    ) -> bool:
        """Check if trailing stop has been hit for a position.

        Returns True if the position should be stopped out.
        """
        stop_state = self.trailing_stops.get(trade_id)
        if not stop_state:
            return False

        stop_level = stop_state.update(current_prob, time_remaining_frac)

        if current_prob <= stop_level:
            logger.info(
                f"TRAILING STOP HIT: trade={trade_id} | "
                f"prob={current_prob:.3f} <= stop={stop_level:.3f} | "
                f"high_water={stop_state.highest_prob_seen:.3f}"
            )
            return True

        return False

    def remove_trailing_stop(self, trade_id: str) -> None:
        """Clean up trailing stop state after position close."""
        self.trailing_stops.pop(trade_id, None)

    def record_trade_result(self, pnl: float, current_equity: float | None = None):
        """Update daily P&L and drawdown tracking after a trade closes."""
        self.daily_pnl += pnl
        logger.info(f"Daily P&L updated: ${self.daily_pnl:.2f}")

        if self.daily_pnl <= -self.daily_loss_limit:
            self._circuit_breaker_active = True
            logger.warning("CIRCUIT BREAKER ACTIVATED — daily loss limit hit")

        # Update drawdown tracking
        if current_equity is not None:
            if current_equity > self.peak_equity:
                self.peak_equity = current_equity
            if self.peak_equity > 0:
                self.current_drawdown_pct = (
                    (self.peak_equity - current_equity) / self.peak_equity
                )

    def reset_daily(self):
        """Reset daily counters (call at start of each trading day)."""
        self.daily_pnl = 0.0
        self._circuit_breaker_active = False
        logger.info("Daily risk counters reset")

    def status(self) -> str:
        return (
            f"Risk | daily_pnl=${self.daily_pnl:.2f} | "
            f"limit=${self.daily_loss_limit:.2f} | "
            f"drawdown={self.current_drawdown_pct:.1%} | "
            f"trailing_stops={len(self.trailing_stops)} | "
            f"circuit_breaker={'ACTIVE' if self._circuit_breaker_active else 'off'}"
        )
