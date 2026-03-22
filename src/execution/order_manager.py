"""Order manager — interfaces with Kalshi for order placement.

Supports dry-run mode (logs trades without execution).
"""

from __future__ import annotations

import logging
from typing import Optional

from src.data.kalshi_client import KalshiClient
from src.data.models import TradeRecord, TradeSignal

logger = logging.getLogger("trading.execution.orders")


class OrderManager:
    """Manages order placement and tracking with Kalshi."""

    def __init__(self, config: dict, kalshi_client: KalshiClient | None = None):
        self.dry_run = config.get("trading", {}).get("dry_run", True)
        self.client = kalshi_client
        self._order_log: list[dict] = []

    def place_entry_order(
        self,
        signal: TradeSignal,
        stake_usd: float,
        ticker: str,
    ) -> str | None:
        """Place an entry order (buy YES on the collapsing team).

        Returns order_id or None.
        """
        side = "yes"
        action = "buy"
        # Convert probability to Kalshi price (cents)
        limit_price = max(1, int(signal.entry_prob * 100))
        # Contracts = stake / price
        count = max(1, int(stake_usd / (limit_price / 100)))

        if self.dry_run:
            order_id = f"DRY-{len(self._order_log):04d}"
            logger.info(
                f"[DRY RUN] Entry order: {action} {count} {side} @ {limit_price}¢ | "
                f"ticker={ticker} | order_id={order_id}"
            )
            self._order_log.append({
                "order_id": order_id,
                "type": "entry",
                "ticker": ticker,
                "side": side,
                "action": action,
                "count": count,
                "price": limit_price,
                "signal": signal,
            })
            return order_id

        if not self.client:
            logger.error("No Kalshi client configured for live trading")
            return None

        try:
            result = self.client.place_order(
                ticker=ticker,
                side=side,
                action=action,
                count=count,
                order_type="limit",
                limit_price=limit_price,
            )
            order_id = result.get("order", {}).get("order_id")
            logger.info(f"Entry order placed: {order_id}")
            return order_id
        except Exception as e:
            logger.error(f"Failed to place entry order: {e}")
            return None

    def place_exit_order(
        self,
        trade: TradeRecord,
        exit_prob: float,
        ticker: str,
    ) -> str | None:
        """Place an exit order (sell YES position)."""
        side = "yes"
        action = "sell"
        limit_price = max(1, int(exit_prob * 100))
        count = max(1, int(trade.stake_usd / (trade.entry_prob)))

        if self.dry_run:
            order_id = f"DRY-EXIT-{len(self._order_log):04d}"
            logger.info(
                f"[DRY RUN] Exit order: {action} {count} {side} @ {limit_price}¢ | "
                f"ticker={ticker} | trade={trade.trade_id}"
            )
            self._order_log.append({
                "order_id": order_id,
                "type": "exit",
                "ticker": ticker,
                "trade_id": trade.trade_id,
                "price": limit_price,
            })
            return order_id

        if not self.client:
            logger.error("No Kalshi client configured for live trading")
            return None

        try:
            result = self.client.place_order(
                ticker=ticker,
                side=side,
                action=action,
                count=count,
                order_type="limit",
                limit_price=limit_price,
            )
            order_id = result.get("order", {}).get("order_id")
            logger.info(f"Exit order placed: {order_id}")
            return order_id
        except Exception as e:
            logger.error(f"Failed to place exit order: {e}")
            return None

    @property
    def order_count(self) -> int:
        return len(self._order_log)

    def get_order_log(self) -> list[dict]:
        return list(self._order_log)
