"""Kalshi API client for REST and WebSocket interactions.

Provides:
- KalshiClient: REST API wrapper for market data, trading, and historical data
- KalshiWebSocket: WebSocket wrapper for real-time probability feeds
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import time
from typing import Any, Callable, Optional

import aiohttp
import requests

logger = logging.getLogger("trading.kalshi_client")


# ── REST Client ──────────────────────────────────────────────────────────

class KalshiClient:
    """Synchronous REST client for the Kalshi Trade API v2."""

    def __init__(self, config: dict):
        kalshi_cfg = config.get("kalshi", {})
        self.base_url = kalshi_cfg.get(
            "base_url", "https://api.elections.kalshi.com/trade-api/v2"
        )
        self.api_key = kalshi_cfg.get("api_key", "")
        self.private_key_path = kalshi_cfg.get("private_key_path", "")
        self.rate_limit = kalshi_cfg.get("requests_per_second", 10)
        self._last_request_time = 0.0
        self._session = requests.Session()
        self._setup_auth()

    # ── Auth ──────────────────────────────────────────────────────────

    def _setup_auth(self):
        """Configure authentication headers."""
        if self.api_key:
            self._session.headers.update({
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            })
        else:
            logger.warning("No API key configured — only public endpoints available")
            self._session.headers.update({"Content-Type": "application/json"})

    def _sign_request(self, method: str, path: str, timestamp: str) -> str:
        """Sign a request using RSA-PSS (required for private endpoints).
        
        If private key is not configured, returns empty string.
        """
        if not self.private_key_path:
            return ""
        try:
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import padding

            with open(self.private_key_path, "rb") as f:
                private_key = serialization.load_pem_private_key(f.read(), password=None)

            message = f"{timestamp}{method}{path}".encode()
            signature = private_key.sign(
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )
            return base64.b64encode(signature).decode()
        except Exception as e:
            logger.error(f"Failed to sign request: {e}")
            return ""

    # ── Rate limiting ─────────────────────────────────────────────────

    def _throttle(self):
        """Enforce rate limit."""
        now = time.time()
        min_interval = 1.0 / self.rate_limit
        elapsed = now - self._last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_time = time.time()

    # ── HTTP helpers ──────────────────────────────────────────────────

    def _get(self, path: str, params: dict | None = None) -> dict:
        self._throttle()
        url = f"{self.base_url}{path}"
        try:
            resp = self._session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.error(f"GET {path} failed: {e}")
            return {}

    def _post(self, path: str, data: dict | None = None) -> dict:
        self._throttle()
        url = f"{self.base_url}{path}"
        timestamp = str(int(time.time()))
        sig = self._sign_request("POST", path, timestamp)
        headers = {}
        if sig:
            headers["KALSHI-ACCESS-TIMESTAMP"] = timestamp
            headers["KALSHI-ACCESS-SIGNATURE"] = sig
        try:
            resp = self._session.post(url, json=data, headers=headers, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.error(f"POST {path} failed: {e}")
            return {}

    def _delete(self, path: str) -> dict:
        self._throttle()
        url = f"{self.base_url}{path}"
        try:
            resp = self._session.delete(url, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.error(f"DELETE {path} failed: {e}")
            return {}

    # ── Markets ───────────────────────────────────────────────────────

    def get_markets(
        self,
        status: str = "open",
        series_ticker: str | None = None,
        event_ticker: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> dict:
        """Get paginated list of markets."""
        params = {"status": status, "limit": limit}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if event_ticker:
            params["event_ticker"] = event_ticker
        if cursor:
            params["cursor"] = cursor
        return self._get("/markets", params)

    def get_market(self, ticker: str) -> dict:
        """Get single market by ticker."""
        return self._get(f"/markets/{ticker}")

    def get_market_orderbook(self, ticker: str, depth: int = 10) -> dict:
        """Get current order book for a market."""
        return self._get(f"/markets/{ticker}/orderbook", {"depth": depth})

    def get_market_candlesticks(
        self,
        ticker: str,
        series_ticker: str | None = None,
        period_interval: int = 1,      # minutes
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> dict:
        """Get historical candlestick data for a market."""
        params = {"market_tickers": ticker, "period_interval": period_interval}
        if start_ts:
            params["start_ts"] = start_ts
        if end_ts:
            params["end_ts"] = end_ts
            
        data = self._get("/markets/candlesticks", params)
        markets = data.get("markets", [])
        if markets and isinstance(markets, list) and len(markets) > 0:
            return {"candlesticks": markets[0].get("candlesticks", [])}
        return {"candlesticks": []}

    # ── Events ────────────────────────────────────────────────────────

    def get_events(
        self,
        status: str | None = None,
        series_ticker: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> dict:
        """Get events list."""
        params = {"limit": limit}
        if status:
            params["status"] = status
        if series_ticker:
            params["series_ticker"] = series_ticker
        if cursor:
            params["cursor"] = cursor
        return self._get("/events", params)

    def get_event(self, event_ticker: str) -> dict:
        """Get single event by ticker."""
        return self._get(f"/events/{event_ticker}")

    # ── Historical Data ───────────────────────────────────────────────

    def get_historical_markets(
        self,
        limit: int = 100,
        cursor: str | None = None,
        min_close_ts: int | None = None,
        max_close_ts: int | None = None,
    ) -> dict:
        """Get historical (settled) markets."""
        params = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        if min_close_ts:
            params["min_close_ts"] = min_close_ts
        if max_close_ts:
            params["max_close_ts"] = max_close_ts
        return self._get("/historical/markets", params)

    # ── Trading ───────────────────────────────────────────────────────

    def place_order(
        self,
        ticker: str,
        side: str,                  # "yes" or "no"
        action: str,                # "buy" or "sell"
        count: int = 1,
        order_type: str = "market",
        limit_price: int | None = None,   # cents (1-99)
    ) -> dict:
        """Place an order on a market."""
        data: dict[str, Any] = {
            "ticker": ticker,
            "side": side,
            "action": action,
            "count": count,
            "type": order_type,
        }
        if limit_price is not None and order_type == "limit":
            data["yes_price"] = limit_price if side == "yes" else None
            data["no_price"] = limit_price if side == "no" else None
        return self._post("/portfolio/orders", data)

    def cancel_order(self, order_id: str) -> dict:
        """Cancel an open order."""
        return self._delete(f"/portfolio/orders/{order_id}")

    def get_positions(self) -> dict:
        """Get current portfolio positions."""
        return self._get("/portfolio/positions")

    def get_balance(self) -> dict:
        """Get current account balance."""
        return self._get("/portfolio/balance")


# ── WebSocket Client ─────────────────────────────────────────────────────

class KalshiWebSocket:
    """Async WebSocket client for real-time market data from Kalshi."""

    def __init__(self, config: dict):
        kalshi_cfg = config.get("kalshi", {})
        self.ws_url = kalshi_cfg.get(
            "ws_url", "wss://api.elections.kalshi.com/trade-api/ws/v2"
        )
        self.api_key = kalshi_cfg.get("api_key", "")
        self._ws = None
        self._callbacks: dict[str, list[Callable]] = {}
        self._running = False

    def on_price_update(self, callback: Callable[[str, float], None]):
        """Register a callback for price updates.
        
        Callback receives (ticker: str, yes_price: float).
        """
        self._callbacks.setdefault("price", []).append(callback)

    def on_trade(self, callback: Callable[[dict], None]):
        """Register a callback for trade events."""
        self._callbacks.setdefault("trade", []).append(callback)

    async def connect(self):
        """Connect to the WebSocket."""
        import websockets

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            self._ws = await websockets.connect(
                self.ws_url,
                additional_headers=headers,
                ping_interval=30,
                ping_timeout=10,
            )
            self._running = True
            logger.info("WebSocket connected to Kalshi")
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            raise

    async def subscribe(self, tickers: list[str], channels: list[str] | None = None):
        """Subscribe to market updates for given tickers."""
        if channels is None:
            channels = ["orderbook_delta", "ticker"]

        msg = {
            "id": 1,
            "cmd": "subscribe",
            "params": {
                "channels": channels,
                "market_tickers": tickers,
            },
        }
        if self._ws:
            await self._ws.send(json.dumps(msg))
            logger.info(f"Subscribed to {len(tickers)} tickers: {channels}")

    async def listen(self):
        """Listen for messages and dispatch to callbacks."""
        if not self._ws:
            raise RuntimeError("Not connected. Call connect() first.")

        try:
            async for message in self._ws:
                try:
                    data = json.loads(message)
                    msg_type = data.get("type", "")

                    if msg_type == "orderbook_delta" or msg_type == "ticker":
                        ticker = data.get("msg", {}).get("market_ticker", "")
                        yes_price = data.get("msg", {}).get("yes_price", 0)
                        # Normalize price to probability (Kalshi prices are cents)
                        prob = yes_price / 100.0 if yes_price > 1 else yes_price

                        for cb in self._callbacks.get("price", []):
                            cb(ticker, prob)

                    elif msg_type == "trade":
                        for cb in self._callbacks.get("trade", []):
                            cb(data.get("msg", {}))

                except json.JSONDecodeError:
                    logger.warning(f"Non-JSON WebSocket message: {message[:100]}")

        except Exception as e:
            logger.error(f"WebSocket listen error: {e}")
            self._running = False

    async def disconnect(self):
        """Close the WebSocket connection."""
        self._running = False
        if self._ws:
            await self._ws.close()
            logger.info("WebSocket disconnected")

    @property
    def is_connected(self) -> bool:
        return self._running and self._ws is not None
