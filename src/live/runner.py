"""Live trading runner.

Main event loop:
1. Connect to Kalshi WebSocket for live probability feeds
2. For each price update: compute features → classify regime → evaluate entry/exit
3. If trade signal: check risk → place order
4. Monitor open positions for exit conditions
5. Log everything, display CLI status
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from src.data.kalshi_client import KalshiClient, KalshiWebSocket
from src.data.models import GameState, Regime, TradeStatus
from src.data.storage import TradingDatabase
from src.execution.order_manager import OrderManager
from src.execution.portfolio import PortfolioManager
from src.execution.risk import RiskManager
from src.features.engine import FeatureEngine
from src.strategy.cross import CrossStrategy
from src.strategy.non_cross import NonCrossStrategy
from src.strategy.regime_router import RegimeRouter
from src.utils.logging_config import load_config

logger = logging.getLogger("trading.live.runner")


class LiveTradingRunner:
    """Real-time trading loop connecting Kalshi data → strategy → execution."""

    def __init__(self, config: dict | None = None):
        self.config = config or load_config()

        # Core components
        self.kalshi = KalshiClient(self.config)
        self.ws = KalshiWebSocket(self.config)
        self.db = TradingDatabase(
            self.config.get("storage", {}).get("db_path", "data/trading.db")
        )

        # Execution
        self.portfolio = PortfolioManager(self.config)
        self.risk = RiskManager(self.config)
        self.orders = OrderManager(self.config, self.kalshi)

        # Strategy
        self.router = RegimeRouter(self.config)

        # Feature engine
        self.feature_engine = FeatureEngine()

        # Game state tracking
        self.active_games: dict[str, GameState] = {}   # ticker → GameState

        # Strategies for exit monitoring
        self.nc_strategy = NonCrossStrategy()
        self.cr_strategy = CrossStrategy()

        self._running = False

    def start(self):
        """Start the live trading loop."""
        logger.info("=" * 60)
        logger.info("LIVE TRADING RUNNER STARTING")
        logger.info(f"Dry run: {self.config.get('trading', {}).get('dry_run', True)}")
        logger.info("=" * 60)

        # Load ML models
        self.router.load_models()

        # Start async event loop
        asyncio.run(self._run())

    async def _run(self):
        """Async main loop."""
        self._running = True

        # Connect WebSocket
        try:
            await self.ws.connect()
        except Exception as e:
            logger.error(f"Failed to connect WebSocket: {e}")
            logger.info("Running in poll mode (REST API) instead")
            await self._poll_loop()
            return

        # Register price callback
        self.ws.on_price_update(self._on_price_update)

        # Discover and subscribe to live sports markets
        tickers = self._discover_live_markets()
        if tickers:
            await self.ws.subscribe(tickers)

        # Listen for updates
        try:
            await self.ws.listen()
        except KeyboardInterrupt:
            logger.info("Shutdown requested")
        finally:
            await self.ws.disconnect()
            self.db.close()
            self._running = False

    async def _poll_loop(self):
        """Fallback: poll-based loop using REST API."""
        logger.info("Starting REST poll loop (10s interval)")
        try:
            while self._running:
                tickers = self._discover_live_markets()
                for ticker in tickers:
                    try:
                        market = self.kalshi.get_market(ticker)
                        if market:
                            yes_price = market.get("market", {}).get("yes_bid", 50)
                            prob = yes_price / 100.0
                            self._on_price_update(ticker, prob)
                    except Exception as e:
                        logger.error(f"Poll error for {ticker}: {e}")
                await asyncio.sleep(10)
        except KeyboardInterrupt:
            logger.info("Shutdown requested")
        finally:
            self.db.close()
            self._running = False

    def _discover_live_markets(self) -> list[str]:
        """Find live sports markets on Kalshi."""
        sports = self.config.get("trading", {}).get("sports", ["NCAAB", "ATP"])
        tickers = []

        for sport in sports:
            try:
                result = self.kalshi.get_markets(status="open")
                markets = result.get("markets", [])
                for m in markets:
                    ticker = m.get("ticker", "")
                    title = m.get("title", "").upper()
                    if sport.upper() in title or sport.lower() in ticker.lower():
                        tickers.append(ticker)
                        # Init game state if not tracked
                        if ticker not in self.active_games:
                            self._init_game_state(ticker, m, sport)
            except Exception as e:
                logger.error(f"Failed to discover {sport} markets: {e}")

        logger.info(f"Tracking {len(tickers)} live markets")
        return tickers

    def _init_game_state(self, ticker: str, market_data: dict, sport: str):
        """Initialize a GameState for a newly discovered market."""
        title = market_data.get("title", ticker)
        parts = title.split(" vs ") if " vs " in title else [title, "Unknown"]

        gs = GameState(
            game_id=ticker,
            sport=sport,
            team_a=parts[0].strip() if len(parts) > 0 else "Team A",
            team_b=parts[1].strip() if len(parts) > 1 else "Team B",
            start_time=time.time(),
            total_duration_est=7200,   # 2h default
            kalshi_ticker=ticker,
            is_live=True,
        )

        # Pre-populate with current price
        yes_bid = market_data.get("yes_bid", 50)
        gs.add_probability(0.0, yes_bid / 100.0)

        self.active_games[ticker] = gs
        logger.info(f"Tracking new game: {ticker} ({gs.team_a} vs {gs.team_b})")

    def _on_price_update(self, ticker: str, prob: float):
        """Handle a price update for a market."""
        game = self.active_games.get(ticker)
        if not game:
            return

        # Add probability snapshot
        elapsed = time.time() - game.start_time
        game.add_probability(elapsed, prob)

        # Check exit conditions for open positions
        self._check_exits(game, prob)

        # Evaluate for new entry signals
        if not self.portfolio.get_position_for_game(game.game_id):
            self._evaluate_entry(game)

    def _evaluate_entry(self, game: GameState):
        """Run the strategy router to check for entry signals."""
        if len(game.curve.snapshots) < 5:
            return  # need some history

        signal = self.router.evaluate(game)
        if signal is None:
            return

        # Risk check
        allowed, reason = self.risk.check_trade(signal, self.portfolio)
        if not allowed:
            logger.info(f"Trade blocked: {reason}")
            return

        # Open position
        trade = self.portfolio.open_position(signal)

        # Place order
        order_id = self.orders.place_entry_order(
            signal, trade.stake_usd, game.kalshi_ticker
        )
        trade.kalshi_order_id = order_id

        # Store in DB
        self.db.store_trade(trade)

        self._print_status()

    def _check_exits(self, game: GameState, current_prob: float):
        """Check if any open position on this game should be exited."""
        trade = self.portfolio.get_position_for_game(game.game_id)
        if not trade:
            return

        features = self.feature_engine.compute(game)
        should_exit = False

        if trade.regime == Regime.NON_CROSS:
            # Weak team prob
            weak_prob = features.prob_b_current if features.is_team_a_favorite else features.prob_a_current
            from src.data.models import NonCrossParams
            params = NonCrossParams(exit_multiplier=trade.exit_multiplier)
            should_exit = self.nc_strategy.should_exit(weak_prob, trade.entry_prob, params)

        elif trade.regime == Regime.CROSS:
            # Strong team prob
            strong_prob = features.prob_a_current if features.is_team_a_favorite else features.prob_b_current
            from src.data.models import CrossParams
            params = CrossParams(
                exit_multiplier=trade.exit_multiplier,
                exit_strategy=trade.exit_strategy,
            )
            should_exit = self.cr_strategy.should_exit(strong_prob, trade.entry_prob, params)

        if should_exit:
            exit_prob = current_prob
            elapsed = time.time() - game.start_time

            closed = self.portfolio.close_position(trade.trade_id, exit_prob, elapsed)
            self.risk.record_trade_result(closed.pnl_usd)

            # Place exit order
            self.orders.place_exit_order(closed, exit_prob, game.kalshi_ticker)

            # Update DB
            self.db.store_trade(closed)

            self._print_status()

    def _print_status(self):
        """Print current system status to console."""
        print("\n" + "=" * 60)
        print(f"  {time.strftime('%H:%M:%S')} | TRADING STATUS")
        print("-" * 60)
        print(f"  Games tracked: {len(self.active_games)}")
        print(f"  {self.portfolio.summary()}")
        print(f"  {self.risk.status()}")
        print(f"  Orders placed: {self.orders.order_count}")
        print("=" * 60 + "\n")
