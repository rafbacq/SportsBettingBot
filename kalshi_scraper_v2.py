"""
Kalshi Basketball Scraper v2 — Ticks + K-Lines + Probability/Odds
==================================================================
3-phase scraper for settled NBA & NCAA basketball markets on Kalshi.

Phase 1: Scrape ~10,000 settled market metadata
Phase 2: Batch-fetch candlestick (K-line) OHLC data for each market
Phase 3: Fetch tick-level trades for each market

Derived fields computed for betting algorithm:
  - implied_prob       = yes_price  (Kalshi contracts settle at $1.00)
  - odds_multiplier    = 1 / yes_price  (payout ratio if YES wins)
  - no_odds_multiplier = 1 / no_price   (payout ratio if NO wins)
  - edge_yes           = result_binary * odds_multiplier - 1
  - edge_no            = (1 - result_binary) * no_odds_multiplier - 1

Usage:
    python kalshi_scraper_v2.py                           # defaults
    python kalshi_scraper_v2.py --max-markets 5000        # fewer markets
    python kalshi_scraper_v2.py --candle-interval 60      # hourly candles
    python kalshi_scraper_v2.py --skip-ticks              # skip trade data
    python kalshi_scraper_v2.py --skip-candles            # skip k-lines

Outputs (in ./kalshi_data/):
    markets.csv              — flat market data + derived probabilities
    candlesticks.csv         — OHLC k-line rows (ticker, ts, O, H, L, C, vol)
    trades.csv               — tick-level trades (ticker, ts, price, size, side)
    decision_features.csv    — model-ready candle rows + flow/timing labels
    raw_markets.json         — full API responses
    scrape_summary.json      — run metadata & stats
"""

import requests
import pandas as pd
import json
import time
import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Optional

# ─── Configuration ──────────────────────────────────────────────────────────────

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
REQUEST_DELAY = 0.35  # seconds between requests
MAX_RETRIES = 3

# Basketball classification patterns
NBA_SERIES = ["kxnba", "nba"]
NCAA_SERIES = ["kxncaab", "kxncaam", "ncaab", "ncaam", "kxmm", "marchmad"]

NBA_TITLE_KW = [
    "nba",
    "pro basketball",
]

NCAA_TITLE_KW_STRICT = [
    "march madness",
    "college basketball",
    "ncaab",
    "ncaam",
]

NCAA_TITLE_KW_CONTEXTUAL = [
    "final four",
    "sweet sixteen",
    "elite eight",
    "round of 64",
    "round of 32",
    "first four",
]

BASKETBALL_CONTEXT_KW = [
    "basketball",
    "points",
    "rebounds",
    "assists",
    "three-pointers",
    "3-pointers",
    "double-double",
    "triple-double",
    "spread",
    "moneyline",
    "total points",
]


# ─── Helpers ────────────────────────────────────────────────────────────────────

def ts_now() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def safe_float(v) -> Optional[float]:
    """Parse dollar-string or numeric to float, or None."""
    if v is None:
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def iso_to_ts(value) -> Optional[int]:
    """Parse ISO timestamp strings into unix seconds."""
    if not value:
        return None
    try:
        return int(datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp())
    except (ValueError, TypeError):
        return None


def safe_div(numerator, denominator) -> Optional[float]:
    """Return numerator / denominator while protecting zero and missing values."""
    if numerator is None or denominator in (None, 0):
        return None
    try:
        return numerator / denominator
    except ZeroDivisionError:
        return None


def derive_result_fields(market: dict) -> dict:
    """
    Normalize Kalshi resolution states into model-friendly label metadata.

    `scalar` settlements on nominally binary props are not true YES/NO outcomes,
    so they should be excluded from supervised training labels.
    """
    result_raw = market.get("result")
    settlement_value = safe_float(market.get("settlement_value_dollars"))

    result_binary = None
    label_kind = "unresolved"
    label_usable = False

    if result_raw == "yes":
        result_binary = 1
        label_kind = "binary"
        label_usable = True
    elif result_raw in ("no", "all_no"):
        result_binary = 0
        label_kind = "binary"
        label_usable = True
    elif result_raw == "scalar":
        label_kind = "scalar"
    elif result_raw:
        label_kind = "other"

    return {
        "result": result_raw,
        "result_binary": result_binary,
        "label_kind": label_kind,
        "label_usable": label_usable,
        "settlement_value": settlement_value,
    }


def classify(market: dict) -> Optional[str]:
    """Return 'NBA', 'NCAA', or None."""
    series = (market.get("series_ticker") or "").lower()
    event = (market.get("event_ticker") or "").lower()
    title = (market.get("title") or "").lower()
    subtitle = (market.get("subtitle") or market.get("sub_title") or "").lower()
    blob = f"{series} {event} {title} {subtitle}"
    has_basketball_context = any(kw in blob for kw in BASKETBALL_CONTEXT_KW)

    for p in NCAA_SERIES:
        if p in series or p in event:
            return "NCAA"
    for p in NBA_SERIES:
        if p in series or p in event:
            return "NBA"
    for kw in NCAA_TITLE_KW_STRICT:
        if kw in blob:
            return "NCAA"
    for kw in NCAA_TITLE_KW_CONTEXTUAL:
        if kw in blob and has_basketball_context:
            return "NCAA"
    for kw in NBA_TITLE_KW:
        if kw in blob:
            return "NBA"
    return None


# ─── API Client ─────────────────────────────────────────────────────────────────

class KalshiAPI:
    def __init__(self, delay=REQUEST_DELAY):
        self.base = BASE_URL
        self.delay = delay
        self.s = requests.Session()
        self.s.headers["Accept"] = "application/json"
        self.s.headers["User-Agent"] = "KalshiScraper/2.0"
        self.request_count = 0

    def _get(self, path: str, params: dict = None) -> dict:
        time.sleep(self.delay)
        self.request_count += 1
        url = f"{self.base}{path}"
        for attempt in range(MAX_RETRIES):
            try:
                r = self.s.get(url, params=params, timeout=30)
                if r.status_code == 429:
                    wait = 10 * (attempt + 1)
                    print(f"    ⚠ Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                return r.json()
            except requests.exceptions.RequestException as e:
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(2 * (attempt + 1))
        return {}

    # ── Markets (paginated) ──────────────────────────────────────────────

    def get_markets_page(self, params: dict) -> tuple[list, str]:
        data = self._get("/markets", params)
        return data.get("markets", []), data.get("cursor", "")

    def get_all_markets(
        self,
        status="settled",
        series_ticker=None,
        min_settled_ts=None,
        max_settled_ts=None,
        max_total=None,
    ) -> list[dict]:
        all_m = []
        cursor = None
        while True:
            p = {"limit": 1000, "status": status, "mve_filter": "exclude"}
            if series_ticker:
                p["series_ticker"] = series_ticker
            if min_settled_ts:
                p["min_settled_ts"] = min_settled_ts
            if max_settled_ts:
                p["max_settled_ts"] = max_settled_ts
            if cursor:
                p["cursor"] = cursor

            batch, cursor = self.get_markets_page(p)
            all_m.extend(batch)
            if len(all_m) % 3000 < 1000:
                print(f"      ... {len(all_m)} markets so far")
            if not cursor or not batch:
                break
            if max_total and len(all_m) >= max_total:
                break
        return all_m

    # ── Candlesticks (batch — up to 100 tickers, 10k candles) ────────────

    def get_batch_candlesticks(
        self,
        tickers: list[str],
        start_ts: int,
        end_ts: int,
        period_interval: int = 60,
    ) -> dict:
        """
        Returns {ticker: [candle, ...]} for up to 100 tickers.
        Each candle has: end_period_ts, price.{open,high,low,close,mean}_dollars,
                         yes_bid/yes_ask OHLC, volume_fp, open_interest_fp.
        """
        if not tickers:
            return {}
        params = {
            "market_tickers": ",".join(tickers[:100]),
            "start_ts": start_ts,
            "end_ts": end_ts,
            "period_interval": period_interval,
            "include_latest_before_start": "true",
        }
        try:
            data = self._get("/markets/candlesticks", params)
        except Exception as e:
            print(f"    ⚠ Batch candlestick error: {e}")
            return {}

        result = {}
        for entry in data.get("markets", []):
            t = entry.get("market_ticker", "")
            result[t] = entry.get("candlesticks", [])
        return result

    # ── Single market candlesticks (fallback) ────────────────────────────

    def get_candlesticks(
        self,
        series_ticker: str,
        ticker: str,
        start_ts: int,
        end_ts: int,
        period_interval: int = 60,
    ) -> list[dict]:
        try:
            data = self._get(
                f"/series/{series_ticker}/markets/{ticker}/candlesticks",
                {
                    "start_ts": start_ts,
                    "end_ts": end_ts,
                    "period_interval": period_interval,
                },
            )
            return data.get("candlesticks", [])
        except Exception:
            return []

    # ── Trades / Ticks ───────────────────────────────────────────────────

    def get_trades(
        self,
        ticker: str,
        min_ts: int = None,
        max_ts: int = None,
        max_trades: int = 500,
    ) -> list[dict]:
        """Get tick-level trades for a single market."""
        all_t = []
        cursor = None
        while True:
            p = {"limit": 1000, "ticker": ticker}
            if min_ts:
                p["min_ts"] = min_ts
            if max_ts:
                p["max_ts"] = max_ts
            if cursor:
                p["cursor"] = cursor
            try:
                data = self._get("/markets/trades", p)
            except Exception:
                break
            trades = data.get("trades", [])
            all_t.extend(trades)
            cursor = data.get("cursor", "")
            if not cursor or not trades or len(all_t) >= max_trades:
                break
        return all_t[:max_trades]

    # ── Sports filter discovery ──────────────────────────────────────────

    def get_sports_filters(self) -> dict:
        try:
            return self._get("/search/filters_by_sport")
        except Exception:
            return {}


# ─── Phase 1: Collect settled basketball markets ────────────────────────────────

def phase1_collect_markets(api: KalshiAPI, days_back: int, max_markets: int) -> list[dict]:
    """
    Gather up to max_markets settled NBA + NCAA basketball markets.
    """
    print("\n" + "=" * 65)
    print("  PHASE 1: Collecting settled basketball markets")
    print("=" * 65)

    cutoff_ts = ts_now() - days_back * 86400
    basketball = []
    seen = set()

    # 1a — Discover series tickers from sports filter
    print("\n  🔍 Discovering series tickers from sports filter...")
    discovered_series = set()
    filters = api.get_sports_filters()
    sports = filters.get("filters_by_sports", {})
    for sport_name, sport_data in sports.items():
        if "basketball" not in sport_name.lower() and "hoops" not in sport_name.lower():
            continue
        print(f"     Found sport group: {sport_name}")
        blob = json.dumps(sport_data).lower()
        # Extract any series_ticker values
        if isinstance(sport_data, dict):
            for v in sport_data.values():
                if isinstance(v, list):
                    for item in v:
                        if isinstance(item, dict) and "series_ticker" in item:
                            discovered_series.add(item["series_ticker"])
    if discovered_series:
        print(f"     Series from filter: {discovered_series}")

    # 1b — Pull markets by discovered series tickers
    for st in discovered_series:
        print(f"\n  📥 Pulling settled markets for series={st}...")
        ms = api.get_all_markets(
            status="settled",
            series_ticker=st,
            min_settled_ts=cutoff_ts,
            max_total=max_markets,
        )
        for m in ms:
            t = m.get("ticker")
            if t and t not in seen:
                league = classify(m)
                if league:
                    m["_league"] = league
                    basketball.append(m)
                    seen.add(t)
        print(f"     → basketball matches: {len(basketball)}")
        if len(basketball) >= max_markets:
            break

    # 1c — Broad scan of all settled markets to catch stragglers
    if len(basketball) < max_markets:
        print(f"\n  📥 Broad scan of all settled markets (last {days_back} days)...")
        all_settled = api.get_all_markets(
            status="settled",
            min_settled_ts=cutoff_ts,
            max_total=max_markets * 5,  # oversample, then filter
        )
        print(f"     Total settled in window: {len(all_settled)}")
        for m in all_settled:
            t = m.get("ticker")
            if t and t not in seen:
                league = classify(m)
                if league:
                    m["_league"] = league
                    basketball.append(m)
                    seen.add(t)
            if len(basketball) >= max_markets:
                break
        print(f"     → basketball total after broad scan: {len(basketball)}")

    basketball = basketball[:max_markets]
    nba_count = sum(1 for m in basketball if m.get("_league") == "NBA")
    ncaa_count = len(basketball) - nba_count
    print(f"\n  ✅ Phase 1 complete: {len(basketball)} markets (NBA={nba_count}, NCAA={ncaa_count})")
    return basketball


# ─── Phase 2: Fetch candlestick (K-line) data ──────────────────────────────────

def _parse_candle(ticker: str, league: str, c: dict) -> dict:
    """Flatten one candlestick into a row with derived probability fields."""
    price = c.get("price", {})
    yes_bid = c.get("yes_bid", {})
    yes_ask = c.get("yes_ask", {})

    close_p = safe_float(price.get("close_dollars"))
    open_p = safe_float(price.get("open_dollars"))
    high_p = safe_float(price.get("high_dollars"))
    low_p = safe_float(price.get("low_dollars"))
    mean_p = safe_float(price.get("mean_dollars"))

    # Derived: on Kalshi, yes_price ≈ implied probability
    # odds_multiplier = payout if YES wins = 1 / yes_price
    implied_prob = close_p
    odds_yes = (1.0 / close_p) if close_p and close_p > 0 else None
    odds_no = (1.0 / (1.0 - close_p)) if close_p and close_p < 1.0 else None

    return {
        "ticker": ticker,
        "league": league,
        "end_period_ts": c.get("end_period_ts"),
        # Price OHLC (these ARE the implied probabilities, range 0-1)
        "price_open": open_p,
        "price_high": high_p,
        "price_low": low_p,
        "price_close": close_p,
        "price_mean": mean_p,
        "price_previous": safe_float(price.get("previous_dollars")),
        # Yes bid/ask spread
        "yes_bid_open": safe_float(yes_bid.get("open_dollars")),
        "yes_bid_close": safe_float(yes_bid.get("close_dollars")),
        "yes_ask_open": safe_float(yes_ask.get("open_dollars")),
        "yes_ask_close": safe_float(yes_ask.get("close_dollars")),
        # Volume & OI
        "volume": safe_float(c.get("volume_fp")),
        "open_interest": safe_float(c.get("open_interest_fp")),
        # ── Derived for betting algo ──
        "implied_prob": implied_prob,         # = close price
        "odds_multiplier_yes": odds_yes,      # 1/p  — what you multiply buying YES
        "odds_multiplier_no": odds_no,        # 1/(1-p) — what you multiply buying NO
        # Spread = ask - bid (tightness of market)
        "spread": (
            safe_float(yes_ask.get("close_dollars") or 0) -
            safe_float(yes_bid.get("close_dollars") or 0)
        ) if yes_ask.get("close_dollars") and yes_bid.get("close_dollars") else None,
    }


def phase2_candlesticks(
    api: KalshiAPI,
    markets: list[dict],
    period_interval: int = 60,
) -> list[dict]:
    """
    Fetch K-line data for all markets using the batch endpoint.
    Batches of 100 tickers at a time.
    """
    print("\n" + "=" * 65)
    print(f"  PHASE 2: Fetching candlestick data ({period_interval}min intervals)")
    print("=" * 65)

    all_candle_rows = []
    tickers_with_ts = []

    # Prepare: for each market, compute its time window
    for m in markets:
        ticker = m["ticker"]
        league = m.get("_league", "")
        # Use open_time → close_time (or expiration) as the candle window
        open_t = m.get("open_time")
        close_t = m.get("close_time") or m.get("expiration_time") or m.get("latest_expiration_time")
        if not open_t or not close_t:
            continue

        try:
            start = int(datetime.fromisoformat(open_t.replace("Z", "+00:00")).timestamp())
            end = int(datetime.fromisoformat(close_t.replace("Z", "+00:00")).timestamp())
        except Exception:
            continue

        tickers_with_ts.append({
            "ticker": ticker,
            "league": league,
            "series_ticker": m.get("series_ticker", ""),
            "start_ts": start,
            "end_ts": end,
        })

    print(f"  Markets with valid time windows: {len(tickers_with_ts)}")

    tickers_with_ts = sorted(tickers_with_ts, key=lambda row: (row["start_ts"], row["end_ts"]))

    # Kalshi's batch endpoint is effectively limited by total candle volume, not just ticker count.
    # Small dynamic batches avoid the "0 candles returned" failure mode at 1-minute granularity.
    max_batch_tickers = 100
    max_estimated_candles = 8000
    batches = []
    current_batch = []
    current_estimated = 0
    for item in tickers_with_ts:
        estimated_candles = max(1, ((item["end_ts"] - item["start_ts"]) // (period_interval * 60)) + 1)
        would_overflow = (
            current_batch
            and (
                len(current_batch) >= max_batch_tickers
                or current_estimated + estimated_candles > max_estimated_candles
            )
        )
        if would_overflow:
            batches.append(current_batch)
            current_batch = []
            current_estimated = 0

        current_batch.append(item)
        current_estimated += estimated_candles

    if current_batch:
        batches.append(current_batch)

    for batch_num, batch in enumerate(batches, start=1):
        batch_tickers = [b["ticker"] for b in batch]
        batch_start = min(b["start_ts"] for b in batch)
        batch_end = max(b["end_ts"] for b in batch)
        league_map = {b["ticker"]: b["league"] for b in batch}

        print(
            f"    Batch {batch_num}/{len(batches)}: "
            f"{len(batch_tickers)} tickers, window {batch_start}->{batch_end}"
        )

        result = api.get_batch_candlesticks(
            tickers=batch_tickers,
            start_ts=batch_start,
            end_ts=batch_end,
            period_interval=period_interval,
        )

        # If a large batch returns no candles, retry each ticker individually.
        if batch_tickers and not any(result.get(ticker) for ticker in batch_tickers):
            print("      Batch returned no candles, retrying tickers individually...")
            result = {}
            for item in batch:
                single_result = api.get_batch_candlesticks(
                    tickers=[item["ticker"]],
                    start_ts=item["start_ts"],
                    end_ts=item["end_ts"],
                    period_interval=period_interval,
                )
                result.update(single_result)

        batch_candles = 0
        for ticker, candles in result.items():
            lg = league_map.get(ticker, "")
            for c in candles:
                row = _parse_candle(ticker, lg, c)
                all_candle_rows.append(row)
                batch_candles += 1

        if batch_num % 10 == 0 or batch_num == len(batches):
            print(f"      → batch candles: {batch_candles}, cumulative: {len(all_candle_rows)}")

    print(f"\n  ✅ Phase 2 complete: {len(all_candle_rows)} candlestick rows")
    return all_candle_rows


# ─── Phase 3: Fetch tick-level trade data ───────────────────────────────────────

def phase3_ticks(
    api: KalshiAPI,
    markets: list[dict],
    max_trades_per_market: int = 200,
) -> list[dict]:
    """
    Fetch individual trades for each market.
    Each trade = one tick with price, size, taker_side.
    """
    print("\n" + "=" * 65)
    print(f"  PHASE 3: Fetching tick-level trades (up to {max_trades_per_market}/market)")
    print("=" * 65)

    all_tick_rows = []
    total = len(markets)

    for idx, m in enumerate(markets):
        ticker = m["ticker"]
        league = m.get("_league", "")

        # Get time window
        open_t = m.get("open_time")
        close_t = m.get("close_time") or m.get("expiration_time")
        min_ts = None
        max_ts = None
        try:
            if open_t:
                min_ts = int(datetime.fromisoformat(open_t.replace("Z", "+00:00")).timestamp())
            if close_t:
                max_ts = int(datetime.fromisoformat(close_t.replace("Z", "+00:00")).timestamp())
        except Exception:
            pass

        trades = api.get_trades(
            ticker=ticker,
            min_ts=min_ts,
            max_ts=max_ts,
            max_trades=max_trades_per_market,
        )

        for t in trades:
            yes_p = safe_float(t.get("yes_price_dollars"))
            no_p = safe_float(t.get("no_price_dollars"))
            count = safe_float(t.get("count_fp"))

            all_tick_rows.append({
                "ticker": ticker,
                "league": league,
                "trade_id": t.get("trade_id"),
                "created_time": t.get("created_time"),
                "yes_price": yes_p,
                "no_price": no_p,
                "count": count,
                "taker_side": t.get("taker_side"),  # "yes" or "no"
                # Derived
                "implied_prob": yes_p,
                "odds_multiplier_yes": (1.0 / yes_p) if yes_p and yes_p > 0 else None,
                "odds_multiplier_no": (1.0 / no_p) if no_p and no_p > 0 else None,
                "notional": (count * 1.0) if count else None,  # $1 per contract
            })

        if (idx + 1) % 200 == 0 or idx == total - 1:
            print(f"    [{idx+1}/{total}] total ticks: {len(all_tick_rows)}")

    print(f"\n  ✅ Phase 3 complete: {len(all_tick_rows)} trade ticks")
    return all_tick_rows


# ─── Flatten market metadata with derived betting fields ────────────────────────

def flatten_market(m: dict) -> dict:
    """Convert raw market dict to flat row with betting-algo–ready fields."""
    derived_league = classify(m) or m.get("_league", "")
    result_info = derive_result_fields(m)
    last_p = safe_float(m.get("last_price_dollars"))
    yes_bid = safe_float(m.get("yes_bid_dollars"))
    yes_ask = safe_float(m.get("yes_ask_dollars"))
    no_bid = safe_float(m.get("no_bid_dollars"))
    no_ask = safe_float(m.get("no_ask_dollars"))
    volume = safe_float(m.get("volume_fp"))
    oi = safe_float(m.get("open_interest_fp"))
    result_raw = result_info["result"]
    result_binary = result_info["result_binary"]

    # Implied probability at last trade
    implied_prob = last_p

    # Odds multipliers
    odds_yes = (1.0 / last_p) if last_p and last_p > 0 else None
    odds_no = (1.0 / (1.0 - last_p)) if last_p and last_p < 1.0 else None

    # Retrospective edge: if you bought YES at last_price, what was your P&L per $1?
    # edge_yes = result_binary * (1/price) - 1  → positive means profitable
    edge_yes = None
    edge_no = None
    if result_binary is not None and odds_yes is not None:
        edge_yes = result_binary * odds_yes - 1.0
    if result_binary is not None and odds_no is not None:
        edge_no = (1 - result_binary) * odds_no - 1.0

    return {
        "ticker": m.get("ticker"),
        "event_ticker": m.get("event_ticker"),
        "series_ticker": m.get("series_ticker", ""),
        "league": derived_league,
        "title": m.get("title", ""),
        "subtitle": m.get("subtitle", m.get("sub_title", "")),
        "yes_sub_title": m.get("yes_sub_title", ""),
        "no_sub_title": m.get("no_sub_title", ""),
        "market_type": m.get("market_type"),
        "status": m.get("status"),
        # ── Result ──
        "result": result_raw,
        "label_kind": result_info["label_kind"],
        "label_usable": result_info["label_usable"],
        "result_binary": result_binary,        # 1=YES won, 0=NO won
        "settlement_value": result_info["settlement_value"],
        # ── Prices ──
        "last_price": last_p,                  # = implied probability at close
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "no_bid": no_bid,
        "no_ask": no_ask,
        # ── Probability & Odds (your algo's core inputs) ──
        "implied_prob_close": implied_prob,     # P(YES) at market close
        "odds_multiplier_yes": odds_yes,        # payout ratio buying YES
        "odds_multiplier_no": odds_no,          # payout ratio buying NO
        "edge_yes": edge_yes,                   # retrospective: profit/loss per $1 YES
        "edge_no": edge_no,                     # retrospective: profit/loss per $1 NO
        # ── Volume / Liquidity ──
        "volume": volume,
        "open_interest": oi,
        "spread": (yes_ask - yes_bid) if yes_ask is not None and yes_bid is not None else None,
        "notional_value": safe_float(m.get("notional_value_dollars")),
        # ── Timestamps ──
        "open_time": m.get("open_time"),
        "close_time": m.get("close_time"),
        "expiration_time": m.get("expiration_time") or m.get("latest_expiration_time"),
        "settlement_ts": m.get("settlement_ts"),
        "created_time": m.get("created_time"),
        # ── Strike / line info ──
        "strike_type": m.get("strike_type"),
        "floor_strike": m.get("floor_strike"),
        "cap_strike": m.get("cap_strike"),
        "functional_strike": m.get("functional_strike"),
        "can_close_early": m.get("can_close_early"),
        "rules_primary": (m.get("rules_primary") or "")[:300],
    }


# ─── Decision-feature engineering ───────────────────────────────────────────────

def build_trade_interval_features(
    tick_rows: list[dict],
    candle_interval_minutes: int,
) -> pd.DataFrame:
    """
    Aggregate raw trades into candle-aligned flow features.
    These features help identify whether aggressive YES/NO flow preceded a move.
    """
    columns = [
        "ticker",
        "end_period_ts",
        "trade_count",
        "trade_contracts",
        "yes_trade_contracts",
        "no_trade_contracts",
        "trade_imbalance",
        "trade_vwap_yes",
        "avg_trade_size",
        "max_trade_size",
        "last_trade_yes_price",
        "last_trade_no_price",
    ]
    if not tick_rows:
        return pd.DataFrame(columns=columns)

    df_t = pd.DataFrame(tick_rows).copy()
    if df_t.empty or "created_time" not in df_t.columns:
        return pd.DataFrame(columns=columns)

    df_t["created_dt"] = pd.to_datetime(df_t["created_time"], utc=True, errors="coerce")
    df_t = df_t.dropna(subset=["created_dt", "ticker"])
    if df_t.empty:
        return pd.DataFrame(columns=columns)

    interval_seconds = candle_interval_minutes * 60
    epoch_seconds = (df_t["created_dt"].astype("int64") // 10**9).astype("int64")
    df_t["end_period_ts"] = ((epoch_seconds // interval_seconds) + 1) * interval_seconds

    df_t["trade_rows"] = 1
    df_t["contracts"] = pd.to_numeric(df_t["count"], errors="coerce").fillna(0.0)
    df_t["yes_price"] = pd.to_numeric(df_t["yes_price"], errors="coerce")
    df_t["no_price"] = pd.to_numeric(df_t["no_price"], errors="coerce")
    df_t["yes_trade_contracts"] = df_t["contracts"].where(df_t["taker_side"] == "yes", 0.0)
    df_t["no_trade_contracts"] = df_t["contracts"].where(df_t["taker_side"] == "no", 0.0)
    df_t["trade_vwap_numerator"] = df_t["yes_price"].fillna(0.0) * df_t["contracts"]

    df_t = df_t.sort_values(["ticker", "end_period_ts", "created_dt"])
    grouped = df_t.groupby(["ticker", "end_period_ts"], as_index=False).agg(
        trade_count=("trade_rows", "sum"),
        trade_contracts=("contracts", "sum"),
        yes_trade_contracts=("yes_trade_contracts", "sum"),
        no_trade_contracts=("no_trade_contracts", "sum"),
        trade_vwap_numerator=("trade_vwap_numerator", "sum"),
        avg_trade_size=("contracts", "mean"),
        max_trade_size=("contracts", "max"),
        last_trade_yes_price=("yes_price", "last"),
        last_trade_no_price=("no_price", "last"),
    )

    grouped["trade_imbalance"] = grouped.apply(
        lambda row: safe_div(
            row["yes_trade_contracts"] - row["no_trade_contracts"],
            row["trade_contracts"],
        ),
        axis=1,
    )
    grouped["trade_vwap_yes"] = grouped.apply(
        lambda row: safe_div(row["trade_vwap_numerator"], row["trade_contracts"]),
        axis=1,
    )

    return grouped[columns]


def build_decision_features(
    markets: list[dict],
    candle_rows: list[dict],
    tick_rows: list[dict],
    candle_interval_minutes: int,
) -> pd.DataFrame:
    """
    Build one row per candle with timing, liquidity, momentum, and outcome labels.
    This is the dataset you can backtest entry rules on.
    """
    if not candle_rows:
        return pd.DataFrame()

    df_c = pd.DataFrame(candle_rows).copy()
    if df_c.empty:
        return pd.DataFrame()

    market_meta = []
    for m in markets:
        derived_league = classify(m)
        if not derived_league:
            continue
        result_info = derive_result_fields(m)

        close_ts = iso_to_ts(
            m.get("close_time") or m.get("expiration_time") or m.get("latest_expiration_time")
        )
        open_ts = iso_to_ts(m.get("open_time"))

        market_meta.append({
            "ticker": m.get("ticker"),
            "league": derived_league,
            "title": m.get("title", ""),
            "market_type": m.get("market_type"),
            "market_open_ts": open_ts,
            "market_close_ts": close_ts,
            "market_duration_minutes": safe_div(
                (close_ts - open_ts) if close_ts and open_ts else None,
                60,
            ),
            "market_close_prob": safe_float(m.get("last_price_dollars")),
            "label_kind": result_info["label_kind"],
            "label_usable": result_info["label_usable"],
            "result_binary": result_info["result_binary"],
        })

    df_meta = pd.DataFrame(market_meta).drop_duplicates(subset=["ticker"])
    valid_tickers = set(df_meta["ticker"].dropna())

    df_c["end_period_ts"] = pd.to_numeric(df_c["end_period_ts"], errors="coerce")
    df_c = df_c[df_c["ticker"].isin(valid_tickers)].copy()
    if df_c.empty:
        return pd.DataFrame()

    for col in [
        "price_open",
        "price_high",
        "price_low",
        "price_close",
        "price_mean",
        "price_previous",
        "volume",
        "open_interest",
        "spread",
        "yes_bid_close",
        "yes_ask_close",
    ]:
        if col in df_c.columns:
            df_c[col] = pd.to_numeric(df_c[col], errors="coerce")

    df = df_c.merge(df_meta, on="ticker", how="left", suffixes=("", "_market"))
    if "league_market" in df.columns:
        df["league"] = df["league"].fillna(df["league_market"])
        df = df.drop(columns=["league_market"])

    df = df.sort_values(["ticker", "end_period_ts"]).reset_index(drop=True)

    by_ticker_close = df.groupby("ticker")["price_close"]
    by_ticker_volume = df.groupby("ticker")["volume"]

    df["candle_return_1"] = by_ticker_close.diff()
    df["candle_return_3"] = by_ticker_close.transform(lambda s: s - s.shift(3))
    df["candle_return_5"] = by_ticker_close.transform(lambda s: s - s.shift(5))
    df["rolling_price_mean_5"] = by_ticker_close.transform(
        lambda s: s.rolling(5, min_periods=1).mean()
    )
    df["rolling_volatility_5"] = by_ticker_close.transform(
        lambda s: s.rolling(5, min_periods=2).std()
    )
    df["rolling_volume_5"] = by_ticker_volume.transform(
        lambda s: s.rolling(5, min_periods=1).sum()
    )
    df["rolling_volume_mean_5"] = by_ticker_volume.transform(
        lambda s: s.rolling(5, min_periods=1).mean()
    )

    df["candle_range"] = df["price_high"] - df["price_low"]
    df["candle_body"] = df["price_close"] - df["price_open"]
    df["minutes_to_close"] = (df["market_close_ts"] - df["end_period_ts"]) / 60.0
    df["minutes_from_open"] = (df["end_period_ts"] - df["market_open_ts"]) / 60.0
    df["market_progress"] = df.apply(
        lambda row: safe_div(row["minutes_from_open"], row["market_duration_minutes"]),
        axis=1,
    )
    df["price_vs_rolling_mean_5"] = df["price_close"] - df["rolling_price_mean_5"]
    df["volume_vs_rolling_mean_5"] = df.apply(
        lambda row: safe_div(row["volume"], row["rolling_volume_mean_5"]),
        axis=1,
    )
    df["relative_spread"] = df.apply(
        lambda row: safe_div(row["spread"], row["price_close"]),
        axis=1,
    )
    df["distance_to_close_prob"] = df["market_close_prob"] - df["price_close"]

    df["realized_edge_yes_entry"] = df.apply(
        lambda row: (
            row["result_binary"] * (1.0 / row["price_close"]) - 1.0
            if pd.notna(row["result_binary"])
            and pd.notna(row["price_close"])
            and row["price_close"] > 0
            else None
        ),
        axis=1,
    )
    df["realized_edge_no_entry"] = df.apply(
        lambda row: (
            (1.0 - row["result_binary"]) * (1.0 / (1.0 - row["price_close"])) - 1.0
            if pd.notna(row["result_binary"])
            and pd.notna(row["price_close"])
            and row["price_close"] < 1.0
            else None
        ),
        axis=1,
    )
    df["label_yes_win"] = df["result_binary"]

    trade_features = build_trade_interval_features(tick_rows, candle_interval_minutes)
    if not trade_features.empty:
        trade_features = trade_features[trade_features["ticker"].isin(valid_tickers)].copy()
        df = df.merge(trade_features, on=["ticker", "end_period_ts"], how="left")
    else:
        for col in [
            "trade_count",
            "trade_contracts",
            "yes_trade_contracts",
            "no_trade_contracts",
            "trade_imbalance",
            "trade_vwap_yes",
            "avg_trade_size",
            "max_trade_size",
            "last_trade_yes_price",
            "last_trade_no_price",
        ]:
            df[col] = None

    for col in [
        "trade_count",
        "trade_contracts",
        "yes_trade_contracts",
        "no_trade_contracts",
        "avg_trade_size",
        "max_trade_size",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["trade_imbalance"] = pd.to_numeric(df["trade_imbalance"], errors="coerce").fillna(0.0)
    df["has_trade_flow"] = df["trade_count"] > 0
    df["trade_vwap_edge"] = df["trade_vwap_yes"] - df["price_close"]

    preferred_order = [
        "ticker",
        "league",
        "title",
        "market_type",
        "end_period_ts",
        "market_open_ts",
        "market_close_ts",
        "minutes_from_open",
        "minutes_to_close",
        "market_progress",
        "price_open",
        "price_high",
        "price_low",
        "price_close",
        "price_mean",
        "price_previous",
        "candle_body",
        "candle_range",
        "candle_return_1",
        "candle_return_3",
        "candle_return_5",
        "rolling_price_mean_5",
        "rolling_volatility_5",
        "price_vs_rolling_mean_5",
        "volume",
        "rolling_volume_5",
        "rolling_volume_mean_5",
        "volume_vs_rolling_mean_5",
        "open_interest",
        "spread",
        "relative_spread",
        "trade_count",
        "trade_contracts",
        "yes_trade_contracts",
        "no_trade_contracts",
        "trade_imbalance",
        "trade_vwap_yes",
        "trade_vwap_edge",
        "avg_trade_size",
        "max_trade_size",
        "last_trade_yes_price",
        "last_trade_no_price",
        "has_trade_flow",
        "market_close_prob",
        "distance_to_close_prob",
        "label_kind",
        "label_usable",
        "result_binary",
        "label_yes_win",
        "realized_edge_yes_entry",
        "realized_edge_no_entry",
    ]
    ordered_columns = [col for col in preferred_order if col in df.columns]
    remaining_columns = [col for col in df.columns if col not in ordered_columns]
    return df[ordered_columns + remaining_columns]


# ─── Save everything ────────────────────────────────────────────────────────────

def save_all(
    markets: list[dict],
    candle_rows: list[dict],
    tick_rows: list[dict],
    output_dir: str,
    api_requests: int,
    candle_interval_minutes: int,
):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Markets
    flat = [flatten_market(m) for m in markets]
    df_m = pd.DataFrame(flat)
    if not df_m.empty and "close_time" in df_m.columns:
        df_m = df_m.sort_values("close_time", ascending=False)
    df_m.to_csv(os.path.join(output_dir, "markets.csv"), index=False)
    print(f"  ✓ markets.csv              ({len(df_m)} rows)")

    # 2. Candlesticks
    if candle_rows:
        df_c = pd.DataFrame(candle_rows)
        if "end_period_ts" in df_c.columns:
            df_c = df_c.sort_values(["ticker", "end_period_ts"])
        df_c.to_csv(os.path.join(output_dir, "candlesticks.csv"), index=False)
        print(f"  ✓ candlesticks.csv         ({len(df_c)} rows)")
    else:
        candle_path = os.path.join(output_dir, "candlesticks.csv")
        if os.path.exists(candle_path):
            os.remove(candle_path)
        print(f"  ⚠ candlesticks.csv         (skipped)")

    # 3. Trades / ticks
    if tick_rows:
        df_t = pd.DataFrame(tick_rows)
        if "created_time" in df_t.columns:
            df_t = df_t.sort_values(["ticker", "created_time"])
        df_t.to_csv(os.path.join(output_dir, "trades.csv"), index=False)
        print(f"  ✓ trades.csv               ({len(df_t)} rows)")
    else:
        print(f"  ⚠ trades.csv               (skipped)")

    # 4. Raw JSON
    raw_path = os.path.join(output_dir, "raw_markets.json")
    with open(raw_path, "w") as f:
        json.dump(markets, f, indent=2, default=str)
    print(f"  ✓ raw_markets.json         (full API data)")

    # 5. Decision features
    decision_df = build_decision_features(
        markets=markets,
        candle_rows=candle_rows,
        tick_rows=tick_rows,
        candle_interval_minutes=candle_interval_minutes,
    )
    if not decision_df.empty:
        decision_df.to_csv(os.path.join(output_dir, "decision_features.csv"), index=False)
        print(f"  ✓ decision_features.csv    ({len(decision_df)} rows)")
    else:
        decision_path = os.path.join(output_dir, "decision_features.csv")
        if os.path.exists(decision_path):
            os.remove(decision_path)
        print(f"  ⚠ decision_features.csv    (requires candlestick data)")

    # 6. Summary
    nba_count = sum(1 for m in flat if m.get("league") == "NBA")
    ncaa_count = sum(1 for m in flat if m.get("league") == "NCAA")
    summary = {
        "scrape_time": datetime.now(timezone.utc).isoformat(),
        "total_markets": len(flat),
        "nba_markets": nba_count,
        "ncaa_markets": ncaa_count,
        "label_usable_markets": int(sum(1 for m in flat if m.get("label_usable"))),
        "scalar_settlement_markets": int(sum(1 for m in flat if m.get("label_kind") == "scalar")),
        "candlestick_rows": len(candle_rows),
        "trade_tick_rows": len(tick_rows),
        "decision_feature_rows": len(decision_df),
        "candle_interval_minutes": candle_interval_minutes,
        "api_requests_made": api_requests,
        "unique_series_tickers": list(set(m.get("series_ticker", "") for m in flat if m.get("series_ticker"))),
    }
    with open(os.path.join(output_dir, "scrape_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ scrape_summary.json")


# ─── CLI ────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Kalshi Basketball Scraper v2")
    p.add_argument("--max-markets", "-m", type=int, default=10000,
                   help="Max markets to collect (default: 10000)")
    p.add_argument("--days-back", "-d", type=int, default=180,
                   help="How far back to look (default: 180)")
    p.add_argument("--candle-interval", "-c", type=int, default=60,
                   choices=[1, 60, 1440],
                   help="Candlestick interval in minutes: 1, 60, 1440 (default: 60)")
    p.add_argument("--max-ticks", type=int, default=200,
                   help="Max trades per market (default: 200)")
    p.add_argument("--skip-candles", action="store_true",
                   help="Skip candlestick fetching")
    p.add_argument("--skip-ticks", action="store_true",
                   help="Skip trade/tick fetching")
    p.add_argument("--output-dir", "-o", default="./kalshi_data",
                   help="Output directory (default: ./kalshi_data)")
    args = p.parse_args()

    print("╔" + "═" * 63 + "╗")
    print("║  Kalshi Basketball Scraper v2 — Ticks + K-Lines + Odds       ║")
    print("╠" + "═" * 63 + "╣")
    print(f"║  Max markets:      {args.max_markets:<43}║")
    print(f"║  Days back:        {args.days_back:<43}║")
    print(f"║  Candle interval:  {args.candle_interval}min{' ':39}║")
    print(f"║  Max ticks/market: {args.max_ticks:<43}║")
    print(f"║  Skip candles:     {str(args.skip_candles):<43}║")
    print(f"║  Skip ticks:       {str(args.skip_ticks):<43}║")
    print(f"║  Output:           {args.output_dir:<43}║")
    print("╚" + "═" * 63 + "╝")

    api = KalshiAPI()

    # Phase 1
    markets = phase1_collect_markets(api, args.days_back, args.max_markets)
    if not markets:
        print("\n  ✗ No basketball markets found. Exiting.")
        sys.exit(1)

    # Phase 2
    candle_rows = []
    if not args.skip_candles:
        candle_rows = phase2_candlesticks(api, markets, args.candle_interval)

    # Phase 3
    tick_rows = []
    if not args.skip_ticks:
        tick_rows = phase3_ticks(api, markets, args.max_ticks)

    # Save
    print("\n" + "=" * 65)
    print("  SAVING RESULTS")
    print("=" * 65)
    save_all(
        markets,
        candle_rows,
        tick_rows,
        args.output_dir,
        api.request_count,
        args.candle_interval,
    )

    # Final summary
    nba = sum(1 for m in markets if m.get("_league") == "NBA")
    ncaa = len(markets) - nba
    print("\n╔" + "═" * 63 + "╗")
    print("║  ✅ SCRAPE COMPLETE" + " " * 44 + "║")
    print(f"║  NBA:  {nba:<55}║")
    print(f"║  NCAA: {ncaa:<55}║")
    print(f"║  Candlestick rows: {len(candle_rows):<43}║")
    print(f"║  Trade tick rows:  {len(tick_rows):<43}║")
    print(f"║  API requests:     {api.request_count:<43}║")
    print("╚" + "═" * 63 + "╝")


if __name__ == "__main__":
    main()
