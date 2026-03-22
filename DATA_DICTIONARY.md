# Kalshi Scraper v2 — Output Data Dictionary

## Quick Start

```bash
pip install requests pandas

# Full scrape: ~10k markets + hourly candles + ticks
python3 kalshi_scraper_v2.py

# Fast mode: markets only, no candle/tick data
python3 kalshi_scraper_v2.py --skip-candles --skip-ticks

# Fine-grained: 5k markets, 1-minute candles, 500 ticks each
python3 kalshi_scraper_v2.py -m 5000 --candle-interval 1 --max-ticks 500
```

---

## How Kalshi Pricing Maps to Probability & Odds

Kalshi contracts settle at **$1.00** if YES, **$0.00** if NO.

```
yes_price = $0.35  →  market says 35% chance YES wins
                   →  if you buy YES at $0.35 and it wins, you get $1.00
                   →  odds_multiplier_yes = 1/0.35 = 2.86×  (your money × 2.86)
                   →  odds_multiplier_no  = 1/0.65 = 1.54×

If YES wins:   profit = $1.00 - $0.35 = $0.65  per contract
If NO  wins:   profit = $1.00 - $0.65 = $0.35  per contract
```

Your betting algo core equation:

```
expected_value_yes = P(actual_win) × odds_multiplier_yes - 1
expected_value_no  = P(actual_loss) × odds_multiplier_no  - 1

Bet YES when: your_estimated_prob > implied_prob (market underprices YES)
Bet NO  when: your_estimated_prob < implied_prob (market overprices YES)
```

---

## File: `markets.csv`

One row per settled market. This is your primary dataset.

| Column | Type | Description |
|--------|------|-------------|
| `ticker` | str | Unique market ID (e.g., `KXNBA-25MAR21-LAL-BOS`) |
| `event_ticker` | str | Parent event/game ID |
| `series_ticker` | str | Series grouping (e.g., `KXNBA`) |
| `league` | str | `NBA` or `NCAA` |
| `title` | str | Full market description |
| `result` | str | Raw result: `yes`, `no`, `all_no` |
| `label_kind` | str | `binary`, `scalar`, `unresolved`, or `other` |
| `label_usable` | bool | `True` only when the market resolved to a clean YES/NO label |
| **`result_binary`** | **int** | **1 = YES won, 0 = NO won** |
| **`implied_prob_close`** | **float** | **Market's closing probability for YES (0–1)** |
| **`odds_multiplier_yes`** | **float** | **Payout multiplier if you buy YES = 1/price** |
| **`odds_multiplier_no`** | **float** | **Payout multiplier if you buy NO = 1/(1-price)** |
| **`edge_yes`** | **float** | **Retrospective P&L: result × odds_yes - 1** |
| **`edge_no`** | **float** | **Retrospective P&L: (1-result) × odds_no - 1** |
| `last_price` | float | Final trade price (= implied_prob_close) |
| `yes_bid` / `yes_ask` | float | Final bid/ask for YES contracts |
| `no_bid` / `no_ask` | float | Final bid/ask for NO contracts |
| `spread` | float | yes_ask - yes_bid (market tightness) |
| `volume` | float | Total contracts traded |
| `open_interest` | float | Outstanding contracts at close |
| `settlement_value` | float | Settlement payout per contract |
| `functional_strike` | str | Spread/total line (for spread & totals markets) |
| `strike_type` | str | `greater`, `less`, etc. |
| `open_time` | ISO | When trading opened |
| `close_time` | ISO | When trading closed |

### How to use `edge_yes` / `edge_no`

These are **retrospective** — they tell you what would have happened if you bought at the closing price.

- `edge_yes = +1.86` → buying YES at close would have returned +186% (it won, odds were 2.86×)
- `edge_yes = -1.00` → buying YES at close would have lost 100% (it lost)
- `edge_no = +0.54`  → buying NO at close would have returned +54%

Filter for profitable patterns:
```python
# Markets where YES was underpriced (won at high odds)
df[(df.result_binary == 1) & (df.odds_multiplier_yes > 2.0)]

# Markets where favorite (high prob) still won
df[(df.result_binary == 1) & (df.implied_prob_close > 0.7)]
```

---

## File: `candlesticks.csv`

K-line / OHLC data. Multiple rows per market (one per time period).

| Column | Type | Description |
|--------|------|-------------|
| `ticker` | str | Market ID |
| `league` | str | NBA / NCAA |
| `end_period_ts` | int | Unix timestamp for candle period end |
| **`price_open`** | **float** | **Opening probability for the period** |
| **`price_high`** | **float** | **Highest probability during period** |
| **`price_low`** | **float** | **Lowest probability during period** |
| **`price_close`** | **float** | **Closing probability for period** |
| `price_mean` | float | Mean probability during period |
| `price_previous` | float | Previous period's closing price |
| `yes_bid_open/close` | float | Bid side OHLC |
| `yes_ask_open/close` | float | Ask side OHLC |
| `volume` | float | Contracts traded in this period |
| `open_interest` | float | Open interest at period end |
| **`implied_prob`** | **float** | **= price_close (convenience alias)** |
| **`odds_multiplier_yes`** | **float** | **1 / price_close** |
| **`odds_multiplier_no`** | **float** | **1 / (1 - price_close)** |
| **`spread`** | **float** | **ask_close - bid_close** |

### K-line analysis patterns

```python
# Load candles for a specific game
game = df_candles[df_candles.ticker == 'KXNBA-25MAR21-LAL-BOS']
game = game.sort_values('end_period_ts')

# Probability trajectory over the game
plt.plot(game.end_period_ts, game.implied_prob)

# Volatility = high - low range per candle
game['volatility'] = game.price_high - game.price_low

# Momentum = close - open
game['momentum'] = game.price_close - game.price_open

# Find "odds collapse" moments (big probability swing)
big_swings = game[game.volatility > 0.15]
```

---

## File: `trades.csv`

Tick-level individual trades. Finest granularity available.

| Column | Type | Description |
|--------|------|-------------|
| `ticker` | str | Market ID |
| `league` | str | NBA / NCAA |
| `trade_id` | str | Unique trade ID |
| `created_time` | ISO | Trade execution timestamp |
| **`yes_price`** | **float** | **Price of YES contract (= implied prob)** |
| **`no_price`** | **float** | **Price of NO contract (= 1 - yes_price)** |
| `count` | float | Number of contracts in this trade |
| **`taker_side`** | **str** | **`yes` or `no` — which side initiated** |
| `implied_prob` | float | = yes_price |
| `odds_multiplier_yes` | float | 1 / yes_price |
| `odds_multiplier_no` | float | 1 / no_price |
| `notional` | float | Dollar notional (count × $1) |

### Tick analysis patterns

```python
# Taker-side imbalance (buy pressure indicator)
trades = df_trades[df_trades.ticker == 'SOME_TICKER']
yes_volume = trades[trades.taker_side == 'yes']['count'].sum()
no_volume = trades[trades.taker_side == 'no']['count'].sum()
imbalance = (yes_volume - no_volume) / (yes_volume + no_volume)

# VWAP (volume-weighted average probability)
vwap = (trades.yes_price * trades['count']).sum() / trades['count'].sum()

# Price impact: large trades vs small trades
large = trades[trades['count'] > trades['count'].quantile(0.9)]
small = trades[trades['count'] < trades['count'].quantile(0.5)]
```

---

## Runtime Estimates

| Config | Markets | API Calls | Est. Time |
|--------|---------|-----------|-----------|
| `--skip-candles --skip-ticks` | 10,000 | ~30 | ~15s |
| `--skip-ticks` (candles only) | 10,000 | ~130 | ~2min |
| Full (candles + ticks) | 10,000 | ~10,130 | ~60–90min |
| `--skip-candles -m 2000` | 2,000 | ~2,030 | ~15min |
| `-m 1000 --max-ticks 50` | 1,000 | ~1,030 | ~8min |

The tick phase dominates runtime (1 API call per market). Adjust `--max-ticks` and `--max-markets` to balance coverage vs speed.

---

## File: `decision_features.csv`

One row per candle, enriched with timing, flow, and retrospective labels. This is the best starting point for learning entry rules.

| Column | Type | Description |
|--------|------|-------------|
| `ticker` | str | Market ID |
| `end_period_ts` | int | Candle end timestamp |
| `minutes_to_close` | float | Minutes remaining until market close |
| `minutes_from_open` | float | Minutes elapsed since market open |
| `market_progress` | float | Fraction of market lifetime elapsed |
| `candle_return_1/3/5` | float | Short-horizon momentum features |
| `rolling_volatility_5` | float | 5-candle realized volatility |
| `volume_vs_rolling_mean_5` | float | Volume spike indicator |
| `spread` / `relative_spread` | float | Execution-cost proxy |
| `trade_count` | float | Number of trades in the candle window |
| `trade_contracts` | float | Total traded contracts in the window |
| `yes_trade_contracts` / `no_trade_contracts` | float | Flow by aggressive side |
| `trade_imbalance` | float | `(yes - no) / total`, buy-pressure signal |
| `trade_vwap_yes` | float | Trade-flow VWAP for YES |
| `trade_vwap_edge` | float | VWAP minus candle close price |
| `distance_to_close_prob` | float | Final closing price minus current candle price |
| `label_kind` | str | Whether this market has a clean binary outcome or a scalar settlement |
| `label_usable` | bool | Filter on this before supervised training |
| `result_binary` | int | Final market outcome |
| `realized_edge_yes_entry` | float | Ex-post return if you bought YES at this candle close |
| `realized_edge_no_entry` | float | Ex-post return if you bought NO at this candle close |

### How to use it

- Use momentum, volatility, spread, and trade imbalance as model inputs.
- Use `minutes_to_close` and `market_progress` to separate pregame vs near-resolution behavior.
- Filter to `label_usable == True` when training a YES/NO classifier or edge model.
- Treat `realized_edge_yes_entry` and `realized_edge_no_entry` as labels for backtesting only, not live inputs.
- If you cap `--max-ticks`, trade-flow columns become partial snapshots rather than full-flow measurements.
