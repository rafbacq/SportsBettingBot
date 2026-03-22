# AI Dual-Regime Rebound Trading System

A machine-learning-optimized sports betting bot for Kalshi prediction markets that identifies and exploits temporary probability mispricings using two complementary strategies.

## How It Works

The system decomposes live sports probability movements into two regimes:

| | **Non-Cross Model** | **Cross Model** |
|---|---|---|
| **Target** | Weak team partial rebound | Strong team full recovery |
| **Entry** | Prob drops to 1–5% | Favorite collapses to 3–20% |
| **Exit** | Below 50% (×4–12 multiplier) | Can hold through 50% (×5–20) |
| **Key Metric** | OP = P₀/Pₜ | S = P₀/Pₜ |
| **Hit Rate** | Low | Moderate |
| **Payoff** | High per win | Very high |

All entry thresholds, exit multipliers, and time windows are **learned by ML models** from historical data — not hardcoded.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train models on synthetic data (for testing)
python scripts/train_models.py --synthetic --max-samples 1000

# Run backtest
python scripts/backtest.py --synthetic --max-samples 500

# Run live (dry-run mode by default)
python scripts/run_live.py

# Run tests
pytest tests/ -v
```

## Project Structure

```
SportsBettingBot/
├── config/settings.yaml          # All configuration
├── src/
│   ├── data/                     # Kalshi API, CSV loader, models, storage
│   ├── features/engine.py        # OP/S values, momentum, volatility
│   ├── ml/                       # Regime classifier, param optimizers, trainer
│   ├── strategy/                 # Non-Cross, Cross, regime router
│   ├── execution/                # Portfolio, risk management, orders
│   └── live/runner.py            # Real-time trading loop
├── scripts/                      # CLI tools (train, backtest, live)
└── tests/                        # Unit tests
```

## Configuration

Edit `config/settings.yaml` to set:
- **Kalshi API credentials** (or set `KALSHI_API_KEY` / `KALSHI_PRIVATE_KEY_PATH` env vars)
- **Trading limits** (max positions, daily loss limit, stake sizes)
- **Sports to track** (NCAAB, ATP)
- **dry_run: true** (default) — must explicitly set to `false` for live trading

## Training with Real Data

Place CSV files in `data/` with columns:
```
game_id, timestamp, team_a, team_b, prob_a, prob_b, time_remaining, sport
```

Then train:
```bash
python scripts/train_models.py --data-dir data/
```

The ML system learns optimal parameters for entry zones, OP/S thresholds, exit multipliers, and time constraints from your historical data.

## Safety

- **Dry-run mode** is enabled by default
- **Circuit breaker** halts trading if daily loss limit is hit
- **Position limits** prevent over-exposure
- Live trading requires typing `YES` at the confirmation prompt
