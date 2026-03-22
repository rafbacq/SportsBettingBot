import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.kalshi_client import KalshiClient
from src.live.runner import LiveTradingRunner
from src.utils.logging_config import load_config, setup_logging

logger = logging.getLogger("trading.api")

app = FastAPI(title="Kalshi AI Trading API")

# Load config
config = load_config("config/settings.yaml")
setup_logging(config)

# Enable CORS for the local React app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global State ─────────────────────────────────────────────────────────────

# Load models
MODELS_DIR = "src/ml/models"
models = {}

def load_models():
    global models
    for name in ["regime_classifier", "rebound_classifier", "multiplier_regressor", "ev_estimator"]:
        path = os.path.join(MODELS_DIR, f"{name}.joblib")
        if os.path.exists(path):
            models[name] = joblib.load(path)
        else:
            logger.warning(f"Model not found: {path}")

# Initialize Kalshi client
kalshi_client = KalshiClient(config)

# Bot state
class BotState:
    is_running: bool = False
    spend_limit: float = 10.00
    runner: Optional[LiveTradingRunner] = None
    runner_task: Optional[asyncio.Task] = None
    recent_logs: List[dict] = []

bot_state = BotState()

# ─── API Models ───────────────────────────────────────────────────────────────

class BotConfig(BaseModel):
    spend_limit: float

class CandleData(BaseModel):
    """Candlestick data sent from the frontend."""
    ticker: str
    candles: List[dict]  # [{price: {close_dollars: ...}, ...}, ...]

class TradeRecommendation(BaseModel):
    ticker: str
    action: str  # "BUY YES", "BUY NO", "NO TRADE"
    regime: str
    rebound_prob: float
    expected_ev: float
    target_exit: float
    stop_loss: float
    reasons: List[str]

# ─── Feature Computation ──────────────────────────────────────────────────────

def extract_prob_from_candle(c: dict) -> float:
    """Extract probability from a Kalshi candlestick dict, trying multiple field paths."""
    price = c.get("price", {})
    # Try close_dollars first (v2 default), then close_cents
    if isinstance(price, dict):
        if "close_dollars" in price:
            v = float(price["close_dollars"])
            if v > 0:
                return v
        if "close_cents" in price:
            v = float(price["close_cents"]) / 100.0
            if v > 0:
                return v
    # Some candlestick shapes have price directly
    if "close" in c:
        return float(c["close"])
    return 0.0

def candles_to_probs(candles: List[dict]) -> List[float]:
    """Convert a list of candlestick dicts to a list of probabilities."""
    probs = []
    for c in candles:
        p = extract_prob_from_candle(c)
        if p > 0:
            probs.append(p)
    return probs

def compute_features(prob_trajectory: List[float], initial_prob: float, market_progress: float, is_favorite_side: bool) -> dict:
    if not prob_trajectory:
        return {}
    probs = np.array(prob_trajectory)
    current = probs[-1]

    # Core features
    implied_prob = current
    prob_weak = current if not is_favorite_side else 1 - current
    prob_strong = current if is_favorite_side else 1 - current
    initial_weak = initial_prob if not is_favorite_side else 1 - initial_prob
    initial_strong = initial_prob if is_favorite_side else 1 - initial_prob

    op_value = initial_weak / max(prob_weak, 0.001)
    s_value = initial_strong / max(prob_strong, 0.001)

    prob_drop = initial_prob - current
    prob_drop_pct = prob_drop / max(initial_prob, 0.001)

    strength_ratio = initial_prob / max(1 - initial_prob, 0.001)

    # Momentum features
    candle_return_1 = probs[-1] - probs[-2] if len(probs) >= 2 else 0
    candle_return_3 = probs[-1] - probs[-4] if len(probs) >= 4 else candle_return_1
    candle_return_5 = probs[-1] - probs[-6] if len(probs) >= 6 else candle_return_3

    # Volatility
    if len(probs) >= 5:
        returns = np.diff(probs[-5:])
        rolling_vol = float(np.std(returns)) if len(returns) > 0 else 0
    else:
        rolling_vol = 0

    return {
        "implied_prob": implied_prob,
        "initial_prob": initial_prob,
        "market_progress": market_progress,
        "candle_return_1": candle_return_1,
        "candle_return_3": candle_return_3,
        "candle_return_5": candle_return_5,
        "rolling_volatility_5": rolling_vol,
        "volume": 0,
        "volume_vs_rolling_mean_5": 0,
        "spread": 0.02,
        "relative_spread": 0.02,
        "trade_imbalance": 0,
        "trade_vwap_edge": 0,
        "op_value": op_value,
        "s_value": s_value,
        "strength_ratio": strength_ratio,
        "prob_weak": prob_weak,
        "prob_strong": prob_strong,
        "prob_drop": prob_drop,
        "prob_drop_pct": prob_drop_pct,
    }

def predict(features_dict: dict) -> dict:
    if "regime_classifier" not in models:
        load_models()
    
    if "regime_classifier" not in models:
        raise ValueError("Models not loaded")

    feature_names = models["regime_classifier"]["feature_names"]
    X = np.array([[features_dict.get(f, 0.0) for f in feature_names]])

    regime_clf = models["regime_classifier"]["model"]
    rebound_clf = models["rebound_classifier"]["model"]
    mult_reg = models["multiplier_regressor"]["model"]
    ev_reg = models["ev_estimator"]["model"]

    regime_pred = regime_clf.predict(X)[0]
    regime = "cross" if regime_pred == 1 else "non_cross"

    rebound_proba = float(rebound_clf.predict_proba(X)[0][1])
    mult_pred = float(mult_reg.predict(X)[0])
    ev_pred = float(ev_reg.predict(X)[0])

    return {
        "regime": regime,
        "rebound_prob": rebound_proba,
        "predicted_multiplier": mult_pred,
        "predicted_ev": ev_pred,
    }

def _analyze_probs(ticker: str, probs: List[float]) -> TradeRecommendation:
    """Run the ML model on a probability trajectory and return a recommendation."""
    # Pad short trajectories
    while len(probs) < 5:
        probs.insert(0, probs[0] if probs else 0.5)
    
    initial_prob = probs[0]
    current_prob = probs[-1]
    
    features = compute_features(probs, initial_prob, market_progress=0.5, is_favorite_side=(initial_prob >= 0.5))
    preds = predict(features)
    
    is_buy = preds["predicted_ev"] > 0.5 and preds["rebound_prob"] > 0.45
    
    if is_buy:
        action = "BUY YES"
        target = min(current_prob * preds["predicted_multiplier"], 0.95)
        stop = max(current_prob * 0.5, 0.01)
        reasons = ["EV is highly positive", f"Rebound probability is {preds['rebound_prob']:.0%}"]
    else:
        action = "NO TRADE"
        target = 0.0
        stop = 0.0
        reasons = []
        if preds["predicted_ev"] <= 0.5:
            reasons.append(f"EV too low ({preds['predicted_ev']:+.2f}, need > +0.50)")
        if preds["rebound_prob"] <= 0.45:
            reasons.append(f"Rebound prob too low ({preds['rebound_prob']:.0%})")
            
    return TradeRecommendation(
        ticker=ticker,
        action=action,
        regime=preds["regime"].upper(),
        rebound_prob=preds["rebound_prob"],
        expected_ev=preds["predicted_ev"],
        target_exit=target,
        stop_loss=stop,
        reasons=reasons
    )


# ─── API Endpoints ────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up API server and loading models...")
    load_models()

# ─── Recommendations: POST with candle data from the frontend ─────────────
# The frontend already successfully fetches candlesticks through its proxy.
# Instead of duplicating that fetch in Python (which fails without auth),
# the frontend sends its candle data here for ML analysis.

@app.post("/api/recommendations")
async def post_recommendations(data: CandleData):
    """Analyze candlestick data sent from the frontend."""
    try:
        probs = candles_to_probs(data.candles)
        if not probs:
            # If no valid prices, return a neutral recommendation
            return TradeRecommendation(
                ticker=data.ticker,
                action="NO TRADE",
                regime="NONE",
                rebound_prob=0.0,
                expected_ev=0.0,
                target_exit=0.0,
                stop_loss=0.0,
                reasons=["No valid candlestick price data available"]
            )
        return _analyze_probs(data.ticker, probs)
    except Exception as e:
        logger.error(f"Error in POST recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Keep the GET endpoint as a fallback (tries to fetch candles from Python client)
@app.get("/api/recommendations")
async def get_recommendations(ticker: str):
    """Fallback: try to fetch candles directly from Kalshi API."""
    try:
        end_ts = int(datetime.now(timezone.utc).timestamp())
        start_ts = end_ts - 7 * 24 * 3600  # Last 7 days for maximum coverage
        
        candle_data = kalshi_client.get_market_candlesticks(
            ticker=ticker,
            period_interval=60,
            start_ts=start_ts,
            end_ts=end_ts,
        )
        candles = candle_data.get("candlesticks", [])
        probs = candles_to_probs(candles) if candles else []
        
        if not probs:
            return TradeRecommendation(
                ticker=ticker,
                action="NO TRADE",
                regime="NONE",
                rebound_prob=0.0,
                expected_ev=0.0,
                target_exit=0.0,
                stop_loss=0.0,
                reasons=["Could not fetch candlestick data — try using POST endpoint"]
            )
        return _analyze_probs(ticker, probs)
        
    except Exception as e:
        logger.error(f"Error in GET recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ─── Bot Endpoints ───────────────────────────────────────────────────────────

@app.post("/api/bot/start")
async def start_bot():
    if bot_state.is_running:
        return {"status": "Already running"}
        
    bot_state.is_running = True
    bot_state.recent_logs.append({"time": datetime.now().isoformat(), "msg": "Starting autonomous trading bot..."})
    
    async def bot_loop():
        while bot_state.is_running:
            try:
                # Fetch events WITH nested markets — this way market prices
                # come embedded in the event response (same as the frontend)
                events_data = kalshi_client._get("/events", {
                    "status": "open",
                    "limit": 10,
                    "with_nested_markets": "true",
                })
                events = events_data.get("events", [])
                
                if not events:
                    bot_state.recent_logs.append({"time": datetime.now().isoformat(), "msg": "⚠️ No events returned from API — check API key in config/settings.yaml"})
                    if len(bot_state.recent_logs) > 50: bot_state.recent_logs.pop(0)
                    await asyncio.sleep(10)
                    continue
                
                for event in events:
                    if not bot_state.is_running: break
                    
                    event_title = event.get("title", "")
                    nested_markets = event.get("markets", [])
                    
                    if not nested_markets:
                        continue
                    
                    for market in nested_markets[:5]:  # Limit to 5 markets per event
                        if not bot_state.is_running: break
                        
                        mticker = market.get("ticker", "")
                        mtitle = market.get("yes_sub_title") or market.get("title") or mticker
                        
                        # Get price from nested market data
                        # The events-with-nested-markets response includes these fields:
                        price_val = 0.0
                        for field in ["last_price", "yes_bid", "last_price_dollars", "yes_bid_dollars"]:
                            raw = market.get(field)
                            if raw:
                                try:
                                    v = float(raw)
                                    if v > 0:
                                        price_val = v if v <= 1.0 else v / 100.0
                                        break
                                except (ValueError, TypeError):
                                    continue
                        
                        prob_pct = round(price_val * 100)
                        bot_state.recent_logs.append({
                            "time": datetime.now().isoformat(), 
                            "msg": f"Scanning: {mtitle} ({prob_pct}%) [{mticker[:30]}]"
                        })
                        if len(bot_state.recent_logs) > 50: bot_state.recent_logs.pop(0)
                        
                        # Only analyze markets in interesting probability ranges
                        if 0.03 < price_val < 0.97:
                            # Fetch candlesticks for ML analysis
                            end_ts = int(datetime.now(timezone.utc).timestamp())
                            start_ts = end_ts - 7 * 24 * 3600  # 7 days
                            candle_data = kalshi_client.get_market_candlesticks(
                                ticker=mticker,
                                period_interval=60,
                                start_ts=start_ts,
                                end_ts=end_ts,
                            )
                            candles = candle_data.get("candlesticks", [])
                            probs = candles_to_probs(candles) if candles else []
                            
                            # If we have candle data, analyze it
                            if probs:
                                while len(probs) < 5:
                                    probs.insert(0, probs[0])
                                
                                init_prob = probs[0]
                                curr_prob = probs[-1]
                                
                                feats = compute_features(probs, init_prob, 0.5, init_prob >= 0.5)
                                if feats:
                                    preds = predict(feats)
                                    ev = preds["predicted_ev"]
                                    
                                    bot_state.recent_logs.append({
                                        "time": datetime.now().isoformat(),
                                        "msg": f"  → Analyzed: EV={ev:+.2f}, Rebound={preds['rebound_prob']:.0%}, Regime={preds['regime'].upper()}"
                                    })
                                    
                                    if ev > 0.5 and preds["rebound_prob"] > 0.45:
                                        msg = f"🚀 BUY SIGNAL: {mtitle} | EV: +{ev:.2f}, Regime: {preds['regime'].upper()}"
                                        bot_state.recent_logs.append({"time": datetime.now().isoformat(), "msg": msg})
                                        exec_msg = f"🟢 SIMULATED BUY YES for {mticker} at {curr_prob:.2f} (Limit: ${bot_state.spend_limit})"
                                        bot_state.recent_logs.append({"time": datetime.now().isoformat(), "msg": exec_msg})
                            else:
                                bot_state.recent_logs.append({
                                    "time": datetime.now().isoformat(),
                                    "msg": f"  → No candle data for {mticker[:30]}, skipping analysis"
                                })
                        
                        await asyncio.sleep(1.0)  # Rate limit between markets
                
                # Cycle complete, wait before next scan
                bot_state.recent_logs.append({"time": datetime.now().isoformat(), "msg": "── Scan cycle complete, waiting 15s ──"})
                if len(bot_state.recent_logs) > 50: bot_state.recent_logs.pop(0)
                await asyncio.sleep(15)
                
            except Exception as e:
                logger.error(f"Bot loop error: {e}")
                bot_state.recent_logs.append({"time": datetime.now().isoformat(), "msg": f"❌ Error: {str(e)[:80]}"})
                await asyncio.sleep(5)
            
    bot_state.runner_task = asyncio.create_task(bot_loop())
    return {"status": "Started"}

@app.post("/api/bot/stop")
async def stop_bot():
    if not bot_state.is_running:
        return {"status": "Not running"}
        
    bot_state.is_running = False
    if bot_state.runner_task:
        bot_state.runner_task.cancel()
    
    bot_state.recent_logs.append({"time": datetime.now().isoformat(), "msg": "Bot stopped."})
    return {"status": "Stopped"}

@app.get("/api/bot/status")
async def get_bot_status():
    return {
        "is_running": bot_state.is_running,
        "spend_limit": bot_state.spend_limit,
        "logs": bot_state.recent_logs[-20:]  # Return last 20 logs
    }

@app.post("/api/bot/config")
async def config_bot(cfg: BotConfig):
    bot_state.spend_limit = cfg.spend_limit
    bot_state.recent_logs.append({"time": datetime.now().isoformat(), "msg": f"Spend limit updated to ${bot_state.spend_limit}"})
    return {"status": "Config updated", "spend_limit": bot_state.spend_limit}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
