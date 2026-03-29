import pandas as pd
import os
from datetime import datetime

def main():
    if not os.path.exists("kalshi_data") or not os.path.exists("kalshi_data/candlesticks.csv") or not os.path.exists("kalshi_data/markets.csv"):
        print("Required kalshi_data files missing.")
        return

    print("Loading Kalshi data...")
    markets = pd.read_csv("kalshi_data/markets.csv")
    candles = pd.read_csv("kalshi_data/candlesticks.csv")

    # We need: game_id, timestamp, team_a, team_b, prob_a, prob_b, time_remaining, sport

    cols = [c for c in ["ticker", "league", "_league", "yes_sub_title", "no_sub_title", "expiration_time"] if c in markets.columns]
    markets = markets[cols]
    
    print("Merging data...")
    df = pd.merge(candles, markets, on="ticker", how="inner")
    
    print("Formatting...")
    # Convert expiration_time (ISO string) to timestamp
    df["exp_ts"] = pd.to_datetime(df["expiration_time"]).apply(lambda x: x.timestamp())
    
    out = pd.DataFrame()
    out["game_id"] = df["ticker"]
    out["timestamp"] = df["end_period_ts"]
    out["team_a"] = df.get("yes_sub_title", pd.Series("Team A", index=df.index)).fillna("Team A")
    out["team_b"] = df.get("no_sub_title", pd.Series("Team B", index=df.index)).fillna("Team B")
    out["prob_a"] = df["price_close"].fillna(0.5)
    out["prob_b"] = 1.0 - out["prob_a"]
    out["time_remaining"] = df["exp_ts"] - df["end_period_ts"]
    out["sport"] = df.get("league", df.get("_league", pd.Series("UNKNOWN", index=df.index))).fillna("UNKNOWN")
    
    os.makedirs("data", exist_ok=True)
    out_path = "data/converted_kalshi.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved {len(out)} rows to {out_path}")

if __name__ == "__main__":
    main()
