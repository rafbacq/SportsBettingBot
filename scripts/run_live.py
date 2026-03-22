#!/usr/bin/env python3
"""Run the live trading system.

Usage:
    python scripts/run_live.py --config config/settings.yaml
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logging_config import load_config, setup_logging
from src.live.runner import LiveTradingRunner


def main():
    parser = argparse.ArgumentParser(description="Run live trading")
    parser.add_argument("--config", default="config/settings.yaml", help="Config file")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config)

    dry_run = config.get("trading", {}).get("dry_run", True)
    print("\n" + "=" * 50)
    print("  AI DUAL-REGIME REBOUND TRADING SYSTEM")
    print(f"  Mode: {'DRY RUN' if dry_run else '*** LIVE TRADING ***'}")
    print("=" * 50 + "\n")

    if not dry_run:
        confirm = input("⚠️  LIVE TRADING MODE. Type 'YES' to confirm: ")
        if confirm != "YES":
            print("Aborted.")
            return

    runner = LiveTradingRunner(config)
    runner.start()


if __name__ == "__main__":
    main()
