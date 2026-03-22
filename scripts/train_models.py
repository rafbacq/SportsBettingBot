#!/usr/bin/env python3
"""Train all ML models for the trading system.

Usage:
    python scripts/train_models.py --data-dir data/ --output-dir src/ml/models/
    python scripts/train_models.py --synthetic --max-samples 1000
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logging_config import load_config, setup_logging
from src.ml.trainer import ModelTrainer


def main():
    parser = argparse.ArgumentParser(description="Train ML models")
    parser.add_argument("--data-dir", default="data", help="Directory with CSV data")
    parser.add_argument("--output-dir", default=None, help="Model output directory")
    parser.add_argument("--config", default="config/settings.yaml", help="Config file")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--max-samples", type=int, default=None, help="Max games to use")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.output_dir:
        config.setdefault("ml", {})["models_dir"] = args.output_dir

    setup_logging(config)

    trainer = ModelTrainer(config)
    results = trainer.train_all(
        data_dir=args.data_dir,
        use_synthetic=args.synthetic,
        max_samples=args.max_samples,
    )

    print("\n" + "=" * 50)
    print("TRAINING RESULTS")
    print("=" * 50)
    for section, metrics in results.items():
        if isinstance(metrics, dict):
            print(f"\n{section}:")
            for k, v in metrics.items():
                print(f"  {k}: {v}")
        else:
            print(f"{section}: {metrics}")


if __name__ == "__main__":
    main()
