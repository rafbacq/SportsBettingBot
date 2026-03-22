"""Logging configuration for the trading system."""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Union

import yaml


def load_config(config_path: str = "config/settings.yaml") -> dict:
    """Load and return the YAML configuration, resolving env vars."""
    with open(config_path, "r") as f:
        raw = f.read()

    # Resolve ${ENV_VAR} placeholders
    for key, value in os.environ.items():
        raw = raw.replace(f"${{{key}}}", value)

    return yaml.safe_load(raw)


def setup_logging(config: Union[dict, None] = None) -> logging.Logger:
    """Set up rotating file + console logging. Returns root logger."""
    if config is None:
        config = load_config()

    log_cfg = config.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    log_file = log_cfg.get("log_file", "data/trading.log")
    max_bytes = log_cfg.get("max_bytes", 10_485_760)
    backup_count = log_cfg.get("backup_count", 5)

    # Ensure log directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Root logger
    root = logging.getLogger("trading")
    root.setLevel(level)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(fmt)
    root.addHandler(console)

    # File handler
    file_handler = RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    return root
