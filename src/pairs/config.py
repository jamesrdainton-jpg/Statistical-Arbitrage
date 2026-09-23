"""Loading config.yaml and the helpers that read the universe out of it."""

from pathlib import Path

import yaml

# .../src/pairs/config.py -> .../src/pairs -> .../src -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(path=None):
    """Read config.yaml (or another YAML file) into a dict."""
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path):
    """Turn a config path such as 'data/prices.csv' into an absolute path."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def universe_tickers(config):
    """Flatten the sector -> tickers mapping into one de-duplicated list."""
    tickers = []
    for sector_tickers in config["data"]["universe"].values():
        for ticker in sector_tickers:
            if ticker not in tickers:
                tickers.append(ticker)
    return tickers


def ticker_sectors(config):
    """Invert the universe into a ticker -> sector mapping."""
    return {
        ticker: sector
        for sector, sector_tickers in config["data"]["universe"].items()
        for ticker in sector_tickers
    }
