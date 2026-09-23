import numpy as np
import pandas as pd
import yfinance as yf

from pairs.config import resolve_path

_COVERAGE_TOLERANCE = pd.Timedelta(days=7)


def download_prices(tickers, start, end, path=None):
    tickers = list(tickers)
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance returned no data for {tickers}")

    # With several tickers the columns are a (field, ticker) MultiIndex; with
    # one ticker older versions return plain field columns instead.
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices.dropna(axis=1, how="all").sort_index()
    prices.index.name = "Date"

    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        print(f"  warning: no data returned for {', '.join(missing)}")

    # Keep the column order given in the config rather than Yahoo's.
    prices = prices[[t for t in tickers if t in prices.columns]]

    if path is not None:
        path = resolve_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(path)

    return prices


def load_prices(path):
    return pd.read_csv(resolve_path(path), index_col=0, parse_dates=True)


def get_prices(tickers, start, end, path, refresh=False):

    tickers = list(tickers)
    cache = resolve_path(path)

    if not refresh and cache.exists():
        cached = load_prices(cache)
        if _covers(cached, tickers, start, end):
            # Trim the cache to the configured window so the CSV on disk always
            # matches config.yaml rather than keeping a stale wider range.
            trimmed = cached.loc[str(start):str(end), tickers]
            trimmed.to_csv(cache)
            return trimmed
        print("  cache does not cover the request; re-downloading")

    prices = download_prices(tickers, start, end, path=cache)
    return prices.loc[str(start):str(end)]


def _covers(cached, tickers, start, end):

    if cached.empty or not set(tickers).issubset(cached.columns):
        return False

    wanted_start = pd.Timestamp(start)
    # An end date in the future can never be covered, so cap it at today.
    wanted_end = min(pd.Timestamp(end), pd.Timestamp.today().normalize())
    return (
        cached.index.min() <= wanted_start + _COVERAGE_TOLERANCE
        and cached.index.max() >= wanted_end - _COVERAGE_TOLERANCE
    )


def clean_prices(prices, max_missing_frac=0.05):

    missing_frac = prices.isna().mean()
    dropped = missing_frac[missing_frac > max_missing_frac]
    if len(dropped):
        print(
            "  dropping "
            + ", ".join(f"{t} ({f:.1%} missing)" for t, f in dropped.items())
        )

    prices = prices[missing_frac[missing_frac <= max_missing_frac].index]
    return prices.ffill().dropna()


def to_log_prices(prices):
    return np.log(prices)

def split_data(prices, split_date):
    split_date = pd.Timestamp(split_date)
    train_data = prices.loc[prices.index < split_date]
    test_data = prices.loc[prices.index >= split_date]
    return train_data, test_data


def walk_forward_windows(prices, train_days=252, test_days=126):
    """
    Yield (train, test) DataFrame pairs for walk-forward validation.

    split_data() gives ONE train/test cut. This gives a series of them, sliding
    through history so every window trades a period it was not selected on.

    The geometry, with train_days=504 (2y) and test_days=126 (6mo):

        window 0:  train [0     : 504]   test [504 : 630]
        window 1:  train [126   : 630]   test [630 : 756]
        window 2:  train [252   : 756]   test [756 : 882]

    Two things to notice:
      - Training blocks OVERLAP (each slides forward by test_days, not by its
        own length). That is intended - you retrain on the most recent 2 years
        every 6 months, exactly as you would live.
      - Test blocks sit end to end and never overlap, so concatenating them
        gives one continuous out-of-sample return series for Steps 8-10.

    This is a rolling window (training length stays fixed). The alternative is
    an expanding window, where training always starts at index 0 and grows.
    Rolling adapts faster when relationships break down; expanding uses more
    data. Rolling matches the plan's sketch, so start there.
    """
    n = len(prices)

    for end in range(train_days, n - test_days + 1, test_days):
        yield prices.iloc[end - train_days:end], prices.iloc[end:end + test_days]
