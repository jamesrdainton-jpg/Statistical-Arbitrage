import argparse

from pairs.config import load_config, universe_tickers
from pairs.data import clean_prices, get_prices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="re-download even if the cache already covers the dates",
    )
    args = parser.parse_args()

    config = load_config()
    data_cfg = config["data"]
    tickers = universe_tickers(config)

    print(f"fetching {len(tickers)} tickers: {data_cfg['start']} to {data_cfg['end']}")
    prices = get_prices(
        tickers,
        data_cfg["start"],
        data_cfg["end"],
        data_cfg["cache"],
        refresh=args.refresh,
    )

    print("cleaning")
    cleaned = clean_prices(prices, data_cfg["max_missing_frac"])

    print(
        f"\n{len(cleaned)} rows, {len(cleaned.columns)} tickers kept"
        f"\n{cleaned.index.min().date()} to {cleaned.index.max().date()}"
        f"\ncached at {data_cfg['cache']}"
    )
    print("kept: " + ", ".join(cleaned.columns))

    bench_cfg = config["benchmark"]
    print(f"\nfetching benchmark {bench_cfg['index']}")
    bench = get_prices(
        [bench_cfg["index"]],
        data_cfg["start"],
        data_cfg["end"],
        bench_cfg["cache"],
        refresh=args.refresh,
    )
    print(
        f"{len(bench)} rows, {bench.index.min().date()} to {bench.index.max().date()}"
        f"\ncached at {bench_cfg['cache']}"
    )


if __name__ == "__main__":
    main()
