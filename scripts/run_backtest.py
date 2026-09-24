import matplotlib.pyplot as plt
import pandas as pd

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"
ORANGE = "#eb6834"

from pairs.config import load_config, resolve_path, ticker_sectors
from pairs.data import clean_prices, load_prices, to_log_prices, walk_forward_windows
from pairs.kalman import spread_zscore, kalman_hedge_ratio
from pairs.signals import generate_positions
from pairs.backtest import run_backtest
from pairs.selection import find_pairs
from pairs.metrics import bootstrap_sharpe, buy_and_hold, max_drawdown, sharpe_ratio, trade_stats


config = load_config()
prices = load_prices(config["data"]["cache"])
prices = clean_prices(prices, config["data"]["max_missing_frac"])
log_prices = to_log_prices(prices)

returns = prices.pct_change()

cost_bps = config["costs"]["commission_bps"] + config["costs"]["slippage_bps"]
sectors = ticker_sectors(config)


def run_pair(a, b, train, test):
    full = pd.concat([train, test])
    betas, alphas, errors, variances = kalman_hedge_ratio(
        full[a].values, full[b].values, **config["kalman"]
    )

    z = spread_zscore(errors)
    z = pd.Series(z.values, index=full.index).loc[test.index]
    betas = pd.Series(betas, index=full.index).loc[test.index]

    positions = pd.Series(
        generate_positions(z.values, **config["signals"]), index=test.index
    )
    net, trades = run_backtest(
        returns[a].loc[test.index],
        returns[b].loc[test.index],
        positions,
        betas,
        cost_bps,
    )
    return net, trades

# Claude's code for the formatting 
def plot_equity(oos, bench, bench_name, path):
    span = bench.loc[oos.index.min():oos.index.max()]
    strategy = (1 + oos.reindex(span.index).fillna(0)).cumprod()
    bench = span / span.iloc[0]

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.axhline(1.0, color=GRID, linewidth=1, zorder=1)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    series = [("Pairs strategy", strategy, BLUE), (bench_name, bench, ORANGE)]
    for label, s, colour in series:
        ax.plot(s.index, s.values, color=colour, linewidth=2, label=label, zorder=3)
        ax.annotate(
            f" {label}  {s.iloc[-1] - 1:+.0%}",
            xy=(s.index[-1], s.iloc[-1]),
            color=INK,
            fontsize=9,
            va="center",
        )

    ax.set_title("Growth of £1, out of sample", color=INK, fontsize=13, loc="left", pad=14)
    ax.set_ylabel("Cumulative growth", color=INK_MUTED, fontsize=10)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=INK)
    ax.margins(x=0.12)

    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    print(f"\nchart written to {path}")


window_returns = []
window_trades = []

for train, test in walk_forward_windows(log_prices, **config["walkforward"]):
    pairs = find_pairs(train, sectors=sectors, **config["selection"])
    results = [run_pair(row["a"], row["b"], train, test) for _, row in pairs.iterrows()]
    if not results:
        continue

    nets = [net for net, _ in results]
    window_returns.append(pd.concat(nets, axis=1).mean(axis=1))
    window_trades.extend(trades for _, trades in results)

    print(
        f"{test.index.min().date()}..{test.index.max().date()}  "
        f"{len(pairs)} pairs: " + ", ".join(f"{r['a']}/{r['b']}" for _, r in pairs.iterrows())
    )

oos = pd.concat(window_returns).sort_index().dropna()
trades = pd.concat(window_trades, ignore_index=True)

print(f"\nout-of-sample: {oos.index.min().date()} to {oos.index.max().date()} ({len(oos)} days)")
print(f"sharpe        {sharpe_ratio(oos):.2f}")
print(f"max drawdown  {max_drawdown(oos):.1%}")
print(f"total return  {(1 + oos).prod() - 1:.1%}")

stats = trade_stats(trades)
print(f"trades        {stats['n_trades']}")
print(f"win rate      {stats['win_rate']:.1%}")
print(f"avg holding   {stats['avg_holding']:.1f} days")

ci = bootstrap_sharpe(oos)
print(f"sharpe 95% CI [{ci['lower']:.2f}, {ci['upper']:.2f}]")

bench_cfg = config["benchmark"]
if resolve_path(bench_cfg["cache"]).exists():
    bench = load_prices(bench_cfg["cache"])[bench_cfg["index"]]
    bh = buy_and_hold(bench, oos.index.min(), oos.index.max())
    print(
        f"\n{bench_cfg['index']} buy and hold:"
        f"  return {bh['total_return']:.1%}"
        f"  sharpe {bh['sharpe']:.2f}"
        f"  max dd {bh['max_drawdown']:.1%}"
    )
    plot_equity(oos, bench, bench_cfg["index"], resolve_path("reports/equity_curve.png"))
else:
    print(f"\n{bench_cfg['index']} not cached; run fetch_data.py for the benchmark")
plt.show()