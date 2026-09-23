import numpy as np
import pandas as pd


def sharpe_ratio(returns):
    return returns.mean()/returns.std() * np.sqrt(252)


def max_drawdown(returns):
    equity = (1 + returns).cumprod()
    return (equity / equity.cummax() - 1).min()


def annualised_turnover(w_a, w_b):
    return np.mean(np.abs(w_a.diff()) + np.abs(w_b.diff())) * (252)



def trade_stats(trades):
    n_trades = len(trades)
    win_rate = (trades["pnl"] > 0).mean()
    avg_holding = trades["days_held"].mean()
    return {"n_trades": n_trades, "win_rate": win_rate, "avg_holding": avg_holding}


def buy_and_hold(prices, start=None, end=None):
    returns = prices.loc[start:end].pct_change().dropna()
    return {
        "total_return": (1 + returns).prod() - 1,
        "sharpe": sharpe_ratio(returns),
        "max_drawdown": max_drawdown(returns),
    }


def bootstrap_sharpe(returns, n_boot=5000, block_size=20, seed=42):

    rng = np.random.default_rng(seed)
    r = returns.dropna().values
    n = len(r)

    n_blocks = int(np.ceil(n / block_size))
    sharpes = []
    for _ in range(n_boot):
        starts = rng.integers(0, n-block_size+1, n_blocks)
        sample = np.concatenate([r[s:(s + block_size)] for s in starts])[:n]
        sharpes.append(sample.mean() / sample.std() * np.sqrt(252))

    lower, upper = np.percentile(sharpes, [2.5, 97.5])
    return {"sharpe": sharpe_ratio(returns), "lower": lower, "upper": upper}
