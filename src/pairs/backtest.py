import numpy as np
import pandas as pd


def run_backtest(returns_a, returns_b, positions, betas, cost_bps):
    
    entry_day = (positions != 0) & (positions != positions.shift(1))
    # fillna(0) covers the flat stretch before the first trade: without it the
    # weights are NaN there and those days drop out of the return series.
    beta_locked = betas.where(entry_day).ffill().fillna(0)

    w_a = positions / ( 1 + np.abs(beta_locked))
    w_b = -positions * beta_locked / ( 1 + np.abs(beta_locked) )

    w_a_held = w_a.shift(1)
    w_b_held = w_b.shift(1) # Avoid Look ahead Bias

    gross = w_a_held * returns_a + w_b_held * returns_b


    turnover = w_a.diff().abs() + w_b.diff().abs()
    costs = turnover * cost_bps / 10_000

    net = gross - costs

    trade_id = (positions != positions.shift(1)).cumsum()

    # Weights are shifted, so a trade's returns lag its positions by a day: the
    # entry day carries only cost, and the last day's gain lands after the
    # position is already flat. Extend each trade by that settlement day,
    # otherwise every trade is booked missing a day of return.
    held = positions.shift(1)
    group = trade_id.where(positions != 0).fillna(
        trade_id.shift(1).where((positions == 0) & (held != 0))
    )

    trade_data = pd.DataFrame({
        "position": positions,
        "beta_locked": beta_locked,
        "net": net,
        "group": group,
    }).dropna(subset=["group"])

    trades = trade_data.groupby("group").apply(_summarise_trade).reset_index(drop=True)

    return net, trades


def _summarise_trade(g):
    on = g[g["position"] != 0]
    return pd.Series({
        "entry_date": on.index[0],
        "exit_date": on.index[-1],
        "days_held": len(on),
        "direction": on["position"].iloc[0],
        "beta": on["beta_locked"].iloc[0],
        "pnl": (1 + g["net"]).prod() - 1,
    })
