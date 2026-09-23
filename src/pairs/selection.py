from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

_MIN_OBSERVATIONS = 250


def half_life(spread):

    spread = np.asarray(spread, dtype=float)
    lagged = spread[:-1]
    delta = np.diff(spread)

    B, _ = np.polyfit(lagged, delta, 1)
    if B >= 0:
        return np.inf
    return -np.log(2) / B


def test_pair(log_a, log_b):
    both = pd.concat([log_a, log_b], axis=1).dropna()
    if len(both) < _MIN_OBSERVATIONS:
        return None

    a = both.iloc[:, 0].to_numpy()
    b = both.iloc[:, 1].to_numpy()
    beta, alpha = np.polyfit(b, a, 1)
    spread = a - (beta * b + alpha)
    _, pvalue, _ = coint(a, b)

    return {
        "beta": beta,
        "alpha": alpha,
        "pvalue": pvalue,
        "half_life": half_life(spread),
        "n_obs": len(both),
    }


def find_pairs(
    log_prices,
    sectors=None,
    pvalue_threshold=0.05,
    min_half_life=5,
    max_half_life=60,
    max_pairs=5,
):

    tickers = list(log_prices.columns)

    rows = []
    for a, b in combinations(tickers, 2):
        result = test_pair(log_prices[a], log_prices[b])
        if result is None:
            continue
        same_sector = (
            sectors is not None
            and a in sectors
            and sectors.get(a) == sectors.get(b)
        )
        rows.append(
            {
                "a": a,
                "b": b,
                "sector": sectors.get(a) if same_sector else "",
                "same_sector": same_sector,
                **result,
            }
        )

    n_tested = len(rows)
    if not n_tested:
        raise RuntimeError("no pair had enough overlapping history to test")

    pairs = pd.DataFrame(rows)

    n_passing = int((pairs["pvalue"] < pvalue_threshold).sum())
    expected_by_chance = pvalue_threshold * n_tested
    print(
        f"  tested {n_tested} pairs; {n_passing} below p={pvalue_threshold} "
        f"({expected_by_chance:.0f} expected by chance alone)"
    )

    keep = (
        (pairs["pvalue"] < pvalue_threshold)
        & (pairs["half_life"] >= min_half_life)
        & (pairs["half_life"] <= max_half_life)
    )
    pairs = pairs[keep]
    pairs = pairs.sort_values(
        ["same_sector", "pvalue"], ascending=[False, True]
    ).reset_index(drop=True)

    if max_pairs is not None:
        pairs = pairs.head(max_pairs)

    print(f"  keeping {len(pairs)} pairs after half-life and ranking filters")
    return pairs
