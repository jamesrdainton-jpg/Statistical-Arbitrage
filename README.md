# Statistical Arbitrage — Pairs Trading Backtester

A walk-forward backtester for cointegration-based pairs trading on FTSE equities,
with a Kalman-filter hedge ratio, explicit transaction costs, and bootstrapped
confidence intervals on the results.

The strategy trades the spread between two historically cointegrated stocks: when
the spread stretches beyond two standard deviations it shorts the rich leg and buys
the cheap one, closing when the gap narrows. Because both legs are held
simultaneously, the position is roughly market-neutral — it bets on the
*relationship* between two companies, not on market direction.

## Results

Out-of-sample, 2016-01-04 to 2025-12-18 (2,500 trading days), after 10bps
round-trip costs:

| | Strategy | FTSE 100 buy & hold |
|---|---|---|
| Total return | **+67.6%** | +61.5% |
| Sharpe | **0.96** | 0.39 |
| Sharpe 95% CI | [0.42, 1.55] | — |
| Max drawdown | **−8.6%** | −36.6% |
| Trades | 340 | 1 |
| Win rate | 64.7% | — |
| Avg holding period | 11.4 days | — |

![Equity curve]

The headline is risk, not return. The strategy edges the index on total return, but
the interesting number is the drawdown — a quarter of the index's, with a Sharpe
roughly 2.5× higher. The market-neutral construction earns its keep in March 2020,
where the index falls 36% and the strategy is barely affected.

Two caveats belong next to those figures rather than buried below. The Kalman
`delta` was chosen while looking at out-of-sample results, so the Sharpe is
optimistic. And in a typical window 31 of 561 candidate pairs clear p < 0.05 when
**28 would pass by chance alone** — the cointegration signal is barely distinguishable
from noise. Both are expanded in [Limitations](#limitations).

## How it works

Each stage is a function in its own module, so the walk-forward loop can re-run the
whole pipeline dozens of times.

| Module | Role |
|---|---|
| `data.py` | Download, cache and clean prices; build walk-forward windows |
| `selection.py` | Engle-Granger cointegration test, half-life, pair ranking |
| `kalman.py` | Time-varying hedge ratio and spread z-score |
| `signals.py` | z-score → position (+1 / 0 / −1) |
| `backtest.py` | Positions → daily net returns and a per-trade record |
| `metrics.py` | Sharpe, drawdown, turnover, trade stats, block bootstrap |

**Universe.** 34 FTSE constituents across 9 sectors — energy, mining, banks,
insurance, retail, staples, pharma, utilities and telecoms.

**Pair selection.** All 561 possible pairs are tested with
`statsmodels.tsa.stattools.coint`, which runs the two-stage Engle-Granger procedure
and returns a p-value with the correct critical values. Pairs must clear p < 0.05
*and* have a spread half-life between 5 and 60 days — too fast is likely noise, too
slow ties up capital for months. Survivors are ranked with same-sector pairs first,
then by p-value, and the top 5 are traded.

**Hedge ratio.** Rather than one fixed β, a Kalman filter treats
`[β, α]` as a random walk and updates it daily as new prices arrive. Yesterday's
posterior becomes today's prior; because everything is Gaussian the update is closed
form. The filter's one-step-ahead prediction error is the spread, and its rolling
z-score drives the signal.

**Position sizing.** `w_A = pos / (1 + |β|)` and `w_B = −pos·β / (1 + |β|)`, giving
total gross exposure of 1. β is **locked at trade entry** and held until exit —
otherwise the daily-drifting Kalman β would imply rebalancing (and paying costs)
every single day.

## Avoiding look-ahead bias

This is the failure mode that makes backtests look brilliant and be worthless, so
it is handled deliberately at three points:

1. **Weights are lagged.** A signal computed from day *t*'s close could not have been
   traded before that close, so the earliest return it can earn is *t* → *t+1*. The
   weights are shifted one day before being multiplied by returns.
2. **Pairs are chosen on training data only.** `find_pairs` never sees the test
   window. Selecting pairs on the same period you evaluate them guarantees a good
   result, because you picked them *for* behaving well in that period.
3. **The z-score uses only past data.** The rolling mean and standard deviation are
   shifted one day, so today's spread is judged against history that excludes it.

Costs are charged on the day weights actually change, not on the lagged weights.

**Verified empirically.** Multiplying the final day's prices by three and re-running
the entire pipeline leaves all 2,499 earlier daily returns bit-identical (maximum
absolute difference 0.0). If future information leaked backwards, those values would
move.

## Walk-forward validation

A single train/test split gives one result that could be luck. Instead the split
rolls through history: train on 252 days, trade the next 126, then slide forward by
the test length and repeat — 20 windows in total, all of which found tradeable pairs.

```
window 0:  train [0   : 252]   test [252 : 378]
window 1:  train [126 : 378]   test [378 : 504]
window 2:  train [252 : 504]   test [504 : 630]
```

Training blocks overlap, which is intended: you retrain on the most recent year
every six months, exactly as you would live. Test blocks never overlap and tile the
timeline without gaps, so concatenating them yields one continuous out-of-sample
series covering 91% of available history — only the initial 252-day training block
is withheld. Pairs are re-selected in every window, so a pair that stops
cointegrating simply stops being traded.

The Kalman filter is warmed up on each window's training data and its state carried
into the trading period — otherwise the first few weeks of every window would trade
on unconverged β estimates.

## Running it

```bash
python -m venv venv && venv\Scripts\activate
pip install -e ".[dev]"

python scripts/fetch_data.py      # download and cache prices (--refresh to force)
python scripts/run_backtest.py    # walk-forward backtest, stats and chart
```

All parameters live in `config.yaml` — universe, dates, costs, window sizes, entry
and exit thresholds. Nothing is hard-coded in the scripts.

Prices are cached to `data/` so results are reproducible; re-running does not
re-download and therefore does not silently change yesterday's results.

## Limitations

Stated plainly, because most of these would reduce the result and some would erase
it.

**Multiple testing is the most serious problem.** With 34 stocks there are 561
possible pairs. At a 5% threshold roughly 28 pass by chance alone even if nothing
were genuinely cointegrated — and a typical window finds 31. The strategy is
therefore selecting its 5 pairs from a pool that is overwhelmingly noise. Selection
happens on training data only, so this is not look-ahead; it is the data-mining
regime where in-sample quality stops predicting out-of-sample quality. The
selection step prints the expected-by-chance count every window for exactly this
reason. A tighter `pvalue_threshold` is the obvious robustness check.

**Parameter selection is not fully clean.** The Kalman `delta` (how fast β may
drift) was originally 1e-4, which produced a z-score with a one-day half-life —
internally inconsistent with a selection filter demanding 5–60 day half-lives. The
filter was adapting so fast it absorbed the very spread it was meant to trade,
and the strategy lost money (Sharpe −0.48) churning noise at 1.6-day holding
periods. Slowing it to 1e-8 fixes that inconsistency and is defensible on those
grounds. **But the value was chosen while looking at out-of-sample results**, which
is the multiple-testing trap applied to parameters instead of pairs. The honest
version selects `delta` per window on training data alone. Treat the headline Sharpe
as optimistic.

**Costs depend entirely on the execution vehicle.** The 10bps modelled here covers
commission and slippage only.

- *Physical shares:* UK stamp duty adds 0.5% on every purchase. Since one leg of
  each pair is bought, roughly 340 trades × 50bps × ~0.5 gross weight consumes on
  the order of 85% of capital across the decade — more than the entire +67.6%. On
  this route the strategy is almost certainly dead.
- *CFDs or spread bets:* no stamp duty, because no beneficial ownership transfers.
  Instead you pay overnight financing — roughly the benchmark rate ± a 2–3% broker
  spread on each leg. The benchmark rate largely cancels in a market-neutral pair;
  the spread does not. At roughly 31% time-in-market that implies a drag near
  0.5–0.8% per year, which the strategy survives. Spread bets are additionally
  exempt from UK CGT.

The CFD route also brings wider spreads, counterparty risk, and margin calls that
could force liquidation precisely during the dislocations the strategy exists to
hold through. The financing figures above are rules of thumb, not quoted rates.
Short borrow costs on the physical route are likewise unmodelled.

**Survivorship bias.** The universe is drawn from companies that are in the index
*today*, so it excludes firms that were delisted or collapsed during the period.
This flatters results to an unknown degree.

**The confidence interval is wide.** [0.42, 1.55] clears zero, but a lower bound of
0.42 is consistent with a fairly mediocre strategy. Combined with the `delta`
caveat, the honest reading is that an edge is plausible but not established.
