# forecast_generation

Turns prices and volatility into one forecast per trading rule.

A forecast is a number on a fixed convention: **10 means hold the average
position, 20 means hold twice it, negative means short.** That convention is
what lets signals from different rules and different instruments be compared,
weighted and added together later.

## Files

| File | Contains | You touch it when |
|---|---|---|
| `trading_rules.py` | The rule functions, the `Rule` record, and the parameter grids | Adding or changing what you trade |
| `forecasts.py` | Scalar estimation, capping, assembling the output | Almost never |

The split is by what changes together. Adding a carry rule means a new function
and a new grid, both in `trading_rules.py`; the machinery that turns any raw
signal into a forecast does not move.

## Inputs and outputs

**In:** an adjusted price series and a volatility series, plus the rules to run.

**Out:** `{code: DataFrame}` — rows are trading days, columns are rule names,
values are scaled, capped forecasts.

```python
from refactory_2026.forecast_generation.forecasts import forecasts_for, all_forecasts

forecasts_for("US10")      # DataFrame: ewmac_8_32, ewmac_32_128
all_forecasts()            # dict, one entry per configured instrument
```

A dict rather than one MultiIndexed frame, because instruments have genuinely
different histories (CORN from 1972, SOFR from 1984) and anything combining them
should have to align them on purpose. The old build packed everything into a
MultiIndex and then needed `unstack().stack().droplevel().sort_index()` chains to
get it back out; worse, MultiIndex operations align silently, and silent
alignment across unequal histories is what produced findings C4 and C5.

## Volatility is passed in, never fetched

```python
ewmac(price, vol, fast=8, slow=32)      # the dependency is visible
ewmac(code, fast=8, slow=32)            # it is not, and can diverge
```

A rule that fetched its own volatility could disagree with the volatility used
to size positions. That is finding C3: the old build normalised forecasts by the
volatility of price *levels* while sizing positions by the volatility of price
*changes*, and nothing made the discrepancy visible.

This is also why `volatility.py` sits outside this folder — position sizing needs
the same numbers.

## Names are derived, never typed

```python
EWMAC_SPANS = [(8, 32), (32, 128)]
DEFAULT_RULES = [Rule(ewmac, fast=f, slow=s) for f, s in EWMAC_SPANS]
```

`Rule.name` is built from the function name and the parameter values, giving
`ewmac_8_32`. Adding a variation is one tuple; no label is invented.

This matters because the name is not decoration — it becomes a column, and later
the key a forecast weight attaches to. A hand-typed name can disagree with its
own parameters, and would silently break the link between a signal and its
weight. `forecasts_from` rejects duplicate names for the same reason.

`ewmac_8_32` names both spans. pysystemtrade writes `ewmac8`, leaving the slow
span to an unstated convention that it is always four times the fast one — which
stops being true the moment you try `(8, 64)`.

## The two steps applied to every raw signal

### 1. Scaling

A raw EWMAC divided by volatility averages about 3.4 in absolute value, not 10:

| Instrument | raw mean abs | after ÷ volatility | scalar to reach 10 |
|---|---|---|---|
| US10 | 1.463 | 3.414 | 2.93 |
| SP500_micro | 40.661 | 3.382 | 2.96 |

Raw, SP500_micro's signal is 28× larger than US10's, purely because its price is
bigger. After dividing by volatility they are almost identical — that is the
whole purpose of the step.

The scalar uses an expanding window of the signal's own history and needs **500
observations**, about two years.

The 10 and the 20 are `config.AVERAGE_ABS_FORECAST` and
`config.MAX_ABS_FORECAST`, not local constants, because other stages must agree
on them: position sizing divides a forecast by the average to get contracts, and
the combined forecast is capped at the same maximum. Were the scaling target
changed here alone, every position would be wrong by a constant factor with
nothing raising an error. The 500-observation requirement stays local, since only
this module uses it.

### 2. Capping

Scaled forecasts are clipped to ±20, twice the target. A rule may double up when
very confident and no more; uncapped, one instrument in a violent trend could
dominate the portfolio's risk.

## No backfill: forecasts start late, on purpose

The scalar cannot be computed for the first 500 observations. pysystemtrade fills
that gap by applying its first scalar backwards, commenting:

```
## BACKFILL OUR FIRST ESTIMATE, SLIGHTLY CHEATING, BUT...
```

That scales 1982 positions using the average signal size of 1982–84, which
nobody knew in 1982. **We do not do it.** Forecasts begin when the scalar exists:

| Instrument | First price | Forecasts from |
|---|---|---|
| CORN | 1972-10-18 | 1974-10 |
| SP500_micro | 1982-09-14 | 1984-09 |
| US10 | 1982-08-30 | 1984-08 |
| SOFR | 1984-03-23 | 1986-03 |

It costs about two years per instrument out of forty, and it means early-period
results will not match the old build. It buys the guarantee that no forecast is
scaled using information from its own future.

Scalars are estimated per instrument, never pooled, matching
`forecast_scalar_estimate["pool_instruments"] = False` in the reference config.

## What is deliberately not here

**Forecast weights, the diversification multiplier, and the combined forecast.**
They look forecast-shaped, so they feel like they belong here. They do not: the
weights are fitted on the *P&L of each rule*, which needs position sizing and
cost modelling. Putting them here would make this folder depend on stages
downstream of it.

The boundary: **this folder produces one forecast per rule, and knows nothing
about how they are combined or how much money is at stake.**
