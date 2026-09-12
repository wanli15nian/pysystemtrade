# refactory_2026

A rewrite of the futures backtest that lives in `../refactory`.

`../refactory` is **reference only**. It is read to understand what the stages
of the pipeline are, and for nothing else. None of its code, structure or naming
is carried over. It was abandoned because its execution timing was scattered
across the codebase as individual `.shift()` calls that could not be traced, and
because its pipeline ran at module import, leaving no seam at which any
intermediate value could be inspected or tested.

Same scope as that build, so the two stay comparable: 4 instruments (CORN, SOFR,
SP500_micro, US10), 2 EWMAC rules (8/32 and 32/128), $1m of fixed notional
capital, 16% annual volatility target.

## Status

Being built one stage at a time. Only the first stage is started.

| Stage | What it does | State |
|---|---|---|
| 0 | Config and data | in progress |
| 1 | Returns and volatility | not started |
| 2 | Forecasts per rule | not started |
| 3 | Accounting at rule level | not started |
| 4 | Forecast weights and FDM | not started |
| 5 | Position sizing | not started |
| 6 | Accounting at subsystem level | not started |
| 7 | Instrument weights and IDM | not started |
| 8 | Portfolio positions | not started |
| 9 | Accounting at portfolio level | not started |
| 10 | Reporting | not started |

Stages 3, 6 and 9 are the same component applied at three levels, not three
separate pieces of code. Each weighting stage is fitted on the P&L of the stage
below it, which is what fixes this ordering.

## Files

### `config.py`

Run-level settings, and nothing else: no functions, no logic, no work done at
import. It exists so that reading a setting can never start a backtest, which is
what happened in the old build, where the settings sat at the top of the module
that ran the whole pipeline.

Holds the data directory, the instrument list, capital, the risk target, and the
annualisation constant. Paths resolve relative to the file, so the code behaves
the same from any working directory. Anything belonging to one stage alone stays
with that stage, so this file is not allowed to become a list of every parameter
in the system.

### `instruments.py`

Turns `data/csvconfig` into one validated table of static per-instrument
numbers, and is the only module that reads those files.

It exists because the values a backtest needs are split across three CSVs and
one has to be derived: `rolls_per_year` is the length of the `HoldRollCycle`
string, so `Z` means one roll a year and `HMUZ` means four. Doing that join in
one place, once, keeps the rest of the system from repeating it.

It also validates at load time rather than failing deep inside a calculation.
Requested instruments must exist, required fields must be numeric and present,
and every instrument must be priced in USD, since there is no currency
conversion. Rows are returned in the order requested, never sorted, so nothing
downstream can come to rely on alphabetical order.

`load_instrument_config()` returns a DataFrame indexed by instrument code, with
columns `currency_per_point`, `currency`, `commission_per_block`,
`commission_percentage`, `commission_per_trade`, `spread_cost_points`,
`rolls_per_year`.

Two columns carry their unit in the name because the source names hide it and
neither value is money on its own. `Pointsize` became `currency_per_point`: it is
a conversion factor, not the size of a price increment, and writing it as
`X_per_Y` makes a units error visible where it is used. `SpreadCost` became
`spread_cost_points`: the word cost implies currency, but crossing the spread on
one US10 contract is 0.008 *points*, which is $8 only after multiplying by
`currency_per_point`. Getting that wrong is a 1000x error the old name would not
have warned about.

The commission columns carry the `commission_` prefix because the names in the
source file (`PerBlock`, `Percentage`, `PerTrade`) do not say what they charge
for. They are alternative charging structures rather than charges to be summed:
the commission on a fill is the largest of the three.

## Data

`data/` holds only what this build needs: the four instruments above, plus all
fx rates and the four config files. The full 239-instrument set is in
`../refactory/data`.

| Folder | Contents | Used for |
|---|---|---|
| `adjusted_prices_csv` | back-adjusted continuous price | returns, volatility, P&L |
| `multiple_prices_csv` | current contract price and carry | money amounts |
| `roll_calendars_csv` | roll dates | not yet used |
| `fx_prices_csv` | fx rates | not used: all four instruments are USD |
| `csvconfig` | instrument, roll and spread configuration | `instruments.py` |

Facts about these files that shape the code:

- They are **intraday**, up to 90 rows a day at irregular times, not one row per
  day. Prices have to be collapsed to a daily figure deliberately.
- Some price cells are **blank**: 8,581 in CORN, roughly 1,500 to 2,700 in the
  others.
- Adjusted prices are back-adjusted and can be **negative**. US10 starts at
  -36.4. Only differences are meaningful; an adjusted price is never a money
  amount.
- Histories **start and end on different dates**, from 1972 to 1984 and CORN a
  day short of the rest. Instruments are not padded to a shared calendar.

## Conventions

Decided once, recorded here, and not to be re-decided per file.

1. **Business-day close is the canonical frequency.** Exactly one function
   changes frequency, and it lives in the price module. Nothing else resamples.
2. **NaN means not tradeable that day.** Blank rows are dropped when loaded;
   after that a NaN stays a NaN. No filling with zero in the data layer.
3. **Two price types, deliberately awkward to confuse.** Adjusted price for
   returns, volatility and P&L. Raw price for money amounts. Separate functions,
   different names, so a mix-up is visible where it is called.
4. **Index contract:** timezone-naive `DatetimeIndex`, sorted, unique, named
   `date`, checked on load.
5. **Units belong in the name** wherever a number is not what it appears to be.
   `currency_per_point` converts a price difference into currency;
   `spread_cost_points` is in price points, not money.
6. **Timing lives in one place.** When positions arrive, a single function will
   turn a decision series into the series that is held (for P&L) and the series
   that is traded (for costs), with the lag as one parameter. No other module
   shifts anything. This is the rule the old build broke.

## Running it

Imports are absolute from the repository root:

```python
from refactory_2026 import config
from refactory_2026.instruments import load_instrument_config

load_instrument_config()
```

Start Python from the repository root, or put that directory on `PYTHONPATH`.
Paths inside the package do not depend on the working directory.
