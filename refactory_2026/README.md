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
| 0 | Config and data | done |
| 1 | Returns and volatility | done |
| 2 | Forecasts per rule | done |
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

### `data_cleaning/`

Turns the price files into one closing price per trading day. The only place
that reads price files, and the only place that changes frequency.

`daily_prices.py` holds the rule and the loaders. A trading day is the window
`(23:00 on the previous day, 23:00 today]`, and its close is the last print
inside it. The boundary matters because these sessions run past midnight: on
Monday 2018-09-03 the prints at 23:15, 23:30 and 23:45 are the opening of
*Tuesday's* session, so taking the last print of the calendar date would use a
Tuesday price as Monday's close. pysystemtrade's own `resample("1B").last()`
does exactly that, and additionally folds Sunday prints backward into Friday,
overwriting 91 Friday closes for US10.

The close is the last print in the window rather than the 23:00 print
specifically. A 23:00 print supplies it about 95% of the time; requiring one
would blank roughly 500 real trading days per instrument, mostly in 2020 to
2022 where the feed changes shape and days end at 19:00, 21:00 or 22:00.

Sessions dated at a weekend are dropped. They come from prints stamped exactly
23:00 on a Sunday, which open Monday's session; Monday always has its own later
prints here, so no close is lost. Sunday prints from 23:15 onward already fall
in Monday's window and are kept.

`daily_prices.py` also exposes the two loaders, `adjusted_price(code)` and
`current_contract_price(code)`. `cleaning_report.py` prints what cleaning
discarded and is imported by nothing in the pipeline. See
[`data_cleaning/README.md`](data_cleaning/README.md) for the detail.

Only days with prices appear in the output. Missing weekdays are left out, not
inserted as NaN, because this data cannot tell a market holiday from a hole in
the feed. Around 247 of each instrument's weekday gaps are days when nothing
traded anywhere, while 27 to 139 are days when the other instruments did trade;
those include genuine product-specific holidays such as SOFR's silence on
Thanksgiving 1997, not only feed failures. Inventing a NaN row for each would
assert a distinction the data does not support, so the gaps are reported by
`gaps_by_year` instead.

### `volatility.py`

How much an instrument moves in a day, in price points. Points rather than
percent because the adjusted series goes negative, so a percentage return would
be meaningless: US10 starts at −36.4.

It sits outside both stages that use it, because two stages use it. Forecasts
divide by volatility to make signals comparable — raw ewmac32 averages 1.46 for
US10 and 40.66 for SP500_micro, but 3.41 and 3.38 after dividing. Position sizing
divides by it to decide how many contracts carry a normal amount of risk: one
US10 contract swings about $433 a day against SOFR's $180, so the same risk is
23 contracts of one and 56 of the other. Were it owned by either stage, the other
would need its own copy, and two copies drifting apart is finding C3.

The estimator matches the reference default: an EWMA standard deviation over 35
days, blended 70/30 with a 10-year EWMA of itself. The blend stops position sizes
exploding when recent volatility collapses.

One accepted consequence of dropping holiday rows: differencing can span a
holiday, so about one day in 26 is a two or three day change, which overstates
volatility very slightly. Correcting it would mean discarding observations to
flatter a variance estimate, which is worse.

### `forecast_generation/`

One forecast per trading rule, on the convention that 10 means hold the average
position and 20 means twice it. `rules.py` holds the rule functions and the
parameter grids; `forecasts.py` holds the scaling and capping every rule passes
through. Output is `{code: DataFrame}` with rule names as columns.

Volatility is passed into a rule, never fetched by it, so a rule cannot diverge
from the volatility used to size positions. Rule names are derived from
parameters (`ewmac_8_32`), never typed, because the name is the key a forecast
weight later attaches to.

Forecast scalars need 500 observations and are **not** backfilled, so forecasts
start about two years into each instrument's history. pysystemtrade backfills and
calls it "SLIGHTLY CHEATING" in its own source; we take the shorter history
instead. See [`forecast_generation/README.md`](forecast_generation/README.md).

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

1. **One close per trading day is the canonical frequency.** Exactly one
   function collapses prints to days, `daily_prices.daily_close`. Nothing else
   resamples, and a trading day is a session window, never a calendar date.
2. **A date in a price series is a day that traded.** Blank prints are dropped
   on load, and weekdays without prices are absent rather than NaN. So a
   cleaned price series contains no missing values at all, and no stage may
   fill one with zero. Anything counting rows is counting real observations,
   but the rows are not a regular calendar: instruments must be aligned
   explicitly when they are combined, never assumed to share an index.
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
