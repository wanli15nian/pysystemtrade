# data_cleaning

Turns the price files in `../data` into one closing price per trading day. This
is the only part of the system that reads price files, and the only part that
changes frequency. Everything downstream receives daily series and never learns
that the files are intraday.

## Files

| File | Job | Imported by the pipeline? |
|---|---|---|
| `daily_prices.py` | The trading-day rule, and the two price loaders that apply it | yes |
| `cleaning_report.py` | Counts what cleaning discarded, for a human to read | **no** |

The split is by audience, not by topic. `daily_prices.py` is what the pipeline
calls; `cleaning_report.py` is what you read when you want to know whether the
cleaning assumptions still hold. Keeping them apart makes that boundary visible
in the file list.

## `daily_prices.py`

### Why a trading day is not a calendar day

CME futures open on Sunday evening, and each session runs on past midnight. So
the last print bearing a date usually belongs to the *next* day's session. On
Monday 2018-09-03 the US10 file holds prints at 17:45, 23:00, 23:15, 23:30 and
23:45 — only the first two are Monday's. Taking the last print of the calendar
date hands you a Tuesday price as Monday's close.

pysystemtrade does that, via `resample("1B").last()`, and additionally folds
Sunday prints backward into Friday: 91 Friday closes for US10 are overwritten
with prices from a session two days later.

### The rule

A trading day is the window `(23:00 on the previous day, 23:00 today]`, and its
close is the last print inside it.

23:00 is the boundary because every weekday carries a print stamped exactly
23:00 — 312 of 312 weekdays in 2019, 166 of 166 in 2023 — and the intraday grid
resumes at 23:15.

The close is the last print in the window, *not* the 23:00 print specifically. A
23:00 print supplies it about 95% of the time; insisting on one would blank
roughly 500 real trading days per instrument, mostly across 2020 to 2022 where
the feed changes shape and days end at 19:00, 21:00 or 22:00.

### What gets dropped, and why

**Weekend-dated sessions.** These come from a print stamped exactly 23:00 on a
Sunday, which opens Monday's session. Monday has its own later prints in every
case in this data, so no close is lost. Sunday prints from 23:15 onward already
fall inside Monday's window and are kept — they are real traded prices.

**Weekdays with no prints**, which are simply absent rather than NaN. This data
cannot distinguish a market holiday from a hole in the feed: about 247 of each
instrument's weekday gaps are days when no instrument traded, but 27 to 139 are
days when the others did, and those include genuine product-specific holidays
(SOFR is silent on Thanksgiving 1997 while the rest trade) as well as any real
feed failure. Inventing a NaN row would assert a distinction the data does not
support, so `gaps_by_year` reports them instead.

The consequence: **a cleaned price series contains no missing values, and its
index is not a regular calendar.** Instruments must be aligned explicitly when
combined, never assumed to share an index.

### The two price series

They are not interchangeable, and they are separate functions rather than one
function with a flag, so that using the wrong one is visible at the call site.

| Function | Returns | Use for |
|---|---|---|
| `adjusted_price(code)` | back-adjusted continuous history | returns, volatility, P&L |
| `current_contract_price(code)` | price of the contract held | anything denominated in money |

The adjusted series is stitched across contract rolls, so it is not a price
anyone could trade at: US10 starts at −36.4. Only its differences carry meaning.
The current-contract series is a real price; multiply it by `currency_per_point`
to get currency.

### Public names

```python
SESSION_CLOSE_HOUR                 # 23
trading_day_of(stamps)             # which trading day each timestamp belongs to
trading_day_closes(prints)         # last print per day, weekends still included
closing_stamps(prints)             # hour that closed each day; diagnostic
daily_close(prints)                # the finished rule: weekends dropped, validated
adjusted_price(code)
current_contract_price(code)
raw_prints(code, series)           # the unfiltered file contents
```

`trading_day_of` and `daily_close` take a series and touch no files, so the rule
can be tested against a hand-built example with no data on disk.

## `cleaning_report.py`

Run it:

```bash
python -m refactory_2026.data_cleaning.cleaning_report
```

`cleaning_report()` gives one row per instrument — sessions kept, date range,
share of closes taken at 23:00, weekend sessions dropped, blank prints dropped,
weekday gaps. `gaps_by_year(code)` breaks the gaps down by year.

Current output:

```
             sessions       first        last  closed_at_2300  weekend_dropped  blanks_dropped  weekday_gaps
CORN            12790  1972-10-18  2023-08-30            95.9               87            8581           481
SOFR             9921  1984-03-23  2023-08-31            95.0              115            2750           369
SP500_micro     10415  1982-09-14  2023-08-31            95.8              156            1537           273
US10            10324  1982-08-30  2023-08-31            94.8               91            1719           375
```

`closed_at_2300` is the figure to watch. At ~95% the boundary assumption holds.
If a new file pushed it far lower, the 23:00 rule has stopped describing the
data and this module needs revisiting.

Every number is derived from `daily_prices`, never recomputed, so the report
cannot drift away from what the pipeline actually does.
