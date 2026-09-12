"""One closing price per trading day, per instrument.

THE PROBLEM

A trading day does not run midnight to midnight. CME futures open on Sunday
evening and each session runs on past midnight, so the last print bearing a
given date usually belongs to the *next* day's session. On Monday 2018-09-03
the US10 file holds prints at 17:45, 23:00, 23:15, 23:30 and 23:45; only the
first two are Monday's, and the rest open Tuesday's session. Taking the last
print of the calendar date would use a Tuesday price as Monday's close.

pysystemtrade does exactly that, via `resample("1B").last()`. It also folds
Sunday prints backward into Friday, overwriting 91 Friday closes for US10 with
prices from a session two days later.

THE RULE

The boundary sits at 23:00: every weekday carries a print stamped exactly 23:00
(312 of 312 weekdays in 2019, 166 of 166 in 2023) and the intraday grid resumes
at 23:15. So a trading day is the window

    (23:00 on the previous day, 23:00 today]

and its close is the last print inside that window.

The close is the last print in the window, not the 23:00 print specifically. A
23:00 print supplies it about 95% of the time; requiring one would blank roughly
500 real trading days per instrument, mostly in 2020 to 2022 where the feed
changes shape and days end at 19:00, 21:00 or 22:00.

WHAT IS DROPPED

Sessions dated at a weekend, which arise from a print stamped exactly 23:00 on a
Sunday: that print opens Monday's session, and Monday has its own later prints
in every case here, so no close is lost. Sunday prints from 23:15 onward already
fall inside Monday's window and are kept.

Weekdays with no prints are left out rather than inserted as NaN, because this
data cannot distinguish a market holiday from a hole in the feed. Around 247 of
each instrument's weekday gaps are days when nothing traded anywhere, but 27 to
139 are days when the other instruments did trade, and those include genuine
product-specific holidays as well as any real feed failure. A NaN row would
assert a distinction the data does not support. `cleaning_report.py` counts the
gaps instead.

TWO PRICE SERIES, NOT INTERCHANGEABLE

    adjusted_price          a stitched continuous history. US10 starts at -36.4,
                            so it is not tradeable and meaningless as a level;
                            only its differences mean anything. For returns,
                            volatility and P&L.

    current_contract_price  the price of the contract actually held. A real,
                            tradeable number. For anything denominated in money.

Separate functions rather than one function with a flag, so that using the wrong
one is visible where it is called.
"""

from __future__ import annotations

import pandas as pd

from refactory_2026 import config

# The hour at which one trading day ends and the next begins.
SESSION_CLOSE_HOUR = 23

ADJUSTED = "adjusted"
CURRENT_CONTRACT = "current_contract"

_DATE_COLUMN = "DATETIME"
_SOURCES = {
    ADJUSTED: (config.DATA_DIR / "adjusted_prices_csv", "price"),
    CURRENT_CONTRACT: (config.DATA_DIR / "multiple_prices_csv", "PRICE"),
}


# ----------------------------------------------------------------------------
# the rule
# ----------------------------------------------------------------------------


def trading_day_of(
    stamps: pd.DatetimeIndex, close_hour: int = SESSION_CLOSE_HOUR
) -> pd.DatetimeIndex:
    """The trading day whose window contains each timestamp.

    A stamp at exactly `close_hour` closes its own day; anything later belongs
    to the next. 23:00 Monday stays Monday, 23:15 Monday becomes Tuesday.
    """
    to_midnight = pd.Timedelta(hours=24 - close_hour)

    return (stamps + to_midnight - pd.Timedelta(nanoseconds=1)).floor("D")


def trading_day_closes(
    prints: pd.Series, close_hour: int = SESSION_CLOSE_HOUR
) -> pd.Series:
    """Last print of every trading day, weekend-dated days included.

    The step before weekends are dropped. Exposed because the report needs to
    see what cleaning discarded, and should not re-derive this itself.
    """
    observed = prints.dropna()

    return observed.groupby(trading_day_of(observed.index, close_hour)).last()


def closing_stamps(
    prints: pd.Series, close_hour: int = SESSION_CLOSE_HOUR
) -> pd.Series:
    """Hour of the print that closed each trading day. Diagnostic only."""
    observed = prints.dropna()
    grouped = observed.groupby(trading_day_of(observed.index, close_hour))

    return grouped.apply(lambda window: window.index[-1].hour)


def daily_close(prints: pd.Series, close_hour: int = SESSION_CLOSE_HOUR) -> pd.Series:
    """Reduce intraday prints to one closing price per trading day.

    Blanks are dropped first, so a missing value at the end of a window cannot
    displace a good print from earlier in it.
    """
    sessions = trading_day_closes(prints, close_hour)

    weekdays = sessions[sessions.index.dayofweek < 5]
    weekdays.index.name = "date"

    return _checked(weekdays)


def _checked(daily: pd.Series) -> pd.Series:
    """Enforce the index contract, so later stages can rely on it."""
    if daily.index.tz is not None:
        raise ValueError("Trading days must be timezone-naive")
    if not daily.index.is_unique:
        raise ValueError("Trading days must be unique")
    if not daily.index.is_monotonic_increasing:
        raise ValueError("Trading days must be sorted")
    if daily.isna().any():
        raise ValueError("A closing price cannot be missing: blanks are dropped")

    return daily


# ----------------------------------------------------------------------------
# reading the files
# ----------------------------------------------------------------------------


def adjusted_price(code: str) -> pd.Series:
    """Daily closes of the back-adjusted continuous series.

    For returns, volatility and P&L. Can be negative; never treat as money.
    """
    return _daily_price(code, ADJUSTED)


def current_contract_price(code: str) -> pd.Series:
    """Daily closes of the contract currently held.

    For money amounts. Multiply by `currency_per_point` to get currency.
    """
    return _daily_price(code, CURRENT_CONTRACT)


def raw_prints(code: str, series: str = ADJUSTED) -> pd.Series:
    """Every print in the file, blanks included: the input to cleaning."""
    directory, column = _source(series)
    path = directory / f"{code}.csv"

    if not path.exists():
        raise FileNotFoundError(f"No price file at {path}")

    table = pd.read_csv(path, parse_dates=[_DATE_COLUMN], index_col=_DATE_COLUMN)

    if column not in table.columns:
        raise ValueError(
            f"{path} has no '{column}' column: found {list(table.columns)}"
        )

    return table[column]


def _daily_price(code: str, series: str) -> pd.Series:
    daily = daily_close(raw_prints(code, series))

    if daily.empty:
        raise ValueError(f"No usable {series} prices for {code}")

    return daily.rename(code)


def _source(series: str):
    if series not in _SOURCES:
        raise ValueError(f"Unknown price series '{series}': expected {list(_SOURCES)}")

    return _SOURCES[series]
