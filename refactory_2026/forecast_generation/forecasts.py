"""Turning raw signals into forecasts that mean the same thing everywhere.

A forecast of 10 means "hold the average position", and 20 means "hold twice
it". Getting a raw signal to that convention takes two steps.

SCALING

After dividing by volatility, a raw EWMAC averages about 3.4 in absolute terms,
not 10. So it is multiplied by a scalar: 10 / 3.41 for US10, 10 / 3.38 for
SP500_micro, which land near 2.93 and 2.96.

The scalar is estimated from the signal's own history, using every observation
up to the date in question and requiring at least 500 of them. That is about two
years, so no forecast exists before then.

NO BACKFILL, DELIBERATELY

pysystemtrade fills that opening gap by taking the first scalar it can compute
and applying it backwards, with the comment:

    ## BACKFILL OUR FIRST ESTIMATE, SLIGHTLY CHEATING, BUT...

In 1982 that scales US10 positions using the average signal size of 1982 to
1984, which was not knowable at the time. We do not do it. Each instrument's
forecasts simply begin once the scalar exists:

    CORN         first price 1972-10-18   forecasts from 1974-10 onward
    US10         first price 1982-08-30   forecasts from 1984-08 onward
    SP500_micro  first price 1982-09-14   forecasts from 1984-09 onward
    SOFR         first price 1984-03-23   forecasts from 1986-03 onward

The cost is about two years per instrument out of forty. The benefit is that no
forecast is scaled by information from its own future. This is a divergence from
the old build, so early-period results will not match it.

Scalars are estimated per instrument, not pooled across them, matching
`forecast_scalar_estimate["pool_instruments"] = False` in the reference config.

CAPPING

Scaled forecasts are clipped to +/-20, twice the target average. A rule that
becomes extremely confident is allowed to double up and no more: without a cap,
one instrument in a strong trend could dominate the whole portfolio's risk.

OUTPUT

One DataFrame per instrument: rows are trading days, columns are rule names.
Instruments are kept in a dict rather than a single MultiIndexed frame, so their
different histories stay visible and combining them has to be done explicitly.
"""

from __future__ import annotations

import pandas as pd

from refactory_2026 import config
from refactory_2026.data_cleaning.daily_prices import adjusted_price
from refactory_2026.forecast_generation.trading_rules import DEFAULT_RULES, Rule
from refactory_2026.volatility import volatility_of_prices

# The average and the cap come from config, because forecast weighting and
# position sizing have to agree with this stage about what a forecast of 10
# means. The 500 observations below are nobody else's business.

# Observations required before a scalar is estimated at all.
MIN_OBSERVATIONS_FOR_SCALAR = 500


def forecast_scalar(raw: pd.Series) -> pd.Series:
    """The multiplier that brings a raw signal to an average absolute value of 10.

    Expanding, so each day uses only that day's history and earlier. NaN until
    `MIN_OBSERVATIONS_FOR_SCALAR` observations exist, and left NaN rather than
    backfilled.
    """
    average_abs = raw.abs().expanding(min_periods=MIN_OBSERVATIONS_FOR_SCALAR).mean()

    return config.AVERAGE_ABS_FORECAST / average_abs


def scaled_forecast(raw: pd.Series) -> pd.Series:
    """A raw signal, scaled to the forecast convention and capped."""
    scaled = raw * forecast_scalar(raw)

    return scaled.clip(lower=-config.MAX_ABS_FORECAST, upper=config.MAX_ABS_FORECAST)


def forecasts_from(
    price: pd.Series, vol: pd.Series, rules: list[Rule] = DEFAULT_RULES
) -> pd.DataFrame:
    """Finished forecasts for one instrument, given its price and volatility.

    Takes series rather than an instrument code, so the whole stage can be
    exercised on a hand-built example with nothing on disk.
    """
    _check_names_are_unique(rules)

    columns = {
        rule.name: scaled_forecast(rule.raw_forecast(price, vol)) for rule in rules
    }

    # Every rule starts once its scalar exists; dropping incomplete rows gives
    # all of them a common start date, and leaves no missing values behind.
    return pd.DataFrame(columns).dropna(how="any")


def forecasts_for(code: str, rules: list[Rule] = DEFAULT_RULES) -> pd.DataFrame:
    """Finished forecasts for one instrument, reading its prices."""
    price = adjusted_price(code)

    return forecasts_from(price, volatility_of_prices(price), rules)


def all_forecasts(
    codes: list[str] | None = None, rules: list[Rule] = DEFAULT_RULES
) -> dict[str, pd.DataFrame]:
    """Forecasts per instrument, keyed by code.

    A dict, not a MultiIndexed frame: instruments have different histories, and
    anything combining them must align them on purpose.
    """
    if codes is None:
        codes = config.INSTRUMENTS

    return {code: forecasts_for(code, rules) for code in codes}


def _check_names_are_unique(rules: list[Rule]) -> None:
    names = [rule.name for rule in rules]
    duplicated = {name for name in names if names.count(name) > 1}

    if duplicated:
        raise ValueError(
            f"Rule names must be unique, since weights attach to them: {duplicated}"
        )
