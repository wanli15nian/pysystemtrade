"""How much an instrument moves in a day, measured in price points.

WHY POINTS AND NOT PERCENT

The adjusted price series is stitched across contract rolls and goes negative:
US10 starts at -36.4. A percentage return computed from it would be meaningless,
and would flip sign as the price crossed zero. Position sizing works in points
multiplied by `currency_per_point` anyway, so points are the natural unit.
pysystemtrade says the same thing in `rawdata.py`: "volatility of daily returns
(not % returns)".

WHY THIS LIVES OUTSIDE BOTH STAGES THAT USE IT

Two stages consume volatility, for different purposes:

  forecast normalisation  a raw EWMAC is in the instrument's own price units, so
                          it is not comparable across instruments. Dividing by
                          volatility makes it so: raw ewmac32 averages 1.46 for
                          US10 and 40.66 for SP500_micro, but 3.41 and 3.38
                          after dividing.

  position sizing         how many contracts represent a normal amount of risk.
                          One US10 contract swings about $433 a day, one SOFR
                          contract about $180, so the same risk is 23 contracts
                          of the former and 56 of the latter.

If volatility lived inside either stage, the other would have to reach into it
or compute its own copy. Two copies is exactly how the old build ended up
normalising forecasts by the volatility of price *levels* while sizing positions
by the volatility of price *changes* (finding C3).

THE ESTIMATOR

Matches pysystemtrade's configured default (`volatility_calculation` in
defaults.yaml): an EWMA standard deviation over 35 days, needing 10
observations, blended 70/30 with a 10-year EWMA of that same estimate.

The blend is a judgement, not a law. Its purpose is to stop position sizes
exploding when recent volatility collapses: a pure 35-day estimate can fall far
below anything sustainable, and dividing by a too-small number buys too much.

A HOLIDAY MAKES A MULTI-DAY CHANGE

Our price series omits weekdays that never traded, so differencing can span a
holiday and cover two or three days. About one day in 26 is affected, and a
three-day change has around sqrt(3) times the standard deviation, so volatility
comes out very slightly overstated.

This is accepted rather than corrected. Dropping observations to flatter a
variance estimate is worse than a known small bias. pysystemtrade instead keeps
NaN rows for holidays and loses those observations entirely.
"""

from __future__ import annotations

import pandas as pd

from refactory_2026.config import BUSINESS_DAYS_IN_YEAR

# EWMA span for the responsive estimate, in days.
FAST_SPAN = 35

# Observations needed before an estimate is produced at all.
MIN_OBSERVATIONS = 10

# The slow estimate is an EWMA of the fast one over this many years.
SLOW_YEARS = 10

# Weight on the slow estimate; the rest goes on the fast one.
PROPORTION_SLOW = 0.3

# Floor, purely to stop division by zero when a price never moves.
MINIMUM_VOLATILITY = 0.0000000001


def price_change_points(prices: pd.Series) -> pd.Series:
    """Day-on-day change in price, in points.

    Not a return: see the module docstring on why percentages are unusable here.
    """
    return prices.diff()


def volatility_points(changes: pd.Series) -> pd.Series:
    """Blended EWMA standard deviation of daily price changes, in points."""
    fast = changes.ewm(span=FAST_SPAN, min_periods=MIN_OBSERVATIONS).std()
    slow = fast.ewm(span=SLOW_YEARS * BUSINESS_DAYS_IN_YEAR).mean()

    blended = PROPORTION_SLOW * slow + (1 - PROPORTION_SLOW) * fast

    return blended.clip(lower=MINIMUM_VOLATILITY)


def volatility_of_prices(prices: pd.Series) -> pd.Series:
    """Volatility straight from a price series, for callers that hold prices."""
    return volatility_points(price_change_points(prices))
