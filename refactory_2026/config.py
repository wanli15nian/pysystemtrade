"""Run-level settings.

Importing this module never executes anything: it holds only values, so that
reading a setting can never trigger a backtest as a side effect.

What belongs here: values shared across the whole run. What does not: anything
used by a single stage, which belongs with that stage.
"""

from pathlib import Path

# Paths resolve from this file, not the working directory, so the code runs the
# same way whatever folder it is started from.
PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"

# Instruments this build trades. The same four as the old refactory build, so
# results stay comparable with it.
INSTRUMENTS = ["CORN", "SOFR", "SP500_micro", "US10"]

# Notional capital, in the base currency. Fixed: profits are not reinvested.
CAPITAL = 1_000_000

# Target annual volatility of the portfolio, as a fraction of capital.
RISK_TARGET = 0.16

# The forecast convention. A forecast of AVERAGE_ABS_FORECAST means "hold the
# average position", and forecasts are clipped to MAX_ABS_FORECAST either side.
#
# These live here rather than with the forecast code because more than one stage
# has to agree on them: forecast generation scales signals *to* the average,
# position sizing divides *by* it to turn a forecast into contracts, and the
# combined forecast is capped at the same maximum. If the scaling target were
# retuned and position sizing kept dividing by the old number, every position
# would be wrong by a constant factor and nothing would raise an error.
AVERAGE_ABS_FORECAST = 10.0
MAX_ABS_FORECAST = 20.0

# Trading days between deciding a position and trading it. A forecast computed
# from Monday's close is traded at Tuesday's close, and first earns a price
# change on Wednesday.
#
# This is a statement about execution, not a parameter to tune. Setting it to
# zero claims you traded at the very close that produced the signal, which is
# impossible but only marginally optimistic; anything negative would mean
# trading before the signal existed. Measured across the eight instrument/rule
# pairs, a lag of 0 and a lag of 1 give mean Sharpe 0.307 and 0.308, while
# omitting the lag entirely gives 1.164 - all of that difference being
# look-ahead rather than profit.
#
# Every accounting level reads this one value, so they cannot disagree.
TRADE_LAG_DAYS = 1

# Business days per year, for annualising daily figures.
# Note: a business-day calendar actually holds about 261 days a year. This 256
# is pysystemtrade's convention (16 * 16, so the daily-to-annual vol factor is
# exactly 16). Whichever number a statistic uses, it must use it consistently.
BUSINESS_DAYS_IN_YEAR = 256
