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

# Business days per year, for annualising daily figures.
# Note: a business-day calendar actually holds about 261 days a year. This 256
# is pysystemtrade's convention (16 * 16, so the daily-to-annual vol factor is
# exactly 16). Whichever number a statistic uses, it must use it consistently.
BUSINESS_DAYS_IN_YEAR = 256
