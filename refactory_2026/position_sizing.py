"""Turning a forecast into a number of contracts.

Two steps, both shared by more than one stage, which is why they live here
rather than inside either.

    average position    how many contracts represent a normal amount of risk.
                        Stage 3 uses it to build the hypothetical position of a
                        single rule; stage 5 uses it for the real one.

    position            average position, scaled by how strong the forecast is.

THE RISK TARGET

We want the portfolio to move about `RISK_TARGET` (16%) of capital in a year,
which is 16% / sqrt(256) = 1% of capital in a day: $10,000 on $1m. Divide that
by how much one contract moves in a day and you have the number of contracts
representing a normal position.

    US10:  one contract swings about $433 a day  ->  23 contracts
    SOFR:  one contract swings about $180 a day  ->  56 contracts

Both carry the same risk despite being very different contract counts. That is
the entire purpose of sizing this way: it makes a bond future and an equity
future comparable, and it is why volatility has to be measured before anything
can be traded.

FORECAST TO POSITION

A forecast of `AVERAGE_ABS_FORECAST` (10) means "hold the average position", so
the position is simply the average scaled by forecast / 10. A forecast of 20,
the cap, means twice the average; a forecast of -15 means one and a half times
the average, short.

Positions here are not rounded to whole contracts. At rule level they are a
hypothetical used to compare rules, where fractional contracts are the correct
thing to measure. Rounding happens further down, where orders are real.
"""

from __future__ import annotations

import pandas as pd

from refactory_2026 import config


def average_position(
    vol_points: pd.Series,
    currency_per_point: float,
    capital: float = config.CAPITAL,
    risk_target: float = config.RISK_TARGET,
) -> pd.Series:
    """Contracts representing a normal amount of risk, day by day.

    Falls when the instrument becomes more volatile, rises when it calms down:
    the risk being held stays the same size while the market changes around it.
    """
    daily_risk_target = risk_target / (config.BUSINESS_DAYS_IN_YEAR**0.5)
    daily_cash_vol_target = capital * daily_risk_target

    cash_vol_per_contract = vol_points * currency_per_point

    return daily_cash_vol_target / cash_vol_per_contract


def position_from_forecast(
    forecast: pd.Series, average_position: pd.Series
) -> pd.Series:
    """Contracts implied by a forecast, given what a normal position is.

    This is a *decision*: the position chosen at each day's close. What is held
    and what is traded follow from it in `accounting.timing`, never here.
    """
    return forecast / config.AVERAGE_ABS_FORECAST * average_position
