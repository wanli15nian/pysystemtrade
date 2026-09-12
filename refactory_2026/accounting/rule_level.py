"""Stage 3: what each trading rule earned, on each instrument, alone.

This is a *caller* of the accounting component, not part of it. It answers one
question: if this rule traded this instrument by itself, using the whole risk
budget, what would it have made after costs?

That is a hypothetical, and deliberately so. Eight of them exist here - four
instruments times two rules - and they cannot all be true at once, since each
assumes the entire capital. They are measuring devices, not a book.

The hypothetical is well posed because every one of them uses the same capital
and the same risk target, which is what makes their dollar P&L comparable.
Stage 4 fits forecast weights on exactly these series: you cannot decide how
much to trust a fast rule against a slow one without knowing what each earned
*after* the costs of trading at its own speed.

Positions here are not rounded and not buffered. Those belong to the levels
where orders are real.
"""

from __future__ import annotations

import pandas as pd

from refactory_2026 import config
from refactory_2026.accounting.accounting import account_for, contract_for
from refactory_2026.data_cleaning.daily_prices import adjusted_price
from refactory_2026.forecast_generation.forecasts import forecasts_for
from refactory_2026.forecast_generation.trading_rules import DEFAULT_RULES, Rule
from refactory_2026.position_sizing import average_position, position_from_forecast
from refactory_2026.volatility import volatility_of_prices


def rule_positions(code: str, rules: list[Rule] = DEFAULT_RULES) -> pd.DataFrame:
    """The position each rule would decide on, day by day, if it traded alone."""
    prices = adjusted_price(code)
    forecasts = forecasts_for(code, rules)

    normal = average_position(
        vol_points=volatility_of_prices(prices).reindex(forecasts.index),
        currency_per_point=contract_for(code).currency_per_point,
    )

    return forecasts.apply(lambda forecast: position_from_forecast(forecast, normal))


def rule_accounts(
    code: str, rules: list[Rule] = DEFAULT_RULES
) -> dict[str, pd.DataFrame]:
    """Gross, costs and net per rule for one instrument, keyed by rule name."""
    positions = rule_positions(code, rules)

    return {rule: account_for(code, positions[rule]) for rule in positions.columns}


def rule_net_returns(
    codes: list[str] | None = None, rules: list[Rule] = DEFAULT_RULES
) -> dict[str, pd.DataFrame]:
    """Net dollars per day, per rule, per instrument: the input to stage 4.

    A dict of frames rather than one MultiIndexed frame, because instruments
    have different histories and combining them must be done on purpose.
    """
    if codes is None:
        codes = config.INSTRUMENTS

    return {
        code: pd.DataFrame(
            {rule: accounts["net"] for rule, accounts in rule_accounts(code, rules).items()}
        )
        for code in codes
    }
