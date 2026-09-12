"""What a position earned, and what it cost, in dollars per day.

One component, called at three levels. Only the position handed in differs:

    rule level        forecast of one rule, sized as if it traded alone
    subsystem level   combined forecast for one instrument, buffered and rounded
    portfolio level   subsystem position scaled by instrument weight and IDM

Everything after that is identical, so this code never learns which level it is
serving. There is no `level` argument and no branching on one. Whatever decides
what the position should be lives upstream: **accounting measures a position, it
never decides one.**

EVERYTHING IS IN DOLLARS

No Sharpe ratios, no cost-per-unit-of-volatility, no ratios of any kind. Gross
is contracts times the price move times what a point is worth; costs are
dollars paid to trade; net is the difference. Ratios belong to reporting and to
the optimiser, both of which can compute whatever they want from the dollars.

pysystemtrade expresses rule-level costs in Sharpe units because it runs
rule-level P&L on an arbitrary capital of 100, where a dollar cost would be
meaningless. We use real capital instead, which works because net P&L is exactly
linear in capital: doubling capital doubles every number and leaves every ratio
identical. That linearity is worth protecting - see the assumptions below.

WHAT A TRADE COSTS

Two things are paid on every contract that changes hands:

    spread       spread_cost_points * currency_per_point, roughly $8 for CORN,
                 SOFR and US10, and $0.95 for SP500_micro. This dominates:
                 73 to 84 per cent of the total for three of the four.

    commission   the largest of three structures, never their sum: a charge per
                 contract, a percentage of traded value, or a flat charge per
                 order. Only the per-contract one bites for these instruments.

ROLLS ARE A TAX ON HOLDING

A futures contract expires. To keep a position you close the expiring contract
and open the next, paying the full cost on your entire position whether or not
your forecast changed. CORN does this once a year, the others four times.

This is not a detail: for the slow rules roll costs are about 40 per cent of all
costs, because a slow rule barely trades and the holding tax dominates. Ignoring
them would make slow rules look systematically cheaper than they are, right
before stage 4 chooses between fast and slow.

Roll dates are synthesised by spacing `rolls_per_year` evenly through each year,
rather than read from `roll_calendars_csv`, because that file is unusable:
SOFR's covers only 2020-03 to 2021-09 with rolls as little as 12 days apart on
contracts expiring years later - the fingerprint of a data back-fill rather than
a trading record - and every instrument's calendar stops two to three years
before the prices do. The price data independently confirms the configured
frequency: contract changes in multiple_prices run at 1.00, 4.18, 3.98 and 4.00
per year against a configured 1, 4, 4, 4.

ASSUMPTIONS, ALL OF WHICH ERR IN KNOWN DIRECTIONS

  * Costs are present-day costs applied to forty years of history. Trading in
    the 1970s and 1980s cost considerably more than today, so early-period net
    results are optimistic, increasingly so the further back you look.

  * Cost is strictly proportional to contracts traded, which makes net P&L
    linear in capital. A flat per-order commission would break that, since it is
    charged once regardless of size. All four instruments have
    commission_per_trade = 0, and `account` refuses a non-zero one at rule level
    rather than silently distorting the comparison between rules.

  * Positions are not rounded here. Rounding is a step function and would also
    break linearity; it belongs at the levels where orders are real.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from refactory_2026 import config
from refactory_2026.accounting.timing import held_and_traded


@dataclass(frozen=True)
class Contract:
    """What accounting needs to know about an instrument, and nothing more.

    Deliberately not the config row: accounting declares the facts it depends
    on, so renaming a column in `instruments.py` cannot silently change what is
    charged, and a test can build one of these without any files on disk.
    """

    currency_per_point: float
    spread_cost_points: float
    commission_per_block: float
    commission_percentage: float
    commission_per_trade: float
    rolls_per_year: int


def account(
    decision: pd.Series,
    price: pd.Series,
    contract: Contract,
    contract_price: pd.Series | None = None,
    trade_lag: int = config.TRADE_LAG_DAYS,
) -> pd.DataFrame:
    """Gross, costs and net for a position, in dollars per day.

    `decision` is the position chosen at each day's close; the trade lag is
    applied here, so a caller has no way to express a different convention.

    `price` is the adjusted series, whose differences are the P&L. It is not a
    price anyone could trade at, so `contract_price` - the current contract's
    real price - is required if any percentage commission applies.
    """
    held, traded = held_and_traded(decision, trade_lag)

    change = price.diff().reindex(decision.index)
    gross = held * change * contract.currency_per_point

    costs = _trade_costs(
        traded.diff(), contract, _commission_price(contract, price, contract_price)
    ) + _roll_costs(held, contract, _commission_price(contract, price, contract_price))

    accounts = pd.DataFrame({"gross": gross, "costs": costs.fillna(0.0)})
    accounts["net"] = accounts["gross"] - accounts["costs"]

    return accounts.dropna(subset=["gross"])


def _commission_price(
    contract: Contract, price: pd.Series, contract_price: pd.Series | None
) -> pd.Series:
    """The price a percentage commission is charged against.

    The adjusted series goes negative, so it cannot serve: a percentage of -36.4
    is not a cost. Only a real traded price will do, and it is only needed when a
    percentage commission exists.
    """
    if contract.commission_percentage == 0:
        return price

    if contract_price is None:
        raise ValueError(
            "A percentage commission needs the current contract's price: the "
            "adjusted series is not a price anyone could trade at."
        )

    return contract_price.reindex(price.index).ffill()


def _trade_costs(
    trades: pd.Series, contract: Contract, price: pd.Series
) -> pd.Series:
    """Dollars paid to trade, charged on every contract that changes hands."""
    if contract.commission_per_trade != 0:
        raise ValueError(
            "A flat per-order commission breaks the assumption that cost is "
            "proportional to size, which rule-level comparisons rely on."
        )

    contracts = trades.abs()

    spread = contracts * contract.spread_cost_points * contract.currency_per_point
    per_block = contracts * contract.commission_per_block
    percentage = (
        contracts * price * contract.currency_per_point * contract.commission_percentage
    )

    # The largest of the commission structures applies, never their sum.
    commission = pd.concat([per_block, percentage], axis=1).max(axis=1)

    return spread + commission


def _roll_costs(held: pd.Series, contract: Contract, price: pd.Series) -> pd.Series:
    """Dollars paid to roll: the whole position, closed and reopened.

    Charged against the position actually held on the roll date, so a position
    that happens to be flat pays nothing. Spreading the annual cost evenly
    across every day would charge for rolling a position you do not hold.
    """
    rolls = pd.Series(0.0, index=held.index)

    for date in _roll_dates(held.index, contract.rolls_per_year):
        position = held.get(date, np.nan)
        if not pd.isna(position):
            rolls.loc[date] = 2 * abs(position)

    return _trade_costs(rolls, contract, price)


def _roll_dates(index: pd.DatetimeIndex, rolls_per_year: int) -> pd.DatetimeIndex:
    """Roll dates, spaced evenly through each year and snapped to trading days."""
    if rolls_per_year <= 0 or len(index) == 0:
        return pd.DatetimeIndex([])

    days_between = 365 / rolls_per_year
    starts = [
        pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=days_between / 2)
        for year in range(index[0].year, index[-1].year + 1)
    ]
    wanted = pd.DatetimeIndex(
        [start + pd.Timedelta(days=days_between * roll)
         for start in starts
         for roll in range(rolls_per_year)]
    ).sort_values()

    # The first trading day on or after each notional roll date.
    positions = index.searchsorted(wanted)

    return index[np.unique(positions[positions < len(index)])]


def contract_for(code: str) -> Contract:
    """Build a `Contract` from the instrument configuration."""
    from refactory_2026.instruments import load_instrument_config

    details = load_instrument_config([code]).loc[code]

    return Contract(
        currency_per_point=float(details.currency_per_point),
        spread_cost_points=float(details.spread_cost_points),
        commission_per_block=float(details.commission_per_block),
        commission_percentage=float(details.commission_percentage),
        commission_per_trade=float(details.commission_per_trade),
        rolls_per_year=int(details.rolls_per_year),
    )


def account_for(
    code: str, decision: pd.Series, trade_lag: int = config.TRADE_LAG_DAYS
) -> pd.DataFrame:
    """`account`, with the instrument's prices and configuration looked up."""
    from refactory_2026.data_cleaning.daily_prices import (
        adjusted_price,
        current_contract_price,
    )

    contract = contract_for(code)
    needs_real_price = contract.commission_percentage != 0

    return account(
        decision=decision,
        price=adjusted_price(code),
        contract=contract,
        contract_price=current_contract_price(code) if needs_real_price else None,
        trade_lag=trade_lag,
    )
