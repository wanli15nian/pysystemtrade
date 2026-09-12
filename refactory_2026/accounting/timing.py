"""The one place in this codebase that shifts anything.

THE PROBLEM THIS SOLVES

A forecast labelled Monday is computed from Monday's close, so it does not exist
until Monday has finished. A price change labelled Monday is Monday's move,
which happened before that forecast existed. Multiplying the two pairs a
position with the very move that caused it - you are paid for the day whose
information you used to decide. Measured across the eight instrument/rule pairs
here, that mistake turns a mean Sharpe of 0.31 into 1.16.

Nothing errors when you make it. Both series are indexed by date, pandas aligns
them happily, and the result looks plausible. You have to shift deliberately to
be correct, which is why the old build got it wrong in some places and right in
others, with no way to tell which.

THE TIMELINE

    close t              close t+1            close t+2
      |                    |                    |
      decide, using        trade here, at       this day's price change
      everything up        this close           is the first the new
      to this close                             position earns

With `trade_lag` days between deciding and trading:

    held   = decision.shift(trade_lag + 1)   pair with price change for P&L
    traded = decision.shift(trade_lag)       its changes are the trades, priced
                                             at that day's close

The two shifts have different reasons, which is why the code says
`trade_lag + 1` rather than a literal 2. One is the execution assumption, which
could change. The other is structural: a price change labelled t covers t-1 to
t, so a position earning it must have been in place by close t-1.

COSTS COME FROM THE SAME CALL

The old build priced trades on one day and booked the matching P&L on another,
and nothing revealed the inconsistency. Here both series come out of one
function, so they cannot disagree.
"""

from __future__ import annotations

import pandas as pd

from refactory_2026 import config


def held_and_traded(
    decision: pd.Series, trade_lag: int = config.TRADE_LAG_DAYS
) -> tuple[pd.Series, pd.Series]:
    """Split a series of decisions into what is held and what is traded.

    `decision` is the position chosen at each day's close. Returns the position
    held over each day, to be paired with that day's price change, and the
    position standing after each day's trading, whose changes are the trades.
    """
    if trade_lag < 0:
        raise ValueError(
            f"A negative trade lag ({trade_lag}) means trading before the signal "
            "exists. Use 0 to trade at the signal's own close."
        )

    return decision.shift(trade_lag + 1), decision.shift(trade_lag)
