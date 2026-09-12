"""Trading rules, and the variations of them this build trades.

A rule is pure maths: prices and volatility in, a raw signal out. It knows
nothing about scaling, capping, capital or its own name, so it can be tested
against a hand-built series with no files involved.

Volatility is passed in, never fetched. A rule that fetched its own volatility
could disagree with the volatility used to size positions, which is the mistake
behind finding C3 in the old build.

NAMES ARE DERIVED, NEVER TYPED

A variation's name comes from its function and parameters: `ewmac_8_32`. Nothing
is hand-written, so a name cannot disagree with the parameters it labels, and
adding a variation means adding one tuple below rather than inventing a label.

The name matters beyond display: it becomes a column, and later the key that a
forecast weight attaches to. A hand-edited name would silently break the link
between a signal and its weight.

`ewmac_8_32` states both spans. pysystemtrade writes `ewmac8` and leaves the
slow span to an unstated convention that it is always four times the fast one,
which becomes a lie the moment you try (8, 64).
"""

from __future__ import annotations

from typing import Callable

import pandas as pd


def ewmac(price: pd.Series, vol: pd.Series, fast: int, slow: int) -> pd.Series:
    """Fast moving average minus slow, divided by volatility.

    Positive when the price is above its own recent trend. Dividing by
    volatility puts every instrument on the same scale; without it the signal is
    in the instrument's own price units and cannot be compared or combined.
    """
    fast_average = price.ewm(span=fast, min_periods=1).mean()
    slow_average = price.ewm(span=slow, min_periods=1).mean()

    return (fast_average - slow_average) / vol


class Rule:
    """One variation of a trading rule: a function and the parameters it runs with."""

    def __init__(self, function: Callable, **params):
        self.function = function
        self.params = params

    @property
    def name(self) -> str:
        """Derived from the function and its parameters, e.g. 'ewmac_8_32'."""
        values = "_".join(str(value) for value in self.params.values())

        return f"{self.function.__name__}_{values}"

    def raw_forecast(self, price: pd.Series, vol: pd.Series) -> pd.Series:
        return self.function(price, vol, **self.params)

    def __repr__(self) -> str:
        return f"Rule({self.name})"


# The variations traded. Add a pair to trade another; the name follows.
EWMAC_SPANS = [(8, 32), (32, 128)]

DEFAULT_RULES = [Rule(ewmac, fast=fast, slow=slow) for fast, slow in EWMAC_SPANS]
