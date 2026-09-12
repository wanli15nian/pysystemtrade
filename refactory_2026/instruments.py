"""Static per-instrument configuration.

The numbers a backtest needs for each instrument are spread across three files
in data/csvconfig, and one of them is not stored at all but derived:

    point_size, currency, commission_per_block,             <- instrumentconfig.csv
    commission_percentage, commission_per_trade
    spread_cost                                             <- spreadcosts.csv
    rolls_per_year                                          <- len(HoldRollCycle)
                                                               from rollconfig.csv

This module does that join once, validates the result, and returns a DataFrame
indexed by instrument code. Nothing else reads csvconfig.

Columns returned:
    point_size             a price difference of 1.0 is worth this much currency
    currency               currency the contract settles in
    commission_per_block   commission per contract traded
    commission_percentage  commission as a fraction of traded value, not a percent
    commission_per_trade   flat commission per order, whatever its size
    spread_cost            half-spread, in price points
    rolls_per_year         hold-cycle rolls per year

The three commission columns are alternative charging structures, not charges to
be added up: a broker's commission for a fill is the largest of the three.

Rows come back in the order requested, never sorted, so that downstream code
cannot come to depend on alphabetical ordering.
"""

from __future__ import annotations

import pandas as pd

from refactory_2026 import config

CONFIG_DIR = config.DATA_DIR / "csvconfig"

NUMERIC_COLUMNS = [
    "point_size",
    "commission_per_block",
    "commission_percentage",
    "commission_per_trade",
    "spread_cost",
    "rolls_per_year",
]

# Everything is priced in this currency until an instrument needs converting.
BASE_CURRENCY = "USD"


def load_instrument_config(codes: list[str] | None = None) -> pd.DataFrame:
    """Return static configuration for `codes`, defaulting to config.INSTRUMENTS."""
    if codes is None:
        codes = config.INSTRUMENTS

    joined = pd.concat(
        [_instrument_columns(), _spread_cost(), _rolls_per_year()], axis=1
    )

    return _validated(_selected(joined, codes))


def _instrument_columns() -> pd.DataFrame:
    columns = {
        "Pointsize": "point_size",
        "Currency": "currency",
        "PerBlock": "commission_per_block",
        "Percentage": "commission_percentage",
        "PerTrade": "commission_per_trade",
    }
    return _read_config("instrumentconfig.csv")[list(columns)].rename(columns=columns)


def _spread_cost() -> pd.DataFrame:
    return _read_config("spreadcosts.csv")[["SpreadCost"]].rename(
        columns={"SpreadCost": "spread_cost"}
    )


def _rolls_per_year() -> pd.Series:
    """One roll per letter of the hold cycle: 'Z' is annual, 'HMUZ' quarterly."""
    hold_cycle = _read_config("rollconfig.csv")["HoldRollCycle"]
    return hold_cycle.str.len().rename("rolls_per_year")


def _read_config(filename: str) -> pd.DataFrame:
    table = pd.read_csv(CONFIG_DIR / filename, index_col="Instrument")
    table.index.name = "instrument"

    duplicated = table.index[table.index.duplicated()].unique().tolist()
    if duplicated:
        raise ValueError(f"{filename} lists these instruments twice: {duplicated}")

    return table


def _selected(joined: pd.DataFrame, codes: list[str]) -> pd.DataFrame:
    missing = [code for code in codes if code not in joined.index]
    if missing:
        raise ValueError(f"No configuration in {CONFIG_DIR} for: {missing}")

    return joined.loc[codes].copy()


def _validated(selected: pd.DataFrame) -> pd.DataFrame:
    """Fail at load time, with the instrument named, rather than mid-calculation."""
    for column in NUMERIC_COLUMNS:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")

    unusable = selected.index[selected[NUMERIC_COLUMNS].isna().any(axis=1)].tolist()
    if unusable:
        raise ValueError(f"Missing or non-numeric configuration for: {unusable}")

    selected["rolls_per_year"] = selected["rolls_per_year"].astype(int)

    foreign = selected.index[selected["currency"] != BASE_CURRENCY].tolist()
    if foreign:
        raise ValueError(
            f"{foreign} are not priced in {BASE_CURRENCY}, and currency conversion "
            "is not implemented. Drop them or add fx handling."
        )

    return selected
