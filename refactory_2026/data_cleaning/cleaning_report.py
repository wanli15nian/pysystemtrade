"""What cleaning discarded, reported rather than hidden.

Cleaning makes three judgements that a future file could prove wrong: where the
trading day ends, that weekend-dated sessions can go, and that a weekday with no
prints should be absent rather than NaN. Each throws data away, so the counts
are made visible.

Every figure here is derived from `daily_prices`, never recomputed, so this
cannot drift away from what the pipeline actually does.

Nothing in the pipeline imports this. It exists to be read:

    python -m refactory_2026.data_cleaning.cleaning_report
"""

from __future__ import annotations

import pandas as pd

from refactory_2026 import config
from refactory_2026.data_cleaning.daily_prices import (
    SESSION_CLOSE_HOUR,
    closing_stamps,
    daily_close,
    raw_prints,
    trading_day_closes,
)


def cleaning_report(codes: list[str] | None = None) -> pd.DataFrame:
    """One row per instrument: what cleaning produced, and what it dropped."""
    if codes is None:
        codes = config.INSTRUMENTS

    return pd.DataFrame([_summary(code) for code in codes], index=codes)


def gaps_by_year(code: str) -> pd.Series:
    """Weekdays inside an instrument's history carrying no price at all.

    A market holiday and a hole in the feed are indistinguishable here, which is
    why these are reported rather than represented as missing rows.
    """
    traded = daily_close(raw_prints(code)).index
    weekdays = pd.bdate_range(traded[0], traded[-1])

    return pd.Series(weekdays.difference(traded)).dt.year.value_counts().sort_index()


def _summary(code: str) -> dict:
    prints = raw_prints(code)
    all_sessions = trading_day_closes(prints)
    kept = daily_close(prints)

    closed_at_boundary = closing_stamps(prints)[kept.index] == SESSION_CLOSE_HOUR
    weekdays = pd.bdate_range(kept.index[0], kept.index[-1])

    return {
        "sessions": len(kept),
        "first": kept.index[0].date(),
        "last": kept.index[-1].date(),
        "closed_at_2300": round(100 * closed_at_boundary.mean(), 1),
        "weekend_sessions_dropped": len(all_sessions) - len(kept),
        "blank_prints_dropped": int(prints.isna().sum()),
        "weekday_gaps": len(weekdays.difference(kept.index)),
    }


if __name__ == "__main__":
    print(cleaning_report().to_string())
    print()
    for instrument in config.INSTRUMENTS:
        print(f"{instrument} weekday gaps by year:")
        print(gaps_by_year(instrument).to_string())
        print()
