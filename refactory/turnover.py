import numpy as np
import pandas as pd


def calc_turnover(position, vol_scalar, smooth_days: int = 250) -> float:
    position_daily = position.resample("1B").last()
    if isinstance(vol_scalar, float) or isinstance(vol_scalar, int):
        scalar_daily = pd.Series(np.full(position_daily.shape[0], float(vol_scalar)), position_daily.index)
    else:
        scalar_daily = vol_scalar.reindex(position_daily.index, method="ffill")
        scalar_daily = scalar_daily.ewm(smooth_days, min_periods=2).mean()
    position_normalised = position_daily / scalar_daily.ffill()
    turnover_daily = position_normalised.diff().abs().mean()
    turnover_yearly = turnover_daily * 256
    return turnover_yearly
