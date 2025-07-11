import numpy as np
import pandas as pd

from refactory.base import calc_volatility_scalar


def calc_subsystem_turnover(subsystem_position_raw, price, point_size):
    average_position_for_turnover = calc_volatility_scalar(price, point_size)
    subsystem_turnover = turnover_x_y(subsystem_position_raw, average_position_for_turnover)
    return subsystem_turnover


def turnover_x_y(x, y, smooth_y_days: int = 250) -> float:
    '''
    Give the turnover of x normalised for y
    '''

    daily_x = x.resample("1B").last()
    if isinstance(y, float) or isinstance(y, int):
        daily_y = pd.Series(np.full(daily_x.shape[0], float(y)), daily_x.index)
    else:
        daily_y = y.reindex(daily_x.index, method="ffill")
        daily_y = daily_y.ewm(smooth_y_days, min_periods=2).mean()

    x_normalised_for_y = daily_x / daily_y.ffill()
    avg_daily = float(x_normalised_for_y.diff().abs().mean())
    return avg_daily * 256
