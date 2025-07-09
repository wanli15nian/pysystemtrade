import numpy as np
import pandas as pd

from refactory.utils import calc_mixed_volatility


def calc_subsystem_turnover(subsystem_position_raw, raw_price, price, point_size):
    average_position_for_turnover = calc_average_position(raw_price, price, point_size)
    subsystem_turnover = turnover_x_y(subsystem_position_raw, average_position_for_turnover)
    return subsystem_turnover


def calc_average_position(raw_price, price, block_move_value, notional_trading_capital=500000, risk_target=0.25,
                          vol_mult=1.0):
    raw_price, price = raw_price.align(price, join="inner")
    raw_price.ffill(inplace=True)
    price.ffill(inplace=True)

    pnl_vol = vol_mult * calc_mixed_volatility(price.diff(), slow_vol_years=10)
    vol_percent = 100.0 * (pnl_vol / raw_price.abs())
    block_value = block_move_value * raw_price * 0.01
    cash_vol = block_value * vol_percent
    daily_cash_vol_target = (notional_trading_capital * risk_target) / 16
    average_position = daily_cash_vol_target / cash_vol

    return average_position


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
