import numpy as np
import pandas as pd

from refactory.utils import calc_mixed_volatility


def calc_subsystem_turnover(subsystem_position_raw, raw_price, price, point_size):
    average_position_for_turnover = calc_average_position(raw_price, price, point_size)
    subsystem_turnover = turnover_x_y(subsystem_position_raw, average_position_for_turnover)
    return subsystem_turnover


def calc_average_position(raw_price, price, block_move_value, notional_trading_capital=500000, risk_target=0.25,
                          vol_mult=1.0):
    # carry_data = get_instrument_raw_carry_data(instrument).PRICE
    # daily_prices = carry_data.resample('1B').last()
    # denom_price = get_instrument_raw_carry_data(instrument).PRICE
    # denom_price = denom_price.resample('1B').last()

    annual_cash_vol_target = (notional_trading_capital * risk_target)
    daily_cash_vol_target = annual_cash_vol_target / 16

    block_value = block_move_value * raw_price.ffill() * 0.01
    price_returns = price.diff()
    raw_vol = calc_mixed_volatility(price_returns, slow_vol_years=10)
    return_vol = vol_mult * raw_vol

    denom_price = raw_price
    (denom_price, return_vol) = denom_price.align(return_vol, join="right")
    daily_perc_vol1 = 100.0 * (return_vol / denom_price.ffill().abs())

    (block_value, daily_perc_vol) = block_value.align(daily_perc_vol1, join="inner")
    instr_ccy_vol = block_value.ffill() * daily_perc_vol
    instr_value_vol = instr_ccy_vol.ffill()

    average_position_for_turnover = daily_cash_vol_target / instr_value_vol

    return average_position_for_turnover


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
