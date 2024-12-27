from copy import copy

import numpy as np

from refactory.data_source import get_daily_price
from refactory.utils import ewmac, get_volatily


def get_capped_forecast(instrument_code, rule_name):
    '''
    Forecast 不是对当天价格的预判
    Forecast 根据包括当天在内的价格数据，对未来趋势进行判断
    究竟趋势如何就根据过去几天的价格变化
    '''
    price = get_daily_price(instrument_code)
    if rule_name == 'ewmac32':
        raw_ewmac32 = ewmac(price, 32, 128, 1)
        ewmac32 = final_forecast('ewmac32', raw_ewmac32, price, 20)
        return ewmac32
    if rule_name == 'ewmac8':
        raw_ewmac8 = ewmac(price, 8, 32, 1)
        ewmac8 = final_forecast('ewmac8', raw_ewmac8, price, 20)
        return ewmac8
    else:
        raise 'Rule not defined '


def final_forecast(name, raw_forecast, price, upper_cap=20):
    raw_forecast[raw_forecast == 0] = np.nan

    # TODO:为什么有的用价格波动率，有的用收益率波动率？
    vol = get_volatily(price)
    adjust_forecast = raw_forecast / vol

    scalar = get_forecast_scalar(adjust_forecast)
    scaled_forecast = scalar * adjust_forecast

    lower_cap = -upper_cap
    capped_forecast = scaled_forecast.clip(lower=lower_cap, upper=upper_cap)

    capped_forecast.rename(name, inplace=True)

    return capped_forecast


def get_forecast_scalar(raw_forecast, window=250000, min_period=500, target_abs_forecast=10, backfill=True):
    forecast = copy(raw_forecast)
    forecast = forecast.abs()
    ave_abs_value = forecast.rolling(window=window, min_periods=min_period).mean()
    scaling_factor = target_abs_forecast / ave_abs_value
    if backfill:
        scaling_factor = scaling_factor.bfill()
    return scaling_factor
