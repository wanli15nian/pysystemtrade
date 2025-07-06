import numpy as np
import pandas as pd
from copy import copy

from refactory.data_util import get_daily_price
from refactory.forecast import ewmac, rescale_forecast, floor_vol, price_vol


def instrument_forecast_turnover(instrument_code, rule_name):
    forecast_raw = get_capped_forecast(instrument_code, rule_name)
    forecast = forecast_raw.resample("1B").last()

    forecast_scalling = 10.0
    forecast_scalling_daily = pd.Series(np.full(forecast.shape[0], forecast_scalling), forecast.index)
    forecast_normalised = forecast / forecast_scalling_daily.ffill()

    avg_daily = float(forecast_normalised.diff().abs().mean())
    annual_turnover_for_forecast = avg_daily * 256

    print('forecast_turnover_for_individual_instrument')
    return annual_turnover_for_forecast


def get_capped_forecast(instrument, rule_name):
    '''
    Forecast 不是对当天价格的预判
    Forecast 根据包括当天在内的价格数据，对未来趋势进行判断
    究竟趋势如何就根据过去几天的价格变化
    '''
    price = get_daily_price(instrument)
    if rule_name == 'ewmac32':
        raw_ewmac32 = ewmac(price, 32, 128, 1)
        ewmac32 = rescale_forecast(raw_ewmac32 / floor_vol(price_vol(price)))
        ewmac32.rename('ewmac32', inplace=True)
        return ewmac32
    if rule_name == 'ewmac8':
        raw_ewmac8 = ewmac(price, 8, 32, 1)
        ewmac8 = rescale_forecast(raw_ewmac8 / floor_vol(price_vol(price)))
        ewmac8.rename('ewmac8', inplace=True)
        return ewmac8
    else:
        raise 'Rule not defined '


def calculate_weighted_average_with_nans(weights, list_of_values, sum_of_weights_should_be=1.0):
    ## easier to work in np space
    np_weights = np.array(weights)
    np_values = np.array(list_of_values)

    # get safe weights
    weights_times_values_as_np = np_weights * np_values
    empty_weights = np.isnan(weights_times_values_as_np)
    np_weights[empty_weights] = 0.0
    weights_without_nan = copy(np_weights)

    sum_of_values = np.nansum(weights_without_nan)
    renormalise_multiplier = sum_of_weights_should_be / sum_of_values
    normalised_weights = weights_without_nan * renormalise_multiplier

    weights_times_values_as_np = normalised_weights * np_values
    weighted_value = np.nansum(weights_times_values_as_np)

    return weighted_value


def calc_average_turnover(pooled_instruments, forecast_length_weights, rule_name):
    # 获取交易所有instrument年交易频率
    turnovers = [instrument_forecast_turnover(instrument_code, rule_name)
                 for instrument_code in pooled_instruments]
    weighted_avg_turnover = calculate_weighted_average_with_nans(forecast_length_weights, turnovers)
    return weighted_avg_turnover
