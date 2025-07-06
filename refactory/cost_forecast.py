import numpy as np
import pandas as pd
from copy import copy

from refactory.data_util import get_daily_price
from refactory.forecast import ewmac, rescale_forecast, floor_vol, price_vol
from refactory.utils import calc_mixed_volatility


def calc_annual_cost(turnover, cost_per_trade, rolls_per_year):
    transaction_cost = turnover * cost_per_trade
    holding_turnovers = rolls_per_year * 2.0
    holding_cost = holding_turnovers * cost_per_trade
    annual_cost = transaction_cost + holding_cost
    return annual_cost


def calculate_weighted_turnover(weights, list_of_values, sum_of_weights_should_be=1.0):
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


def get_cost_per_trade(price, per_block, per_trade, percentage, price_slippage, point_size, notional_blocks_traded):
    # 单次交易成本，包括slippage和commission
    # FIXME: 在这里作者使用了pd.DateOffset来进行年份计算，而在rolling window中是用365天，原因存疑
    average_price = float(price[price.index[-1] - pd.DateOffset(years=1):].mean())
    commission_percentage = notional_blocks_traded * average_price * point_size * percentage
    commission_per_block = notional_blocks_traded * per_block
    commission = max([per_trade, commission_per_block, commission_percentage])
    slippage = notional_blocks_traded * price_slippage * point_size
    cost = commission + slippage
    vol_daily = calc_mixed_volatility(price.diff(), slow_vol_years=10)
    vol_daily_average = float(vol_daily[price.index[-1] - pd.DateOffset(years=1):].mean())
    ann_std = vol_daily_average * 16 * point_size
    cost_per_trade = cost / ann_std
    return cost_per_trade


def annual_forecast_turnover(forecast_raw):
    forecast = forecast_raw.resample("1B").last()
    # TODO:改为直接除以forecast_scalling
    forecast_scalling = 10.0
    forecast_scalling_daily = pd.Series(np.full(forecast.shape[0], forecast_scalling), forecast.index)
    forecast_normalised = forecast / forecast_scalling_daily.ffill()
    turnover_daily = float(forecast_normalised.diff().abs().mean())
    turnover_annual = turnover_daily * 256
    return turnover_annual


def calc_turnover_weights(forecast_all):
    # 用历史数据的多少来决定每个instrument的权重
    # forecast_length = [len(v) for k, v in forecast_all.items()]
    forecast_length = forecast_all.groupby('instrument').apply(len).to_list()
    total_length = float(sum(forecast_length))
    weights = [l / total_length for l in forecast_length]
    return weights


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
