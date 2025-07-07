import numpy as np
import pandas as pd

from refactory.utils import calc_mixed_volatility


def annual_forecast_turnover(forecast_raw, forecast_scalling=10.0):
    # TODO 为什么要先降到日频，日内不调仓吗？
    # 其实turnover应该是和position相关的，只是系统假设position和forecast成绝对正比
    forecast = forecast_raw.resample("1B").last()
    proportion = forecast / forecast_scalling
    turnover_daily = float(proportion.diff().abs().mean())
    turnover_annual = turnover_daily * 256
    return turnover_annual


def calc_turnover_weights(forecast_all):
    # 用历史数据的多少来决定每个instrument的权重
    forecast_length = forecast_all.groupby('instrument').apply(len).to_list()
    total_length = float(sum(forecast_length))
    weights = [l / total_length for l in forecast_length]
    return weights


def calc_weighted_turnover(weights, turnovers, total=1.0):
    t = np.array(turnovers)
    w = np.array(weights)
    w[np.isnan(w * t)] = 0.0  # 应该是考虑到万一有的turnover没有的情况，对应也就不给weight
    w1 = w * total / np.nansum(w)
    return np.nansum(w1 * t)


# -------------------------------------------------------------------------


def calc_annual_cost(turnover, cost_per_trade, rolls_per_year):
    transaction_cost = turnover * cost_per_trade
    holding_turnovers = rolls_per_year * 2.0
    holding_cost = holding_turnovers * cost_per_trade
    annual_cost = transaction_cost + holding_cost
    return annual_cost


def get_cost_per_trade(price, per_block, per_trade, percentage, price_slippage, point_size, notional_blocks_traded):
    # 单次交易成本，包括slippage和commission
    # TODO: 在这里作者使用了pd.DateOffset来进行年份计算，而在rolling window中是用365天，原因存疑
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
