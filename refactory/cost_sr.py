import numpy as np
import pandas as pd

from refactory.base import calc_fill_cost
from refactory.utils import calc_mixed_volatility


def calc_cost_sr(turnover_annual, average_turnover, weighted_turnover, pnl, price, position_target, info):
    # 计算年成本
    cost_sr_annual = get_cost_sr_annual(weighted_turnover, price, info)
    vol_annual = calc_mixed_volatility(price.diff(), slow_vol_years=10) * 16
    cost_annual = (-cost_sr_annual * vol_annual * position_target).bfill()  # TODO: 向后填充有用未来数据的可能
    # 计算日均成本
    point_size = info['point_size']
    interval_as_year = cost_annual.index.to_series().diff().dt.total_seconds() / (365.25 * 24 * 60 * 60)
    cost_daily = cost_annual * interval_as_year * point_size
    cost_daily_mean = cost_daily.mean()
    # 计算年夏普成本
    pnl_vol_daily = pnl.std()
    cost_sr_annual = 16 * (cost_daily_mean / pnl_vol_daily)
    # 计算平均年夏普成本
    cost_sr = cost_sr_annual * (average_turnover / turnover_annual) * 2
    return cost_sr


def get_cost_sr_annual(weighted_turnover, price, info):
    # 总成本 = 交易成本 + 移仓换月成本，都是以SR计算的，
    cost_sr_per = calc_cost_sr_per(price, info)
    rolls_per_year = int(info['rolls_per_year'])
    holding_cost = rolls_per_year * 2.0 * cost_sr_per
    transaction_cost = weighted_turnover * cost_sr_per
    cost_sr_annual = transaction_cost + holding_cost
    return cost_sr_annual


def calc_cost_sr_per(price, info, notional_blocks_traded=1):
    cost = calc_cost_per(price, info, notional_blocks_traded)
    point_size = info['point_size']
    ann_vol = calc_ann_vol(price, point_size)
    # TODO:前面乘了notional_blocks_traded，这里不乘吗？
    cost_sr = cost / ann_vol
    return cost_sr


def calc_cost_per(price, info, notional_blocks_traded):
    # A股股票必须是100股的整数倍，这个参数是这个100的意思吗？
    blocks = notional_blocks_traded
    # 过去一年的均价
    average_price = float(price[price.index[-1] - pd.DateOffset(years=1):].mean())
    return calc_fill_cost(average_price, blocks, info)


def calc_ann_vol(price, point_size):
    vol = calc_mixed_volatility(price.diff(), slow_vol_years=10)
    vol_average = float(vol[price.index[-1] - pd.DateOffset(years=1):].mean())
    ann_vol = vol_average * 16 * point_size
    return ann_vol


def calc_annual_turnover(forecast_raw, forecast_scalling=10.0):
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
