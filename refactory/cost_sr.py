import numpy as np
import pandas as pd

from refactory.utils import calc_mixed_volatility


def calc_cost_sr_rules(average_turnover_, weighted_turnover_, gross, forecast, price, position_target, info):
    # 可以考虑把这个直接提到最外层
    rules = gross.columns.to_list()
    cost_SR_dict = {rule: calc_cost_sr(
        average_turnover_[rule],
        weighted_turnover_[rule],
        forecast[rule],
        gross[rule],
        price,
        position_target,
        info
    ) for rule in rules}
    return cost_SR_dict


def calc_cost_sr(average_turnover, weighted_turnover, forecast, pnl, price, position_target, info):
    rolls_per_year = int(info['rolls_per_year'])
    point_size = info['point_size']
    # 总成本 = 交易成本 + 移仓换月成本，都是以SR计算的，
    cost_sr_per_trade = calc_per_trade(price, info)
    transaction_cost = weighted_turnover * cost_sr_per_trade
    holding_cost = rolls_per_year * 2.0 * cost_sr_per_trade
    cost_sr = transaction_cost + holding_cost
    # 年波动
    ann_vol = calc_mixed_volatility(price.diff(), slow_vol_years=10) * 16
    # TODO: 向后填充，有用未来数据的可能
    # 年成本
    ann_cost = (-cost_sr * ann_vol * position_target).bfill()
    # 每条记录的实际成本 = 每条记录的时间间隔（以年为单位）* 年成本 * point size
    interval_as_year = ann_cost.index.to_series().diff().dt.total_seconds() / (365.25 * 24 * 60 * 60)
    cost_daily = ann_cost * interval_as_year * point_size
    # cost_daily.iloc[:11] = np.nan  # QUESTION: 为什么前11个数都是Nan
    # if instrument == 'US10':
    #     cost_daily.iloc[:13] = np.nan  # QUESTION: 为什么到了US10是前13个数字
    mean_cost_daily = cost_daily.mean()
    # 年化夏普成本
    pnl_vol_daily = pnl.std()
    cost_sr_annual = 16 * mean_cost_daily / pnl_vol_daily
    # 计算平均夏普成本
    # TODO:这里应该直接传入turnover，不用传forecast，导致语义不清楚
    annual_turnover = calc_annual_turnover(forecast)
    cost_sr = cost_sr_annual * (average_turnover / annual_turnover) * 2
    return cost_sr


def calc_per_trade(price, info, notional_blocks_traded=1):
    point_size = info['point_size']
    spread_cost = info['spread_cost']
    per_trade = info['per_trade']
    per_block = info['per_block']
    percentage = info['percentage']
    # A股股票必须是100股的整数倍，这个参数是这个100的意思吗？
    blocks = notional_blocks_traded
    # 过去一年的均价
    average_price = float(price[price.index[-1] - pd.DateOffset(years=1):].mean())
    # 交易佣金，三种方式只会有一种，其他两种为零，可以用取最大值的方法
    commission_percentage = blocks * average_price * point_size * percentage
    commission_per_block = blocks * per_block
    commission = max([per_trade, commission_per_block, commission_percentage])
    # 交易滑点，这个应该可以加上参数控制滑几个点
    slippage = blocks * spread_cost * point_size
    # 单次交易成本，包括slippage和commission
    cost = commission + slippage
    # 年化波动率
    # TODO:前面乘了notional_blocks_traded，这里不乘吗？
    ann_vol = calc_ann_vol(price, point_size)
    # 夏普成本
    cost_sr = cost / ann_vol
    return cost_sr


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
