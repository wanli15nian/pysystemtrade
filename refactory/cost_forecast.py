import numpy as np


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
