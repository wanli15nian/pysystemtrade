import numpy as np
import pandas as pd


def calc_turnover(position, vol_scalar, smooth_days: int = 250) -> float:
    position_daily = position.resample("1B").last()
    if isinstance(vol_scalar, float) or isinstance(vol_scalar, int):
        scalar_daily = pd.Series(np.full(position_daily.shape[0], float(vol_scalar)), position_daily.index)
    else:
        scalar_daily = vol_scalar.reindex(position_daily.index, method="ffill")
        scalar_daily = scalar_daily.ewm(smooth_days, min_periods=2).mean()
    position_normalised = position_daily / scalar_daily.ffill()
    turnover_daily = position_normalised.diff().abs().mean()
    turnover_yearly = turnover_daily * 256
    return turnover_yearly


def estimate_turnover(forecast_):
    turnover_func = lambda x: x.reset_index(level='instrument', drop=True).apply(calc_annual_turnover)
    turnover_ = forecast_.groupby(level='instrument').apply(turnover_func)
    # average_turnover_ = turnover_.apply(np.nanmean)
    turnover_weight = calc_turnover_weights(forecast_)
    weighted_turnover_ = turnover_.apply(lambda x: calc_weighted_turnover(turnover_weight, x))
    return weighted_turnover_


def calc_annual_turnover(forecast_raw, forecast_scalling=10.0):
    # 其实turnover应该是和position相关的，只是系统假设position和forecast成绝对正比
    forecast = forecast_raw.resample("1B").last()
    proportion = forecast / forecast_scalling
    turnover_daily = proportion.diff().abs().mean()
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
