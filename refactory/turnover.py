import numpy as np


def estimate_turnover_all(forecast_all):
    turnover_func = lambda x: x.reset_index(level='instrument', drop=True).apply(calc_annual_turnover)
    turnover_all = forecast_all.groupby(level='instrument').apply(turnover_func)
    return turnover_all


def calc_annual_turnover(forecast, forecast_scaling=10.0):
    # 其实turnover应该是和position相关的，只是系统假设position和forecast成绝对正比
    # Turnover 在这里的定义是risk 的 turnover, forecast 本质上是risk 的 代表
    # FIXME:如果forecast是分钟频率的，turnover会少算很多
    # FIXME: 所以为什么不直接用Position 来算turnover
    forecast = forecast.resample("1B").last()
    proportion = forecast / forecast_scaling   # Forecast 的平均值在10，除以10 后得出normalised risk exposure, i.e. 1 unit of risk is 160k
    turnover_daily = proportion.diff().abs().mean()  # Average change in risk exposure, 而turnover的定义也是change in risk exposure
    turnover_annual = turnover_daily * 256
    return turnover_annual


def estimate_weighted_turnover(turnover_all, forecast_all):
    forecast_length = forecast_all.groupby('instrument').apply(len).to_list()  # 用历史数据的多少来决定每个instrument的权重
    turnover_weight = [l / sum(forecast_length) for l in forecast_length]
    weighted_turnover = turnover_all.apply(lambda x: calc_weighted_turnover(turnover_weight, x))
    return weighted_turnover


def calc_weighted_turnover(weights, turnovers, total=1.0):
    t = np.array(turnovers)
    w = np.array(weights)
    w[np.isnan(w * t)] = 0.0  # 应该是考虑到万一有的turnover没有的情况，对应也就不给weight
    w1 = w * total / np.nansum(w)
    return np.nansum(w1 * t)

# def calc_turnover(forecast, vol_scalar, smooth_days: int = 250) -> float:
#     position = calc_raw_position(forecast, vol_scalar)
#     position_daily = position.resample("1B").last()
#     if isinstance(vol_scalar, float) or isinstance(vol_scalar, int):
#         scalar_daily = pd.Series(np.full(position_daily.shape[0], float(vol_scalar)), position_daily.index)
#     else:
#         scalar_daily = vol_scalar.reindex(position_daily.index, method="ffill")
#         scalar_daily = scalar_daily.ewm(smooth_days, min_periods=2).mean()
#     position_normalised = position_daily / scalar_daily.ffill()
#     turnover_daily = position_normalised.diff().abs().mean()
#     turnover_yearly = turnover_daily * 256
#     return turnover_yearly
