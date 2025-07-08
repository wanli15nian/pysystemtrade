import numpy as np
import pandas as pd
from scipy.optimize import minimize


def calc_mixed_volatility(data, days=35, min_periods=10, slow_vol_years=20,
                          perc_of_long_vol=0.3, vol_min=0.0000000001,
                          vol_multiplier=1.0):
    # 长期和短期波动进行权重处理
    short_vol = data.ewm(adjust=True, span=days, min_periods=min_periods).std()
    long_vol = short_vol.ewm(adjust=True, span=slow_vol_years * 256).mean()
    vol = perc_of_long_vol * long_vol + (1 - perc_of_long_vol) * short_vol
    vol[vol < vol_min] = vol_min
    vol = vol * vol_multiplier
    return vol


def optimisation(number, corr, norm_mean, norm_stdev):
    def addem(weights):
        return 1.0 - sum(weights)

    def neg_SR(weights, sigma, mus):
        estimated_returns = np.dot(weights, mus)[0]
        stdev = weights.dot(sigma).dot(weights.transpose()) ** 0.5
        sr = -estimated_returns / stdev
        return sr

    mus = np.array(norm_mean, ndmin=2).transpose()  # mus 没问题
    sigma = np.diag(norm_stdev).dot(corr).dot(np.diag(norm_stdev))
    start_weights = np.array([1 / number] * number)
    bounds = [(0.0, 1.0)] * number
    cdict = [{"type": "eq", "fun": addem}]
    ans = minimize(neg_SR, start_weights, (sigma, mus), method='SLSQP', constraints=cdict, bounds=bounds, tol=0.00001)
    weight = ans['x']
    return weight


def single_resampled_set_of_returns(data_dict, frequency: str):
    data_resampled = [pnl.resample(frequency).sum() for pnl in data_dict.values()]

    all_indices = [data_item.index for data_item in data_resampled]
    flattened = [item for sublist in all_indices for item in sublist]
    common_index = list(set(flattened))
    common_index.sort()

    reindexed_data = [data_item.reindex(common_index) for data_item in data_resampled]

    for offset_value, data_item in enumerate(reindexed_data):
        data_item.index = data_item.index + pd.Timedelta("%dus" % offset_value)

    stacked_data = pd.concat(reindexed_data, axis=0).sort_index()
    return stacked_data

# def robust_vol_calc(daily_returns: pd.Series,
#                     days: int = 35,
#                     min_periods: int = 10,
#                     vol_abs_min: float = 0.0000000001,
#                     vol_floor: bool = True,
#                     floor_min_quant: float = 0.05,
#                     floor_min_periods: int = 100,
#                     floor_days: int = 500,
#                     backfill: bool = False, ):
#     vol = daily_returns.ewm(adjust=True, span=days, min_periods=min_periods).std()
#     vol[vol < vol_abs_min] = vol_abs_min
#
#     if vol_floor:
#         vol_min = vol.rolling(min_periods=floor_min_periods, window=floor_days).quantile(q=floor_min_quant)
#         vol_min.iloc[0] = 0.0
#         vol_min.ffill(inplace=True)
#         vol = np.maximum(vol, vol_min)
#     if backfill:
#         # use the first vol in the past, sort of cheating
#         vol_forward_fill = vol.ffill()
#         vol = vol_forward_fill.bfill()
#     return vol
