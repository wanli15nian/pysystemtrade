from copy import copy

import numpy as np
import pandas as pd

from refactory.core import optimisation


def calc_portfolio_weights(net_, positions, target_sr=0.5):
    # FIXME: 应该是按照一个周期(如按周)滚动计算weights，作为未来一个周期的weights
    weights = calc_instrument_weights(net_, target_sr)
    position_weights = calc_position_weights(positions, weights)
    return position_weights


def calc_instrument_weights(net_, target_sr=0.5):
    net_weekly = net_.resample('W').sum()
    std_annual = net_weekly.ewm(span=50000, min_periods=5).std().iloc[-1] * (365.25 / 7.0) ** 0.5
    std_mean = std_annual.mean()
    norm_std = [std_mean] * len(std_annual)
    norm_mean = [target_sr * std_mean] * len(std_annual)

    corr_matrix = calc_corr_matrix(net_weekly).values
    shrunk_corr = shrink_corr_matrix(corr_matrix)
    weights = optimisation(corr=shrunk_corr, norm_mean=norm_mean, norm_stdev=norm_std)
    return weights


def calc_corr_matrix(net, span=500000, min_periods=10):
    corr = net.ewm(span=span, min_periods=min_periods, ignore_na=True).corr(pairwise=True)
    corr_matrix_df = (corr[corr.index.get_level_values(0) < net.index[-1]]
                      .tail(len(corr.columns))
                      .droplevel(0)
                      .clip(lower=0))  # 所有小于0的值设为0
    return corr_matrix_df


def shrink_corr_matrix(corr_matrix, shrunk_rate=0.5):
    new_corr_values = copy(corr_matrix)
    np.fill_diagonal(new_corr_values, np.nan)
    avg_corr = np.nanmean(new_corr_values)

    avg_matrix = np.full(corr_matrix.shape, avg_corr)
    np.fill_diagonal(avg_matrix, 1.0)

    shrunk_corr = shrunk_rate * avg_matrix + (1 - shrunk_rate) * corr_matrix
    return shrunk_corr


def calc_position_weights(positions, weights):
    # 有些instrument在一段时间不能交易，需要把权重调成0，再平滑，再做归一化
    positions[(~positions.isna()).sum(axis=1) == 0] = 0  # 这是什么意思？
    weights_raw = (pd.DataFrame([weights], columns=positions.columns.to_list(), index=[positions.index[0]])
                   .reindex(positions.index, method="ffill"))
    weights_raw[np.isnan(positions)] = 0.0  # 把不能交易的instrument的权重设为0
    weights_daily_raw = weights_raw.resample('1B').mean()  # 意味着position可以是分钟频率的
    weights_daily = weights_daily_raw.ewm(span=125).mean()  # 平滑，防止权重突变
    sum_weights = weights_daily.sum(axis=1).replace(0.0, 0.0001)
    weights_normalised = weights_daily.div(sum_weights, axis=0)
    return weights_normalised
