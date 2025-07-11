from copy import copy

import numpy as np
import pandas as pd

from refactory.utils import optimisation


# TODO: stack这种傻办法需要改成直接用multiIndex做
def calc_portfolio_weights(net_return_raw, positions, target_sr=0.5):
    net_weekly = net_return_raw.resample('W').sum()

    std_annual = net_weekly.ewm(span=50000, min_periods=5).std().iloc[-1] * (365.25 / 7.0) ** 0.5
    std_mean = std_annual.mean()
    norm_std = [std_mean] * len(std_annual)
    norm_mean = [target_sr * std_mean] * len(std_annual)

    corr_matrix = calc_corr_matrix(net_weekly).values
    shrunk_corr = shrink_corr_matrix(corr_matrix)

    weights = optimisation(corr=shrunk_corr, norm_mean=norm_mean, norm_stdev=norm_std)

    instruments = net_weekly.columns.to_list()
    start = net_weekly.index[0]
    weights_df = pd.DataFrame({asset_name: weight for (asset_name, weight) in zip(instruments, weights)},
                              index=[start])

    positions[(~positions.isna()).sum(axis=1) == 0] = 0
    smoothed_instr_weights = calc_smoothed_instr_weights(weights_df, positions)
    normalised_weights = normalise_weights(smoothed_instr_weights)

    return normalised_weights


def calc_smoothed_instr_weights(weights_df, subsystem_positions, smooth_weighting=125):
    instrument_weights = weights_df.reindex(subsystem_positions.index, method="ffill")
    instrument_weights[np.isnan(subsystem_positions)] = 0.0
    daily_unsmoothed_instr_weights = instrument_weights.resample('1B').mean()
    return daily_unsmoothed_instr_weights.ewm(span=smooth_weighting).mean()


def normalise_weights(smoothed_instr_weights):
    sum_weights = smoothed_instr_weights.sum(axis=1)
    zero_rows = sum_weights == 0.0
    sum_weights[zero_rows] = 0.0001  ## avoid Inf
    weight_multiplier = 1.0 / sum_weights
    weight_multiplier_array = np.array([weight_multiplier] * len(smoothed_instr_weights.columns))
    normalised_weights_np = weight_multiplier_array.transpose() * smoothed_instr_weights.values
    normalised_weights = pd.DataFrame(normalised_weights_np, columns=smoothed_instr_weights.columns,
                                      index=smoothed_instr_weights.index)
    return normalised_weights

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

