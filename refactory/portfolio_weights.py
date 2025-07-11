from copy import copy

import numpy as np
import pandas as pd

from refactory.utils import optimisation


# TODO: stack这种傻办法需要改成直接用multiIndex做
def calc_portfolio_weights(net_return_raw, positions, target_sr=0.5):
    net_weekly = net_return_raw.resample('W').sum()

    net_std_annual = net_weekly.ewm(span=50000, min_periods=5).std().iloc[-1] * (365.25 / 7.0) ** 0.5
    norm_std = [net_std_annual.mean()] * len(net_std_annual)
    mean_list = [target_sr * s for s in norm_std]
    # std_mean = net_std_annual.mean()
    # norm_std = [std_mean] * len(net_std_annual)
    # mean_list = [target_sr * std_mean] * len(net_std_annual)

    shrunk_corr_values = shrink_corr_matrix(net_weekly)

    weights = optimisation(corr=shrunk_corr_values, norm_mean=mean_list, norm_stdev=norm_std)

    instruments = net_weekly.columns.to_list()
    start = net_weekly.index[0]
    weights_df = pd.DataFrame({asset_name: weight for (asset_name, weight) in zip(instruments, weights)},
                              index=[start])

    positions[(~positions.isna()).sum(axis=1) == 0] = 0
    smoothed_instr_weights = calc_smoothed_instr_weights(weights_df, positions)
    normalised_weights = normalise_weights(smoothed_instr_weights)

    return normalised_weights


def shrink_corr_matrix(net_weekly):
    corr_matrix_df = calc_corr_matrix(net_weekly)
    corr_matrx_values = corr_matrix_df.values
    new_corr_values = copy(corr_matrx_values)
    np.fill_diagonal(new_corr_values, np.nan)
    avg_corr = np.nanmean(new_corr_values)
    ins = corr_matrix_df.columns
    n = len(ins)
    corr_matrix = np.full((n, n), avg_corr)  # Fill entire matrix with avg_corr
    np.fill_diagonal(corr_matrix, 1.0)  # Set diagonals to 1.0
    avg_corr_matrix = pd.DataFrame(corr_matrix, index=ins, columns=ins)
    shrunk_corr_without_columns = (
            0.5 * avg_corr_matrix.values + (1 - 0.5) * corr_matrx_values)
    shrunk_corr_values = pd.DataFrame(shrunk_corr_without_columns, columns=ins, index=ins).values
    return shrunk_corr_values


def calc_corr_matrix(net, span=500000, min_periods=10):
    corr = net.ewm(span=span, min_periods=min_periods, ignore_na=True).corr(pairwise=True)
    corr_matrix_df = (corr[corr.index.get_level_values(0) < net.index[-1]]
                      .tail(len(corr.columns))
                      .droplevel(0)
                      .clip(lower=0))  # 所有小于0的值设为0
    return corr_matrix_df


def calc_smoothed_instr_weights(weights_df, subsystem_positions, smooth_weighting=125):
    instrument_weights = weights_df.reindex(subsystem_positions.index, method="ffill")
    instrument_weights[np.isnan(subsystem_positions)] = 0.0
    daily_unsmoothed_instr_weights = instrument_weights.resample('1B').mean()
    smoothed_instr_weights = daily_unsmoothed_instr_weights.ewm(span=smooth_weighting).mean()
    return smoothed_instr_weights


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
