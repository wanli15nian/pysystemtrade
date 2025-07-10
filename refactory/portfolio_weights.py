from copy import copy

import numpy as np
import pandas as pd

from refactory.utils import optimisation, stack_df_list


def calc_corr_matrix(net, span=500000, min_periods=10):
    corr = net.ewm(span=span, min_periods=min_periods, ignore_na=True).corr(pairwise=True)
    corr_matrix_df = (corr[corr.index.get_level_values(0) < net.index[-1]]
                      .tail(len(corr.columns))
                      .droplevel(0)
                      .clip(lower=0))  # 所有小于0的值设为0
    return corr_matrix_df


def calc_net_mean_std(net_return_df):
    end = net_return_df.index[-1]
    last_index = net_return_df.index[net_return_df.index < end].size - 1
    ewm_return = net_return_df.ewm(span=50000, min_periods=5)
    exponential_mean = ewm_return.mean()
    annualised_return_mean = exponential_mean.iloc[last_index] * 365.25 / 7.0
    exponential_std = ewm_return.std()
    annualised_return_std = exponential_std.iloc[last_index] * (365.25 / 7.0) ** 0.5
    return annualised_return_mean, annualised_return_std


def calc_avg_corr_matrix(corr_matrix_df, shrinkage_corr=0.5):
    new_corr_values = copy(corr_matrix_df.values)
    np.fill_diagonal(new_corr_values, np.nan)
    avg_corr = np.nanmean(new_corr_values)

    ins = corr_matrix_df.columns
    n = len(ins)
    corr_matrix = np.full((n, n), avg_corr)  # Fill entire matrix with avg_corr
    np.fill_diagonal(corr_matrix, 1.0)  # Set diagonals to 1.0
    avg_corr_matrix = pd.DataFrame(corr_matrix, index=ins, columns=ins)

    shrunk_corr_without_columns = (
            shrinkage_corr * avg_corr_matrix.values + (1 - shrinkage_corr) * corr_matrix_df.values)
    shrunk_corr = pd.DataFrame(shrunk_corr_without_columns, columns=ins, index=ins)

    return shrunk_corr


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


# TODO: stack这种傻办法需要改成直接用multiIndex做
def calc_portfolio_weights(net_return_raw, positions):
    data_dict = {'asset': net_return_raw}
    resampled = [pnl.resample('W').sum() for pnl in data_dict.values()]
    net_return_df = stack_df_list(resampled)

    corr_matrix_df = calc_corr_matrix(net_return_df)
    shrunk_corr = calc_avg_corr_matrix(corr_matrix_df)
    annualised_return_mean, annualised_return_std = calc_net_mean_std(net_return_df)
    # shrunk_means = calc_shrunk_means(annualised_return_mean, annualised_return_std)

    target_sr = 0.5
    norm_std = [annualised_return_std.mean()] * 4
    mean_list = [target_sr * asset_stdev for asset_stdev in norm_std]
    weights = optimisation(corr=shrunk_corr.values, norm_mean=mean_list, norm_stdev=norm_std)

    instruments = net_return_df.columns.to_list()
    start = net_return_df.index[0]
    weights_df = pd.DataFrame({asset_name: weight for (asset_name, weight) in zip(instruments, weights)},
                              index=[start])

    positions[(~positions.isna()).sum(axis=1) == 0] = 0
    smoothed_instr_weights = calc_smoothed_instr_weights(weights_df, positions)
    normalised_weights = normalise_weights(smoothed_instr_weights)

    return normalised_weights
