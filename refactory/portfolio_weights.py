import numpy as np
import pandas as pd
from copy import copy

from refactory.utils import single_resampled_set_of_returns, optimisation


def calc_corr_matrix(net_return_df, span=500000, min_periods=10):
    end = net_return_df.index[-1]
    corr = net_return_df.ewm(span=span, min_periods=min_periods, ignore_na=True).corr(pairwise=True)
    size_of_matrix = len(corr.columns)
    corr_matrix_values = (corr[corr.index.get_level_values(0) < end].tail(size_of_matrix).values)
    corr_matrix_values[corr_matrix_values < 0.0] = 0.0
    corr_matrix_df = pd.DataFrame(corr_matrix_values, columns=corr.columns)
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


def calc_shrunk_means(annualised_return_mean, annualised_return_std, shrinkage_sr=0.9, target_sr=0.5):
    sr_estimates = (annualised_return_mean / annualised_return_std).to_list()
    post_sr_list = [(shrinkage_sr * target_sr) + (1 - shrinkage_sr) * estimatedSR for estimatedSR in sr_estimates]
    shrunk_means_values = (post_sr_list * annualised_return_std).to_list()
    instruments = annualised_return_mean.index.to_list()
    shrunk_means = [(asset_name, mean_value) for (asset_name, mean_value) in zip(instruments, shrunk_means_values)]
    return shrunk_means


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


def calc_portfolio_weights(net_return_raw, positions):
    net_return_df = single_resampled_set_of_returns({'asset': net_return_raw}, 'W')

    corr_matrix_df = calc_corr_matrix(net_return_df)
    shrunk_corr = calc_avg_corr_matrix(corr_matrix_df)
    annualised_return_mean, annualised_return_std = calc_net_mean_std(net_return_df)
    # FIXME:这个数算出来后面为什么不用？
    shrunk_means = calc_shrunk_means(annualised_return_mean, annualised_return_std)

    target_sr = 0.5
    norm_std = [annualised_return_std.mean()] * 4
    mean_list = [target_sr * asset_stdev for asset_stdev in norm_std]
    instruments1 = shrunk_corr.columns.to_list()
    weights = optimisation(len(instruments1), corr=shrunk_corr.values, norm_mean=mean_list, norm_stdev=norm_std)

    start = net_return_df.index[0]
    weights_df = pd.DataFrame({asset_name: weight for (asset_name, weight) in zip(instruments1, weights)},
                              index=[start])

    positions[(~positions.isna()).sum(axis=1) == 0] = 0
    smoothed_instr_weights = calc_smoothed_instr_weights(weights_df, positions)
    normalised_weights = normalise_weights(smoothed_instr_weights)

    return normalised_weights
