import numpy as np
import pandas as pd
from copy import copy


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
