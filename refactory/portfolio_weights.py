import pandas as pd


def calc_corr_matrix(net_return_df, span=500000, min_periods=10):
    end = net_return_df.index[-1]
    corr = net_return_df.ewm(span=span, min_periods=min_periods, ignore_na=True).corr(pairwise=True)
    size_of_matrix = len(corr.columns)
    corr_matrix_values = (corr[corr.index.get_level_values(0) < end].tail(size_of_matrix).values)
    corr_matrix_values[corr_matrix_values < 0.0] = 0.0
    corr_matrix_df = pd.DataFrame(corr_matrix_values, columns=corr.columns)
    return corr_matrix_df
