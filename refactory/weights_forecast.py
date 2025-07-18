from copy import copy

import numpy as np
import pandas as pd

from refactory.base import optimisation, stack_instr


def calc_weights(net_weekly, data_for_reindex, config):
    # 计算年切分点
    end_list = get_end_list(net_weekly.index)

    # 计算年权重
    weight_yearly_raw = pd.DataFrame(
        [calc_weight_yearly(net_weekly, end, config) for end in end_list],
        index=end_list, columns=net_weekly.columns)

    # 加上最开始的日期，用平均权重
    weights_yearly = add_initial_weight(net_weekly, weight_yearly_raw)

    # 把按年的Index ffill成按天的Index
    # weight_df = weights_yearly.reindex(data_for_reindex.index, method='ffill').shift(1).backfill()
    weight_df = fix_weights_to_target_index(weights_yearly, data_for_reindex)
    unsummed_weights = weight_df.resample('1B').mean().ewm(span=125).mean()
    weights_daily = weights_sum_to_one(unsummed_weights)
    return weights_daily


def weights_sum_to_one(weights):
    sum_weights = weights.sum(axis=1).replace(0.0, 0.0001)
    normalised_weights = weights.div(sum_weights, axis=0)
    return normalised_weights


def fix_weights_to_target_index(weights, data):
    data_ffill = data.ffill()
    # data_ffill[(~data_ffill.isna()).sum(axis=1) == 0] = 0
    data_ffill[data_ffill.isna().any(axis=1)] = 0
    resampled_weights = weights.reindex(data_ffill.index, method='ffill')
    resampled_weights[np.isnan(data_ffill)] = 0.0
    return resampled_weights


def add_initial_weight(net_weekly, weight_yearly_raw):
    initial_date = net_weekly.index[0]
    rules = weight_yearly_raw.columns
    initial_weight = pd.DataFrame({rule: 1 / len(rules) for rule in rules}, index=[initial_date])
    weights_yearly = pd.concat([initial_weight, weight_yearly_raw], axis=0)
    return weights_yearly


def calc_weight_yearly(pnl, fit_end, config, floor=True):
    corr_span = config['corr_span']
    corr_min_periods = config['corr_min_periods']
    multiple_span = config['multiple_span']
    multiple_min_periods = config['multiple_min_periods']
    shrinkage_corr = config['shrinkage_corr']
    shrinkage_sr = config['shrinkage_sr']
    sr_target = config['sr_target']
    equalise_vol = config['equalise_vol']
    all_assets = pnl.columns.to_series()

    raw_corr = pnl.ewm(span=corr_span, min_periods=corr_min_periods, ignore_na=True).corr(pairwise=True)
    corr_matrix_values = raw_corr[raw_corr.index.get_level_values(0) <= fit_end].tail(len(pnl.columns)).values
    if floor:
        corr_matrix_values[corr_matrix_values < 0.0] = 0.0
        np.fill_diagonal(corr_matrix_values, 1.0)
    corr_array = np.clip(corr_matrix_values, a_min=0, a_max=None)
    corr_unshrunk = pd.DataFrame(corr_array, index=all_assets, columns=all_assets)

    # 计算标准差和均值
    ewm = pnl.ewm(span=multiple_span, min_periods=multiple_min_periods)
    std_daily = ewm.std().asof(fit_end)
    mean_daily = ewm.mean().asof(fit_end)

    # 年化处理
    std = std_daily * ((365.25 / 7.0) ** 0.5)
    mean_unshrunk = mean_daily * (365.25 / 7.0)

    assets_no_data = assets_with_no_data(corr_unshrunk, std, mean_unshrunk)
    assets = all_assets[~all_assets.isin(assets_no_data)]

    if assets.empty:
        weights = []
        return weights

    elif len(assets) != len(all_assets):
        corr_unshrunk, mean_unshrunk, std = prepare_valid_param(assets, corr_unshrunk, std, mean_unshrunk)

    corr = shrink_corr_to_average(corr_unshrunk, shrinkage_corr)
    mean = shrink_mean_to_average(mean_unshrunk, std, shrinkage_sr, sr_target)
    # 计算归一化标准差和归一化均值

    if equalise_vol:
        std_mean = np.nanmean(std)
        norm_mean = mean * (std_mean / std)
        norm_std = [std_mean] * len(std)
        mean, std = norm_mean, norm_std

    weights = optimisation(corr, mean, std)
    return weights


def shrink_mean_to_average(mean, std, shrinkage_sr, sr_target):
    mean_shrunk = sr_target * shrinkage_sr * std + (1 - shrinkage_sr) * mean
    return mean_shrunk


def shrink_corr_to_average(raw_corr, shrinkage_corr=1.0):
    raw_corr_ = copy(np.array(raw_corr))
    size = len(raw_corr_)
    np.fill_diagonal(raw_corr_, np.nan)

    if np.all(np.isnan(raw_corr_)):
        return np.nan
    average_corr = np.nanmean(raw_corr_)

    dummy_matrix = np.full((size, size), average_corr)
    np.fill_diagonal(dummy_matrix, 1.0)

    shrunk_corr = shrinkage_corr * dummy_matrix + (1 - shrinkage_corr) * np.array(raw_corr)

    return shrunk_corr


def prepare_valid_param(assets_with_data, corr_raw, std_raw, mean_raw):
    std = std_raw[assets_with_data]
    mean = mean_raw[assets_with_data]
    corr = corr_raw[assets_with_data].loc[assets_with_data]
    return corr, mean, std


def assets_with_no_data(corr, std, mean):
    corr_check = (~corr.isna()).sum() < 2
    mean_check = (mean == np.nan)
    std_check = (std == np.nan)
    compiled = corr_check | mean_check | std_check
    compiled = compiled[compiled]
    return compiled.index.to_series()


def calc_div_mult_daily(weights, forecast):
    forecast_weekly = stack_instr(forecast, 'W', 'last')
    end_list = get_end_list(forecast_weekly.index)
    instrument_number = len(forecast.index.get_level_values(0).unique())
    lookback = 250 * instrument_number
    min_periods = 20 * instrument_number

    corr_weekly = pd.Series(
        [calc_forecast_corr(forecast_weekly, end, lookback, min_periods) for end in end_list],
        index=end_list
    )
    first_corr = pd.Series([np.array([[1.0, 0.99], [0.99, 1.0]])], index=[forecast_weekly.index[0]])
    corr_weekly = pd.concat([first_corr, corr_weekly])

    # corr_weekly = [calc_forecast_corr(forecast_weekly, end, lookback, min_periods) for end in end_list]
    # first_corr = np.array([[1.0, 0.99], [0.99, 1.0]])
    # corr_weekly.insert(0, first_corr)

    multiplier_yearly = pd.Series(
        [calc_div_mult_yearly(weights, corr_weekly, end) for end in end_list],
        index=end_list)

    # multiplier_daily = multiplier_yearly.reindex(weights.index, method="ffill").fillna(1.0).ewm(span=125).mean()
    # FIXME: 因为reindex 问题，加一个bfill
    multiplier_daily = multiplier_yearly.reindex(weights.index, method="ffill").shift(1).bfill().fillna(1.0).ewm(
        span=125).mean()
    return multiplier_daily


def calc_forecast_corr_multi(forecast_, lookback=250, periods=20):
    forecast_weekly = forecast_.groupby([pd.Grouper(level=1, freq='W'), 'instrument']).last().droplevel('instrument')
    instruments_num = len(forecast_.index.levels[0])
    corr_weekly = forecast_weekly.ewm(span=lookback * instruments_num, min_periods=periods * instruments_num,
                                      ignore_na=True).corr(pairwise=True)
    return corr_weekly


def calc_forecast_corr(forecast, fit_end, lookback=250, periods=20):
    corr_weekly = forecast.ewm(span=lookback, min_periods=periods, ignore_na=True).corr(pairwise=True)
    corr_matrix_values = corr_weekly[corr_weekly.index.get_level_values(0) <= fit_end].tail(
        len(forecast.columns)).values
    corr = np.clip(corr_matrix_values, a_min=0, a_max=None)
    return corr


def calc_div_mult_yearly(weights_daily, corr_weekly, end, dm_max=2.5):
    corr_weekly = pd.DataFrame(corr_weekly)
    corr_matrix = corr_weekly[corr_weekly.index.get_level_values(0) <= end].tail(
        len(corr_weekly.columns)).values[0][0]
    if len(corr_matrix) == 0:
        return 1.0
    # weights = np.array(weights_daily[weights_daily.index <= end].iloc[-1])
    filtered = weights_daily[weights_daily.index <= end]
    if filtered.empty:
        div_mult = 1.0
    else:
        weights = np.array(filtered.iloc[-1])
        risk = np.sqrt(weights.dot(corr_matrix).dot(weights))
        risk = 1.0 if (np.isnan(risk) or risk < 1e-7) else risk
        div_mult = min(1.0 / risk, dm_max)
    return div_mult


def get_end_list(daily_index):
    # 转成周频的
    daily_series = pd.Series(index=daily_index)
    weekly_series = daily_series.resample('W').last()
    weekly_index = weekly_series.index
    # 从结束日期开始倒推，然后reverse()
    yearly = pd.date_range(weekly_index[-1], weekly_index[0], freq='-365D').to_list()
    yearly.reverse()
    end_list = yearly[1:-1]
    return end_list
