from copy import copy

import numpy as np
import pandas as pd

from refactory.core import optimisation, calc_net


def calc_forecast_weights(gross_inst, cost_sr, instrument):
    instruments = gross_inst.index.levels[0]
    net_daily = pd.concat([calc_net(gross_inst.loc[i], cost_sr) for i in instruments]
                          , keys=instruments, names=['instrument', 'datetime'])

    net_weekly = net_daily.groupby(level=0).resample('W', level=1).sum()
    net_weekly = align_stack(net_weekly)

    instruments_num = len(net_daily.index.levels[0])
    config = {
        'corr_span': instruments_num * 50000,
        'corr_min_periods': instruments_num * 10,
        'multiple_span': instruments_num * 50000,
        'multiple_min_periods': instruments_num * 5,
        'shrinkage_corr': 0.5,
        'shrinkage_sr': 0.9,
        'sr_target': 0.5,
        'equalise_vol': True
    }
    weights_yearly = calc_weights_yearly(net_weekly, config)

    idx_daily = gross_inst.loc[instrument].index
    weights_daily = to_daily_weights(weights_yearly, idx_daily)
    return weights_daily


def calc_instrument_weights(net_daily_inst):
    config = {
        'corr_span': 500000,
        'corr_min_periods': 10,
        'multiple_span': 50000,
        'multiple_min_periods': 5,
        'shrinkage_corr': 0.5,
        'shrinkage_sr': 0.9,
        'sr_target': 0.5,
        'equalise_vol': True
    }

    net_weekly = (net_daily_inst
                  .replace(0.0, pd.NA)
                  .unstack(level=0)
                  .resample('W').sum())
    weights_yearly = calc_weights_yearly(net_weekly, config)

    idx_daily = net_daily_inst.unstack(level=0).index
    weights_daily = to_daily_weights(weights_yearly, idx_daily)
    return weights_daily


def calc_weights_yearly(net_weekly, config):
    # 计算年切分点
    year_end_list = get_end_list(net_weekly.index)
    # 计算年权重
    weights_yearly = pd.DataFrame(
        [calc_weight_yearly(net_weekly, end, config) for end in year_end_list],
        index=year_end_list, columns=net_weekly.columns)
    # 加上最开始的日期，用平均权重
    weights_yearly.loc[net_weekly.index[0]] = 1 / len(weights_yearly.columns)
    weights_yearly = weights_yearly.sort_index()
    return weights_yearly


def to_daily_weights(weights_yearly, idx_daily):
    dailly = weights_yearly.reindex(idx_daily, method='ffill')
    smooth_daily = dailly.resample('1B').mean().ewm(span=125).mean()
    sum_daily = smooth_daily.sum(axis=1).replace(0.0, 0.0001)
    weights_daily = smooth_daily.div(sum_daily, axis=0)
    return weights_daily


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


def calc_rule_div_mult_daily(weights, net_daily):
    net_weekly = net_daily.groupby(level=0).resample('W', level=1).last()
    net_weekly = align_stack(net_weekly)
    instrument_number = len(net_daily.index.get_level_values(0).unique())
    config = {
        'lookback': 250 * instrument_number,
        'min_periods': 20 * instrument_number
    }
    multiplier_daily = calc_div_mult_daily(net_weekly, config, weights)
    return multiplier_daily


def calc_instrument_div_mult_daily(weights, net_daily):
    # FIXME: 没看出来这么做的意义
    # net_weekly = (net_daily.unstack().T
    #               .cumsum().ffill()
    #               .resample('W').last()
    #               .diff())
    net_weekly = net_daily.unstack().T.resample('W').sum()

    config = {
        'lookback': 25,
        'min_periods': 20
    }
    multiplier_daily = calc_div_mult_daily(net_weekly, config, weights)
    return multiplier_daily


def calc_div_mult_daily(net_weekly, config, weights):
    lookback = config['lookback']
    min_periods = config['min_periods']

    end_list = get_end_list(net_weekly.index)

    corr_weekly = pd.Series(
        [calc_forecast_corr(net_weekly, end, lookback, min_periods) for end in end_list],
        index=end_list
    )
    first_corr = pd.Series([np.array([[1.0, 0.99], [0.99, 1.0]])], index=[net_weekly.index[0]])
    corr_weekly = pd.concat([first_corr, corr_weekly])
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


def align_stack(df_multi):
    return (df_multi.unstack(level=0)
            .stack(dropna=False)
            .droplevel('instrument')
            .sort_index(ascending=True))
