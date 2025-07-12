import numpy as np
import pandas as pd

from refactory.base import optimisation


# TODO：又是日频，又是周频，又是年频，有些乱


def calc_forecast_weights(net_):
    # 计算年切分点
    end_list = get_end_list(net_.index.levels[1])

    # net_weekly = net_.groupby([pd.Grouper(level=1, freq='W'), 'instrument']).sum()
    # net_weekly = net_weekly.droplevel('instrument')
    # net_weekly = net_.droplevel('instrument')
    # net_weekly.sort_index(ascending=True, inplace=True)
    # weekly_raw = net_.groupby(level=0).resample('W', level=1).sum()
    # net_weekly = weekly_raw.unstack(level=0).stack(dropna=False).droplevel('instrument').sort_index(ascending=True)
    net_weekly = (net_.groupby(level=0)
                  .resample('W', level=1).sum()
                  .unstack(level=0)
                  .stack(dropna=False)
                  .droplevel('instrument')
                  .sort_index(ascending=True))

    # 计算年权重
    instruments_num = len(net_.index.levels[0])
    weight_yearly_raw = pd.DataFrame(
        [calc_forecast_weight_yearly(instruments_num, net_weekly, end) for end in end_list],
        index=end_list, columns=net_weekly.columns)

    # 加上最开始的日期，用平均权重
    initial_date = net_weekly.index[0]
    rules = weight_yearly_raw.columns
    initial_weight = pd.DataFrame({rule: 1 / len(rules) for rule in rules}, index=[initial_date])
    weights_yearly = pd.concat([initial_weight, weight_yearly_raw], axis=0)

    # 把按年的Index ffill成按天的Index
    universal_index = net_.index.levels[1]
    weight_df = weights_yearly.reindex(universal_index, method='ffill').fillna(1 / len(weights_yearly.columns))
    weights_daily = weight_df.resample('1B').mean().ewm(span=125).mean()

    return weights_daily


def calc_forecast_weight_yearly(instr_num, pnl, fit_end, span_multiple=50000, min_periods_corr=10,
                                min_periods_multiple=5):
    span = instr_num * span_multiple

    min_periods = instr_num * min_periods_corr
    raw_corr = pnl.ewm(span=span, min_periods=min_periods, ignore_na=True).corr(pairwise=True)
    corr_matrix_values = raw_corr[raw_corr.index.get_level_values(0) <= fit_end].tail(len(pnl.columns)).values
    corr = np.clip(corr_matrix_values, a_min=0, a_max=None)

    # 计算标准差和均值
    ewm = pnl.ewm(span=span, min_periods=(instr_num * min_periods_multiple))
    std_daily = ewm.std().asof(fit_end)
    mean_daily = ewm.mean().asof(fit_end)

    # 年化处理
    std = std_daily * ((365.25 / 7.0) ** 0.5)
    mean = mean_daily * (365.25 / 7.0)
    # 计算归一化标准差和归一化均值
    mean_std = np.nanmean(std)
    norm_std = [mean_std] * len(std)
    norm_mean = mean * (mean_std / std)

    weights = optimisation(corr, norm_mean, norm_std)
    return weights


def calc_div_mult_daily(weights_daily, forecast_):
    end_list = get_end_list(weights_daily.index)
    corr_weekly = calc_forecast_corr(forecast_)
    multiplier_yearly = pd.Series(
        [calc_div_mult_yearly(weights_daily, corr_weekly, end) for end in end_list],
        index=end_list)
    multiplier_daily = multiplier_yearly.reindex(weights_daily.index, method="ffill").fillna(1.0).ewm(span=125).mean()
    return multiplier_daily


def calc_forecast_corr(forecast_, lookback=250, periods=20):
    forecast_weekly = forecast_.groupby([pd.Grouper(level=1, freq='W'), 'instrument']).last()
    forecast_weekly = forecast_weekly.droplevel('instrument')
    instruments_num = len(forecast_.index.levels[0])
    corr_weekly = forecast_weekly.ewm(span=lookback * instruments_num, min_periods=periods * instruments_num,
                                      ignore_na=True).corr(pairwise=True)
    return corr_weekly


def calc_div_mult_yearly(weights_daily, corr_weekly, end, dm_max=2.5):
    corr_matrix = corr_weekly[corr_weekly.index.get_level_values(0) <= end].tail(
        len(corr_weekly.columns)).values
    weights = np.array(weights_daily[weights_daily.index <= end].iloc[-1])
    risk = np.sqrt(weights.dot(corr_matrix).dot(weights))
    risk = 1.0 if (np.isnan(risk) or risk < 1e-7) else risk
    return min(1.0 / risk, dm_max)


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
