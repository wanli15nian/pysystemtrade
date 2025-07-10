import numpy as np
import pandas as pd

from refactory.utils import optimisation


# TODO：又是日频，又是周频，又是年频，有些乱

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


def calc_weights_daily(net_):
    # 计算年切分点
    end_list = get_end_list(net_.index.levels[1])

    net_weekly = net_.groupby([pd.Grouper(level=1, freq='W'), 'instrument']).last()
    net_weekly = net_weekly.droplevel('instrument')

    # 计算年权重
    instruments_num = len(net_.index.levels[0])
    weight_yearly_raw = pd.DataFrame(
        [calc_forecast_weights(instruments_num, net_weekly, end) for end in end_list],
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


def calc_forecast_weights(instr_num, pnl, fit_end, span_multiple=50000, min_periods_corr_multiple=10,
                          min_periods_multiple=5):
    span = instr_num * span_multiple

    min_periods = instr_num * min_periods_corr_multiple
    corr = calc_corr_matrix(pnl, fit_end, min_periods, span)

    periods = instr_num * min_periods_multiple
    norm_std, norm_mean = calc_mean_std(pnl, fit_end, periods, span)

    weights = optimisation(corr, norm_mean, norm_std)
    return weights


def calc_corr_matrix(data, fit_end, min_periods, span):
    raw_corr = data.ewm(span=span, min_periods=min_periods, ignore_na=True).corr(
        pairwise=True)  # span 和min_periods 都是config 里面的4倍，因为4个instruments
    corr_matrix_values = (
        raw_corr[raw_corr.index.get_level_values(0) <= fit_end].tail(len(data.columns)).values)  # 截取fit_period之前的数据
    corr_matrix_values = [[max(0, item) for item in sublist] for sublist in corr_matrix_values]
    return corr_matrix_values


def calc_mean_std(data, fit_end, min_periods, span=50000):
    last_index = data.index[data.index <= fit_end].size - 1
    # 计算标准差和均值
    std_daily = data.ewm(span=span, min_periods=min_periods).std().iloc[last_index]
    mean_daily = data.ewm(span=span, min_periods=min_periods).mean().iloc[last_index]
    # 年化处理
    std = std_daily * ((365.25 / 7.0) ** 0.5)
    mean = mean_daily * (365.25 / 7.0)
    # 计算归一化标准差和归一化均值
    norm_std = [(np.nanmean(std))] * len(std)
    norm_mean = mean / (std / np.nanmean(std))
    return norm_std, norm_mean


def calc_div_mult_daily(weights_daily, forecast_):
    end_list = get_end_list(weights_daily.index)
    corr_weekly = calc_corr_weekly(forecast_)
    multiplier_yearly = pd.Series(
        [calc_div_multiplier(weights_daily, get_corr_end(corr_weekly, end), end) for end in end_list],
        index=end_list)
    multiplier_daily = multiplier_yearly.reindex(weights_daily.index, method="ffill").fillna(1.0).ewm(span=125).mean()
    return multiplier_daily


def calc_corr_weekly(forecast_, lookback=250, periods=20):
    forecast_weekly = forecast_.groupby([pd.Grouper(level=1, freq='W'), 'instrument']).last()
    forecast_weekly = forecast_weekly.droplevel('instrument')
    instruments_num = len(forecast_.index.levels[0])
    corr_weekly = forecast_weekly.ewm(span=lookback * instruments_num, min_periods=periods * instruments_num,
                                      ignore_na=True).corr(pairwise=True)
    return corr_weekly


def calc_div_multiplier(weights_daily, corr, end, dm_max=2.5):
    weight_slice = weights_daily[weights_daily.index <= end]
    if weight_slice.shape[0] == 0:
        return 1.0
    weights = np.array(weight_slice.iloc[-1])
    # 计算Portfolio variance in correlation space, 且设Limit
    # FIXME:如果是3个rule，这代码就有问题了
    corr_matrix = np.array([[corr[1], corr[0]], [corr[0], corr[1]]])
    try:
        risk = np.sqrt(weights.dot(corr_matrix).dot(weights))
    except:
        risk = 1.0
    risk = 1.0 if (np.isnan(risk) or risk < 1e-7) else risk
    return min(1.0 / risk, dm_max)


def get_corr_end(corr_weekly, end):
    corr_end = (corr_weekly[corr_weekly.index.get_level_values(0) <= end]
                .tail(len(corr_weekly.index.levels[0]))
                .values)[-1]
    corr_end = [max(0, value) for value in corr_end]
    return corr_end
