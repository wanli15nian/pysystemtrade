import numpy as np
import pandas as pd

from refactory.utils import optimisation, stack_df_list


def calc_weights_and_multiplier(forecast_, net_):
    # TODO: 这个函数返回两个值不够简单，end_list的耦合需要解开
    weights_daily, end_list = calc_weights_daily(net_)

    # 可以直接用weely的index生成同样的end_list吗？如果可以就能直接解开end_list的耦合
    # end_list = generate_yearly_end_list(weights_daily.index)
    corr_weekly = calc_corr_weekly(forecast_)
    multiplier_yearly = pd.Series(
        [calc_div_multiplier(weights_daily, get_corr_end(corr_weekly, end), end) for end in end_list],
        index=end_list)
    multiplier_daily = multiplier_yearly.reindex(weights_daily.index, method="ffill").fillna(1.0).ewm(span=125).mean()
    return weights_daily, multiplier_daily


def get_longest_index(list_of_df):
    longest_index_len = 0
    longest_index = 0
    for df in list_of_df:
        if df.shape[0] > longest_index_len:
            longest_index_len = df.shape[0]
            longest_index = df.index
    return longest_index


def get_multi_index_df(list_of_df, instruments):
    dfs_named = {instr: df for instr, df in zip(instruments, list_of_df)}
    combined_df = pd.concat(dfs_named)
    multi_index_df = (combined_df.rename_axis(['instruments', 'date']).swaplevel().sort_index())
    return multi_index_df



def calc_weights_daily(net_):
    # 可以用net_直接算吗？跳过resample weekly会有影响吗？
    # end_list = generate_yearly_end_list(net_.index.levels[1])

    # 转换成周数据
    weekly_list = [group.reset_index(level='instrument', drop=True).resample('W').sum()
                   for _, group in net_.groupby(level='instrument')]
    net_weekly = stack_df_list(weekly_list)


    temp = get_multi_index_df(weekly_list, net_.index.levels[0])

    longest_index = get_longest_index(weekly_list)
    new_end_list = generate_yearly_end_list(longest_index)

    end_list = generate_yearly_end_list(net_weekly.index)

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
    # end_list = weights_yearly.index[1:].to_list()     #end_list和weights的index只差最开始的一个日期

    # 把按年的Index ffill成按天的Index
    universal_index = net_.index.levels[1]
    weight_df = weights_yearly.reindex(universal_index, method='ffill').fillna(1 / len(weights_yearly.columns))
    weights_daily = weight_df.resample('1B').mean().ewm(span=125).mean()
    # weight_df = weights_yearly.reindex(price.index, method='ffill').fillna(1 / rule_num) # 原先是reindex为price的，改成了net的,简单测试没问题

    return weights_daily, end_list


def generate_yearly_end_list(index):
    # 从结束日期开始倒推，然后reverse()
    yearly = pd.date_range(index[-1], index[0], freq='-365D').to_list()
    yearly.reverse()
    return yearly[1:-1]


def calc_forecast_weights(instr_num, pnl, fit_end, span_multiple=50000, min_periods_corr_multiple=10,
                          min_periods_multiple=5):
    span = instr_num * span_multiple

    min_periods = instr_num * min_periods_corr_multiple
    corr = calc_corr_matrix(pnl, min_periods, fit_end, span)

    periods = instr_num * min_periods_multiple
    norm_std, norm_mean = calc_mean_std(pnl, periods, fit_end, span)

    weights = optimisation(corr, norm_mean, norm_std)
    return weights


def calc_corr_matrix(data, min_periods, fit_end, span=500000):
    raw_corr = data.ewm(span=span, min_periods=min_periods, ignore_na=True).corr(
        pairwise=True)  # span 和min_periods 都是config 里面的4倍，因为4个instruments
    corr_matrix_values = (
        raw_corr[raw_corr.index.get_level_values(0) < fit_end].tail(len(data.columns)).values)  # 截取fit_period之前的数据
    corr_matrix_values = [[max(0, item) for item in sublist] for sublist in corr_matrix_values]
    return corr_matrix_values


def calc_mean_std(data, min_periods, fit_end, span=50000):
    last_index = data.index[data.index < fit_end].size - 1
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


def calc_corr_weekly(forecast_, lookback=250, periods=20):
    weekly_list = [group.droplevel('instrument').resample('W').last()
                   for _, group in forecast_.groupby(level='instrument')]
    forecast_weekly = stack_df_list(weekly_list)
    instruments_num = len(forecast_.index.levels[0])
    corr_weekly = forecast_weekly.ewm(span=lookback * instruments_num, min_periods=periods * instruments_num,
                                      ignore_na=True).corr(pairwise=True)
    return corr_weekly


def calc_div_multiplier(weights_daily, corr, end):
    # 获取weights_daily数组中从0到end的切片
    weight_slice = weights_daily[:end]
    # 如果切片的长度为0，则返回1.0
    if weight_slice.shape[0] == 0:
        return 1.0
    # 获取切片中最后一个元素的数组
    last_weight_for_period = np.array(weight_slice.iloc[-1])
    # 调用calc_div_mult_single_period函数，传入corr和last_weight_for_period，返回结果
    return calc_div_mult_single_period(corr, last_weight_for_period)


def calc_div_mult_single_period(corr, weights, dm_max=2.5):
    # 计算Portfolio variance in correlation space, 且设Limit
    corr_matrix = np.array([[corr[1], corr[0]], [corr[0], corr[1]]])
    try:
        risk = np.sqrt(weights.dot(corr_matrix).dot(weights))
    except:
        return 1.0
    if np.isnan(risk) or risk < 1e-7:
        return 1.0
    return min(1.0 / risk, dm_max)


def get_corr_end(corr_weekly, end):
    corr_end = (corr_weekly[corr_weekly.index.get_level_values(0) < end]
                .tail(len(corr_weekly.index.levels[0]))
                .values)[-1]
    corr_end = [max(0, value) for value in corr_end]
    return corr_end
