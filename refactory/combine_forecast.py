import numpy as np
import pandas as pd

from refactory.utils import optimisation, stack_df_list


def generate_yearly_end_list(index):
    # 从结束日期开始倒推，然后reverse()
    yearly = pd.date_range(index[-1], index[0], freq='-365D').to_list()
    yearly.reverse()
    return yearly[1:-1]


def calc_weights_daily(net_):
    # 转换成周数据
    weekly_list = [group.reset_index(level='instrument', drop=True).resample('W').sum()
                   for _, group in net_.groupby(level='instrument')]
    net_weekly = stack_df_list(weekly_list)
    end_list = generate_yearly_end_list(net_weekly.index)

    # 计算年权重
    rule_num = len(net_.columns)
    instruments_num = len(net_.index.levels[0])
    weight_yearly_raw = pd.DataFrame(
        [calc_forecast_weights(instruments_num, rule_num, net_weekly, end) for end in end_list],
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


def combine_forecast(forecast, forecast_, net_):
    weights_daily, end_list = calc_weights_daily(net_)

    weekly_list = [group.droplevel('instrument').resample('W').last()
                   for _, group in forecast_.groupby(level='instrument')]

    # pooled_forecast_data = reindex_and_stack_list_of_df(weekly_list)
    pooled_forecast_data = stack_df_list(weekly_list)
    pooled_fdm = True
    ew_lookback = 250
    min_periods = 20

    instruments_num = len(forecast_.index.levels[0])
    if pooled_fdm == True:
        ew_lookback = ew_lookback * instruments_num
        min_periods = min_periods * instruments_num
    raw_pooled_corr = pooled_forecast_data.ewm(span=ew_lookback, min_periods=min_periods,
                                               ignore_na=True).corr(pairwise=True)
    size_of_matrix = len(pooled_forecast_data)
    pooled_forecast_corr_list = []
    for fit_end in end_list:
        corr_matrix_values = (raw_pooled_corr[raw_pooled_corr.index.get_level_values(0) < fit_end]
                              .tail(size_of_matrix)
                              .values)
        corr_matrix_values = corr_matrix_values[-1]
        corr_matrix_values = [max(0, value) for value in corr_matrix_values]
        pooled_forecast_corr_list.append(corr_matrix_values)
    # pooled_forecast_corr_list.insert(0, np.array([0.99, 1]))  # 为了让corr_list的element和end_list对齐，先不加起始默认matrix
    div_mult_vector = []
    for corrmatrix, start in zip(pooled_forecast_corr_list, end_list):
        weight_slice = weights_daily[:start]
        if weight_slice.shape[0] == 0:
            div_mult_vector.append(1.0)
            continue

        last_weight_for_period = np.array(weight_slice.iloc[-1])
        div_multiplier = calc_div_mult_single_period(corrmatrix, last_weight_for_period)
        div_mult_vector.append(div_multiplier)
    div_mult = pd.Series(div_mult_vector, index=end_list)
    # forecast_weights_for_rules.index 是fitting period的start dates
    div_mult_unsmoothed_daily = div_mult.reindex(weights_daily.index, method="ffill")
    div_mult_unsmoothed_daily[div_mult_unsmoothed_daily.isna()] = 1.0
    div_mult = div_mult_unsmoothed_daily.ewm(span=125).mean()
    # TODO: combined forecast_rule 有问题
    combined_forecast_without_cap = (weights_daily * forecast).sum(axis=1) * div_mult.ffill()
    combined_forecast = combined_forecast_without_cap.clip(20, -20)  # QUESTION: 小数点后8位开始对不上，暂时不管
    return combined_forecast


def calc_forecast_weights(instr_num, rule_num, pnl_df, fit_end, span_multiple=50000,
                          min_periods_corr_multiple=10, min_periods_multiple=5):
    span = instr_num * span_multiple

    corr = get_corr_estim_for_instr_weight(pnl_df, min_periods_corr_multiple, instr_num, fit_end, span)
    norm_stdev, norm_mean = get_stdev_estim_for_instr_weight(pnl_df, min_periods_multiple, instr_num, fit_end, span)

    weight = optimisation(rule_num, corr, norm_mean, norm_stdev)
    return weight


def calc_weights_yearly(net_):
    resampled = [group.reset_index(level='instrument', drop=True).resample('W').sum()
                 for _, group in net_.groupby(level='instrument')]
    net = stack_df_list(resampled)
    end_list = generate_yearly_end_list(net.index)
    start_date = net.index[0]
    instruments_num = len(net_.index.levels[0])
    column_num = len(net_.columns)
    weight_df_raw = pd.DataFrame(
        [calc_forecast_weights(instruments_num, column_num, net, end) for end in end_list],
        index=end_list, columns=net.columns)
    # To add the initial weight
    initial_weight = pd.DataFrame({col: 1 / column_num for col in weight_df_raw.columns}, index=[start_date])
    weights_yearly = pd.concat([initial_weight, weight_df_raw], axis=0)
    return weights_yearly, end_list


def get_corr_estim_for_instr_weight(data, min_periods_corr_multiple, instr_num, fit_end, span=500000):
    min_periods = instr_num * min_periods_corr_multiple
    raw_corr = data.ewm(span=span, min_periods=min_periods, ignore_na=True).corr(
        pairwise=True)  # span 和min_periods 都是config 里面的4倍，因为4个instruments
    corr_matrix_values = (
        raw_corr[raw_corr.index.get_level_values(0) < fit_end].tail(len(data.columns)).values)  # 截取fit_period之前的数据
    corr_matrix_values = [[max(0, item) for item in sublist] for sublist in corr_matrix_values]
    return corr_matrix_values


def get_stdev_estim_for_instr_weight(data, min_periods_multiple, instr_num, fit_end, span=50000):
    min_periods = instr_num * min_periods_multiple
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


def calc_div_mult_single_period(corr, weights, dm_max=2.5):
    '''
    计算Portfolio variance in correlation space
    且设Limit
    '''
    corrmatrix = np.array([[corr[1], corr[0]], [corr[0], corr[1]]])
    try:
        variance = weights.dot(corrmatrix).dot(weights)
        risk = variance ** 0.5
    except:
        risk = np.nan
    if np.isnan(risk):
        return 1.0
    if risk < 0.0000001:
        return 1.0
    dm = np.min([1.0 / risk, dm_max])
    return dm
