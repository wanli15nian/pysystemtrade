import numpy as np
import pandas as pd

from refactory.utils import get_stdev_estim_for_instr_weight, get_mean_estimator, \
    get_corr_estim_for_instr_weight, optimisation, single_resampled_set_of_returns


def generate_fit_end_list(start_date, end_date):
    '''
    从结束日期开始倒推，然后reverse()
    '''
    start_dates_per_period = pd.date_range(end_date, start_date, freq='-365D').to_list()
    start_dates_per_period.reverse()
    end_list = start_dates_per_period[1:-1]
    return end_list


def calc_forecast_weights(instr_num, pnl_df, fit_end, span_multiple=50000,
                          min_periods_corr_multiple=10, min_periods_multiple=5):
    number_of_rules = len(pnl_df.columns)
    span = instr_num * span_multiple

    corr = get_corr_estim_for_instr_weight(pnl_df, min_periods_corr_multiple, instr_num, fit_end, span)


    norm_stdev, norm_mean = get_stdev_estim_for_instr_weight(pnl_df, min_periods_multiple, instr_num, fit_end, span)

    # mean_list = get_mean_estimator(pnl_df, fit_end, span, instr_num*min_periods_multiple)
    # norm_mean = [a / b for a, b in zip(mean_list, norm_factor)]

    weight = optimisation(number_of_rules, corr, norm_mean, norm_stdev)
    return weight


def reindex_and_stack_list_of_df(list_of_df):
    '''
    提取所有的list中所有df的index
    remove duplicates，把所有数据都reindex
    加毫秒进行区分，然后stack起来
    '''

    from itertools import chain
    all_indices_flattened = list(chain.from_iterable(data_item.index for data_item in list_of_df))
    common_unique_index = sorted(set(all_indices_flattened))
    data_reindexed = [data_item.reindex(common_unique_index) for data_item in list_of_df]
    for offset_value, data_item in enumerate(data_reindexed):
        data_item.index = data_item.index + pd.Timedelta("%dus" % offset_value)
    stacked_data = pd.concat(data_reindexed, axis=0)
    stacked_data = stacked_data.sort_index()
    return stacked_data


def calculate_instrument_weights(pnl_df):
    daily_pnl = pnl_df.resample("1B").sum()
    daily_pnl[daily_pnl == 0.0] = np.nan

    number = len(daily_pnl.columns)
    weekly_ret = daily_pnl.resample('W').sum()  # SP500_micro 的一些数值不对，其他的都能对的上。怀疑是不是一些nan被填充了
    fit_end = weekly_ret.index[-1]
    span = 500000
    min_periods = 10

    norm_stdev, _, = get_stdev_estim_for_instr_weight(weekly_ret, fit_end, span, min_periods)
    norm_mean = [0.5 * asset_stdev for asset_stdev in norm_stdev]
    corr = get_corr_estim_for_instr_weight(weekly_ret, fit_end, span, min_periods)

    weight = optimisation(number, corr, norm_mean, norm_stdev)
    return weight


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


# def get_turnover_for_list_of_rules(instrument_list, trading_rule_list):
#     instrument_turnover_dict = dict()
#     for instrument in instrument_list:
#         instrument_turnover_dict[instrument] = {
#             rule_name: forecast_turnover_for_indiv_instr(instrument, rule_name)
#             for rule_name in trading_rule_list
#         }
#     print('get_turnover_for_list_of_rules')
#     return instrument_turnover_dict


# def calc_net_returns_dict_for_all_instr(dict_of_sr_costs, gross_returns_dict):
#     net_returns_dict = {}
#     for instrument in gross_returns_dict.keys():
#         gross_returns = gross_returns_dict[instrument]
#         net_returns_single_instrument = {}
#         for column_name in gross_returns.columns:
#             gross_returns_daily_std = gross_returns[column_name].std()
#             daily_sr_cost = dict_of_sr_costs[column_name] / 16
#             daily_returns_cost = (daily_sr_cost * gross_returns_daily_std).item()
#             net_returns_single_instrument_rule = gross_returns[column_name] + daily_returns_cost
#             net_returns_single_instrument[column_name] = net_returns_single_instrument_rule
#         net_returns_single_instrument = pd.DataFrame(net_returns_single_instrument)
#         net_returns_dict[instrument] = net_returns_single_instrument  # CLEARED
#     net_returns = single_resampled_set_of_returns(net_returns_dict, frequency='W')  # CLEARED
#     print('calc_net_returns_dict_for_all_instr')
#     return net_returns


# def calc_buffered_pos_given_combined_forecast(volatility_scalar, position_raw):
#     # 小数点后8位开始对不上，暂时不管
#     # position_raw[position_raw < 0] = 0
#     # position_raw.fillna(0.0, inplace=True)
#     position_buffered = calc_buffered_pos_given_raw_pos(position_raw, volatility_scalar, 0.10)
#     return position_buffered


def combine_forecast(forecast, forecast_, net_, price):
    grouped = net_.groupby(level='instrument')
    net_pnl_all = {ins: group.reset_index(level='instrument', drop=True) for ins, group in grouped}
    grouped = forecast_.groupby(level='instrument')
    forecast_df_list = [group.reset_index(level='instrument', drop=True) for instrument, group in grouped]

    instruments_num = len(net_pnl_all)
    net_pnl_stacked = single_resampled_set_of_returns(net_pnl_all, frequency='W')
    start_date = net_pnl_stacked.index[0]
    end_date = net_pnl_stacked.index[-1]
    end_list = generate_fit_end_list(start_date, end_date)
    weight_df = pd.DataFrame(
        [calc_forecast_weights(instruments_num, net_pnl_stacked, end) for end in end_list],
        index=end_list, columns=net_pnl_stacked.columns)
    # To add the initial weight
    universal_index = price.index
    column_num = len(weight_df.columns)
    initial_weight = pd.DataFrame({col: 1 / column_num for col in weight_df.columns}, index=[start_date])
    weight_df = pd.concat([initial_weight, weight_df], axis=0)
    # 把按年的Index ffill成按天的Index
    weight_df = weight_df.reindex(universal_index, method='ffill').fillna(1 / column_num)
    daily_forecast_weights_resampled_unsmoothed = weight_df.resample('1B').mean()
    forecast_weights_for_rules = daily_forecast_weights_resampled_unsmoothed.ewm(span=125).mean()
    # 跳过一个weight normalisation to 1 的函数
    list_of_resampled_forecast = [forecast_df.resample('W').last() for forecast_df in forecast_df_list]
    pooled_forecast_data = reindex_and_stack_list_of_df(list_of_resampled_forecast)
    pooled_fdm = True
    ew_lookback = 250
    min_periods = 20
    if pooled_fdm == True:
        ew_lookback = ew_lookback * instruments_num
        min_periods = min_periods * instruments_num
    raw_pooled_correlations = pooled_forecast_data.ewm(span=ew_lookback, min_periods=min_periods,
                                                       ignore_na=True).corr(pairwise=True)
    size_of_matrix = len(pooled_forecast_data)
    pooled_forecast_corr_list_for_fdm = []
    for fit_end in end_list:
        corr_matrix_values = (raw_pooled_correlations[raw_pooled_correlations.index.get_level_values(0) < fit_end]
                              .tail(size_of_matrix)
                              .values)
        corr_matrix_values = corr_matrix_values[-1]
        corr_matrix_values = [max(0, value) for value in corr_matrix_values]
        pooled_forecast_corr_list_for_fdm.append(corr_matrix_values)
    # pooled_forecast_corr_list_for_fdm.insert(0, np.array([0.99, 1]))  # 为了让corr_list的element和end_list对齐，先不加起始默认matrix
    div_mult_vector = []
    for corrmatrix, start_of_period in zip(pooled_forecast_corr_list_for_fdm, end_list):
        weight_slice = forecast_weights_for_rules[:start_of_period]
        if weight_slice.shape[0] == 0:
            div_mult_vector.append(1.0)
            continue

        last_weight_for_period = np.array(weight_slice.iloc[-1])
        div_multiplier = calc_div_mult_single_period(corrmatrix, last_weight_for_period)
        div_mult_vector.append(div_multiplier)
    div_mult = pd.Series(div_mult_vector, index=end_list)
    # forecast_weights_for_rules.index 是fitting period的start dates
    div_mult_unsmoothed_daily = div_mult.reindex(forecast_weights_for_rules.index, method="ffill")
    div_mult_unsmoothed_daily[div_mult_unsmoothed_daily.isna()] = 1.0
    div_mult = div_mult_unsmoothed_daily.ewm(span=125).mean()
    # TODO: combined forecast_rule 有问题
    combined_forecast_without_cap = (forecast_weights_for_rules * forecast).sum(axis=1) * div_mult.ffill()
    combined_forecast = combined_forecast_without_cap.clip(20, -20)  # QUESTION: 小数点后8位开始对不上，暂时不管
    return combined_forecast


def calc_net_pnl(gross_pnl, cost_SR_dict):
    net_returns_single_instrument = {}
    # TODO: dict_of_instr_cost_with_pooling is specific to the target instrument, how can it be applied widely
    for column_name in gross_pnl.columns:
        cost_SR = cost_SR_dict[column_name]
        gross_pnl_rule = gross_pnl[column_name]

        daily_cost_sr = cost_SR / 16
        daily_cost = (daily_cost_sr * gross_pnl_rule.std()).item()
        net_pnl_rule = gross_pnl_rule + daily_cost

        net_returns_single_instrument[column_name] = net_pnl_rule
    return pd.DataFrame(net_returns_single_instrument)
