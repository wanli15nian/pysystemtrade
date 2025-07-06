import numpy as np
import pandas as pd

from refactory.cost_SR import calc_annual_trading_cost_per_contract
from refactory.data_util import get_point_size, get_daily_price
from refactory.forecast import calculate_forecasts
from refactory.target_volatility import calc_target_position
from refactory.turnover_forecast import instrument_forecast_turnover
from refactory.utils import calc_mixed_volatility, get_stdev_estimator_for_instrument_weight, get_mean_estimator, \
    get_corr_estimator_for_instrument_weight, optimisation, single_resampled_set_of_returns, calc_volatility_scalar


def generate_fit_end_list(start_date, end_date):
    '''
    从结束日期开始倒推，然后reverse()
    '''
    start_dates_per_period = pd.date_range(end_date, start_date, freq='-365D').to_list()
    start_dates_per_period.reverse()
    end_list = start_dates_per_period[1:-1]
    print('generate_fit_end_list')
    return end_list


def calc_cost(pos_target, price, point_size, trading_cost):
    # Actually output in price space to match gross returns
    # These will be annualised figure, make it a small loss every day
    # FIXME: 完全没有看明白这个calc_cost的计算逻辑
    annualised_price_vol_points = calc_mixed_volatility(price.diff(), slow_vol_years=10)
    sr_cost_as_annualised_figure = (-trading_cost * pos_target * annualised_price_vol_points * 16).bfill()
    period_intervals_in_seconds = sr_cost_as_annualised_figure.index.to_series().diff().dt.total_seconds()
    costs_in_points = sr_cost_as_annualised_figure * period_intervals_in_seconds / (365.25 * 24 * 60 * 60)
    costs = costs_in_points * point_size  # 后续有个fx 的序列，但目前不加
    print('calc_cost')
    return costs


def calc_forecast_weights(instruments, pnl_df, fit_end, span_multiple=50000,
                          min_periods_corr_multiple=10, min_periods_multiple=5):
    # instruments = trading_instruments

    number_of_rules = len(pnl_df.columns)
    span = len(instruments) * span_multiple
    min_periods_corr = len(instruments) * min_periods_corr_multiple
    min_periods = len(instruments) * min_periods_multiple
    norm_stdev, norm_factor, stdev_list = get_stdev_estimator_for_instrument_weight(pnl_df, fit_end, span, min_periods)
    mean_list = get_mean_estimator(pnl_df, fit_end, span, min_periods)
    norm_mean = [a / b for a, b in zip(mean_list, norm_factor)]
    corr = get_corr_estimator_for_instrument_weight(pnl_df, fit_end, span, min_periods_corr)  # Corr CLEARED
    weight = optimisation(number_of_rules, corr, norm_mean, norm_stdev)
    print('calc_forecast_weights')
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
    print('combine_instrument_pnl_df')
    return stacked_data


def calculate_instrument_weights(pnl_df):
    daily_pnl = pnl_df.resample("1B").sum()
    daily_pnl[daily_pnl == 0.0] = np.nan

    number = len(daily_pnl.columns)
    weekly_ret = daily_pnl.resample('W').sum()  # SP500_micro 的一些数值不对，其他的都能对的上。怀疑是不是一些nan被填充了
    fit_end = weekly_ret.index[-1]
    span = 500000
    min_periods = 10

    norm_stdev, _, _ = get_stdev_estimator_for_instrument_weight(weekly_ret, fit_end, span, min_periods)
    norm_mean = [0.5 * asset_stdev for asset_stdev in norm_stdev]
    corr = get_corr_estimator_for_instrument_weight(weekly_ret, fit_end, span, min_periods)

    weight = optimisation(number, corr, norm_mean, norm_stdev)
    print('calc_instrument_weights')
    return weight


def calc_div_mult_single_period(corr, weights, dm_max=2.5):
    '''
    计算Portfolio variance in correlation space
    且设Limit
    '''
    # TODO: 查背后原理
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


def calc_gross_daily_pnl_dict_for_all_instr(all_instrument_data, all_instruments):
    gross_daily_pnl_dict = {}
    for instrument in all_instruments:
        price = all_instrument_data[instrument]['price']
        point_size = all_instrument_data[instrument]['point_size']
        forecast = all_instrument_data[instrument]['forecast_df']
        pos_target = all_instrument_data[instrument]['position_target']

        position = forecast.mul(pos_target, axis=0) / 10
        position = position.shift(1)
        gross_pnl = calc_gross_pnl(position, price, point_size)
        gross_pnl = gross_pnl.replace(0, np.nan)
        gross_daily_pnl_dict[instrument] = gross_pnl
    print('calc_gross_returns_dict_for_all_instr')
    return gross_daily_pnl_dict


def calc_subsystem_position(instruments, instrument, all_instrument_data, trading_rule_list):
    price_dict = {i: get_daily_price(i) for i in instruments}
    forecast_dict = {k: calculate_forecasts(v) for k, v in price_dict.items()}

    price = get_daily_price(instrument)
    point_size = get_point_size(instrument)

    forecast_df = calculate_forecasts(price)
    pos_target = calc_target_position(price, point_size, capital=1000000, risk_target=0.16)

    position = forecast_df.mul(pos_target, axis=0) / 10
    position = position.shift(1)
    gross_pnl = calc_gross_pnl(position, price, point_size)
    gross_pnl = gross_pnl.replace(0, np.nan)

    dict_of_instr_cost_sr_with_pooling = {}
    for rule in trading_rule_list:

        annual_cost = calc_annual_cost(forecast_dict, instruments, instrument, rule)
        gross_daily_pnl_series = gross_pnl[rule]

        forecast = forecast_df[rule]
        pos_target = pos_target.reindex(forecast.index, method="ffill")
        ##PROBLEM: cost curve calc remains to be checked
        cost_curve = calc_cost(pos_target=pos_target, price=price,
                               point_size=point_size, trading_cost=annual_cost)
        '''
        cost_SR_annual 算出交易成本与gross returns 波动的比例
        越高，说明成本越难以接受
        当annual_cost_SR等于1的时候，就算gross returns 总是赚的，也会被交易成本给消耗掉
        '''

        cost_curve.iloc[:11] = np.nan  # QUESTION: 为什么前11个数都是Nan
        if instrument == 'US10':
            cost_curve.iloc[:13] = np.nan  # QUESTION: 为什么到了US10是前13个数字
        cost_curve_mean = cost_curve.mean()

        gross_daily_pnl_series = gross_daily_pnl_series.replace(0, np.nan)
        gross_daily_pnl_std = gross_daily_pnl_series.std()
        cost_SR_annual = 16 * cost_curve_mean / gross_daily_pnl_std

        turnover = instrument_forecast_turnover(instrument, rule)
        instr_cost_per_turnover = cost_SR_annual / turnover

        average_turnover = average_turnover_across_instruments(all_instrument_data, instruments, rule)
        cost_multiplier = 2
        pooled_cost = instr_cost_per_turnover * average_turnover * cost_multiplier
        dict_of_instr_cost_sr_with_pooling[rule] = pooled_cost

    gross_daily_pnl_dict = calc_gross_daily_pnl_dict_for_all_instr(all_instrument_data, instruments)
    # TODO: 其实这里的步骤就是把第一个循环的内容重复反方向算了一遍而已，完全可以合并
    net_returns_of_rules_for_all_instr_dict = {}
    for instrument in gross_daily_pnl_dict.keys():
        gross_pnl = gross_daily_pnl_dict[instrument]
        net_returns_single_instrument = {}

        # FIXME: dict_of_instr_cost_with_pooling is specific to the target instrument, how can it be applied widely
        for column_name in gross_pnl.columns:
            gross_daily_pnl_std = gross_pnl[column_name].std()
            daily_cost_sr = dict_of_instr_cost_sr_with_pooling[column_name] / 16
            daily_cost = (daily_cost_sr * gross_daily_pnl_std).item()

            net_returns_single_instrument_rule = gross_pnl[column_name] + daily_cost
            net_returns_single_instrument[column_name] = net_returns_single_instrument_rule

        net_returns_single_instrument = pd.DataFrame(net_returns_single_instrument)
        net_returns_of_rules_for_all_instr_dict[instrument] = net_returns_single_instrument

    net_returns_stacked_for_all_instr = single_resampled_set_of_returns(net_returns_of_rules_for_all_instr_dict,
                                                                        frequency='W')
    start_date = net_returns_stacked_for_all_instr.index[0]
    end_date = net_returns_stacked_for_all_instr.index[-1]
    end_list = generate_fit_end_list(start_date, end_date)
    weight_df = pd.DataFrame(
        [calc_forecast_weights(instruments, net_returns_stacked_for_all_instr, end) for end in end_list],
        index=end_list, columns=net_returns_stacked_for_all_instr.columns)

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
    list_of_forecast_df = [all_instrument_data[instrument]['forecast_df'] for instrument in (instruments)]
    list_of_resampled_forecast = [forecast_df.resample('W').last() for forecast_df in list_of_forecast_df]
    pooled_forecast_data = reindex_and_stack_list_of_df(list_of_resampled_forecast)

    pooled_fdm = True
    ew_lookback = 250
    min_periods = 20
    if pooled_fdm == True:
        ew_lookback = ew_lookback * len(instruments)
        min_periods = min_periods * len(instruments)
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

    # FIXME: combined forecast 有问题
    combined_forecast_without_cap = (forecast_weights_for_rules * forecast_df).sum(axis=1) * div_mult.ffill()
    combined_forecast = combined_forecast_without_cap.clip(20, -20)  # QUESTION: 小数点后8位开始对不上，暂时不管
    vol_scalar = calc_volatility_scalar(instrument, all_instrument_data,
                                        annual_perc_vol_target=0.25,
                                        capital=500000)
    vol_scalar = vol_scalar.reindex(universal_index, method="ffill")
    subsystem_position_raw = vol_scalar * combined_forecast / 10.0
    print('calc_subsystem_position')
    return subsystem_position_raw, vol_scalar


def calc_annual_cost(forecast_dict, instruments, instrument, rule):
    forecast_length_weights = calc_forecast_length_weights(forecast_dict)
    annual_trading_cost_per_contract = calc_annual_trading_cost_per_contract(instrument, rule,
                                                                             instruments,
                                                                             forecast_length_weights)
    return annual_trading_cost_per_contract


def calc_forecast_length_weights(forecast_dict):
    # 用历史数据的多少来决定每个instrument的权重
    forecast_length = [len(v) for k, v in forecast_dict.items()]
    total_length = float(sum(forecast_length))
    forecast_length_weights = [forecast_length / total_length for forecast_length in forecast_length]
    return forecast_length_weights


def average_turnover_across_instruments(all_instrument_data, instruments, rule):
    turnovers = {instrument: all_instrument_data[instrument]['turnover_dict'] for instrument in all_instrument_data}
    all_turnovers = [turnovers[instrument][rule] for instrument in (instruments)]
    average_turnover_across_assets = np.nanmean(all_turnovers)
    return average_turnover_across_assets


def calc_gross_pnl(position, price, point_size):
    pnl_in_points = calc_daily_gross_pnl_in_points(positions=position, prices=price)
    pnl = pnl_in_points * point_size
    daily_pnl = pnl.resample("B").sum()
    daily_pnl = daily_pnl.squeeze()
    print('calc_gross_instrument_pnl')
    # FIXME: 鉴于forecast是个两列的df, daily_pnl_gross也是个两列的df
    # 这就有问题了，应该如何理解这两列的实际持仓呢
    # 计算过程中，我们本质上是把每个rule当成了单独的portfolio来算的，所以才有了用position_target直接乘上去
    # 得出的daily_pnl_gross不能是直接相加吧，如果是的话就不合理了
    # 举例，两个forecast 给出了很弱的信号，所以实际持仓都是目标持仓的60%, 如果直接相加的，反而会导致最终持仓到了目标持仓的120%
    return daily_pnl


def calc_daily_gross_pnl_in_points(positions: pd.Series, prices: pd.Series):
    # TODO: 清理，这里为什么还要shift一次，可能会有问题
    '''
    持仓单位为 “手"
    实际持仓再往后调一天，然后乘以价格变化
    因为实际持仓和价格变化都是当天收盘之后算出来的
    所以第一天的实际持仓算出来后，第二天会那么持仓，然后吃满第二天的价格变化
    pnl也会算进第二天里
    需要去算具体金额的盈亏，还得乘以point_size, 也就是比如说一手多少吨
    '''
    pos_series = positions.groupby(positions.index).last()  # 得到当天最后的持仓
    pos_price_series = pd.concat([pos_series, prices], axis=1)
    if len(pos_price_series.columns) == 2:
        pos_price_series.columns = ["positions", "price"]
    pos_price_series = pos_price_series.ffill()
    daily_price_change = pos_price_series.price.diff()
    adjusted_pos_price_series = pos_price_series.loc[:, pos_price_series.columns != 'price'].shift(1)
    daily_pnl_in_points = adjusted_pos_price_series.mul(daily_price_change, axis=0)
    daily_pnl_in_points[daily_pnl_in_points.isna()] = 0.0
    return daily_pnl_in_points
