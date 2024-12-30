import datetime

import numpy as np
import pandas as pd

from refactory.Fill import Fill
from refactory.apply_buffer_to_position import apply_buffer
from refactory.calculate_forecast import get_capped_forecast
from refactory.data_source import get_point_size, get_roll_parameters, get_daily_price, get_raw_cost_data
from refactory.utils import calculate_mixed_volatility, get_corr_estimator_for_instrument_weight, \
    get_stdev_estimator_for_instrument_weight, get_mean_estimator, optimisation, calculate_weighted_average_with_nans, \
    get_cost_per_trade, single_resampled_set_of_returns, calculate_volatility_scalar
from sysdata.config.configdata import Config
import time
start = time.time()

my_config = Config()
my_config.instruments = ["CORN", "SOFR", "SP500_micro", 'US10']

def get_pos_target_from_risk_target(price, point_size, capital=1000000, risk_target=0.16):
    '''
    根据自行设置的risk target 所计算出的单一品种的目标仓位
    剩余资金的风险暴露应该是0，要不然就是使得整体的风险暴露大于risk target
    '''
    ret_volatility = calculate_mixed_volatility(price.diff(), slow_vol_years=10)
    daily_risk_target = risk_target / (256 ** 0.5)
    daily_cash_vol_target = daily_risk_target * capital
    position_target = daily_cash_vol_target / (ret_volatility * point_size)
    return position_target


def calculate_daily_pnl_in_points_given_pos_prices(positions: pd.Series, prices: pd.Series):
    '''
    实际持仓再往后调一天，然后乘以价格变化
    因为实际持仓和价格变化都是当天收盘之后算出来的
    所以第一天的实际持仓算出来后，第二天会那么持仓，然后吃满第二天的价格变化
    pnl也会算进第二天里
    '''
    pos_series = positions.groupby(positions.index).last()
    both_series = pd.concat([pos_series, prices], axis=1)
    if len(both_series.columns) == 2:
        both_series.columns = ["positions", "price"]
    both_series = both_series.ffill()
    daily_price_change = both_series.price.diff()
    adjusted_both_series = both_series.loc[:, both_series.columns != 'price'].shift(1)
    daily_pnl = adjusted_both_series.mul(daily_price_change, axis=0)
    daily_pnl[daily_pnl.isna()] = 0.0
    return daily_pnl


def calc_cost(pos_target, price, point_size, trading_cost):
    # Actually output in price space to match gross returns
    # These will be annualised figure, make it a small loss every day
    # FIXME: 完全没有看明白这个calc_cost的计算逻辑
    annualised_price_vol_points = calculate_mixed_volatility(price.diff(), slow_vol_years=10)
    sr_cost_as_annualised_figure = (-trading_cost * pos_target * annualised_price_vol_points * 16).bfill()
    period_intervals_in_seconds = sr_cost_as_annualised_figure.index.to_series().diff().dt.total_seconds()
    costs_in_points = sr_cost_as_annualised_figure * period_intervals_in_seconds / (365.25 * 24 * 60 * 60)
    costs = costs_in_points * point_size  # 后续有个fx 的序列，但目前不加

    return costs


def calc_gross_daily_pnl(forecast, point_size, price, position_target):
    '''
    根据品种价格，以及设的波动率目标，有对应的目标仓位
    根据当天的Position target 和对未来的Forecast, 算出来第二天应该有的实际持仓，所以会shift(1)
    '''
    # FIXME: 其次，当forecast信号强烈的时候，是不是意味着实际持仓可以超出Position_target, 从而导致风险暴露超出目标风险暴露
    position = forecast.mul(position_target, axis=0) / 10
    position = position.shift(1)
    # 这边是in points 因为position 是in points 计算的
    pnl_in_points = calculate_daily_pnl_in_points_given_pos_prices(positions=position, prices=price)
    pnl = pnl_in_points * point_size
    daily_pnl_gross = pnl.resample("B").sum()
    #FIXME: 鉴于forecast是个两列的df, daily_pnl_gross也是个两列的df
    # 这就有问题了，应该如何理解这两列的实际持仓呢
    # 计算过程中，我们本质上是把每个rule当成了单独的portfolio来算的，所以才有了用position_target直接乘上去
    # 得出的daily_pnl_gross不能是直接相加吧，如果是的话就不合理了
    # 举例，两个forecast 给出了很弱的信号，所以实际持仓都是目标持仓的60%, 如果直接相加的，反而会导致最终持仓到了目标持仓的120%
    return daily_pnl_gross


def forecast_turnover_for_individual_instrument(instrument_code, rule_name):
    forecast = get_capped_forecast(instrument_code, rule_name)

    average_forecast_for_turnover = 10.0
    y = average_forecast_for_turnover
    daily_forecast = forecast.resample("1B").last()
    daily_y = pd.Series(np.full(daily_forecast.shape[0], float(y)), daily_forecast.index)
    x_normalised_for_y = daily_forecast / daily_y.ffill()
    avg_daily = float(x_normalised_for_y.diff().abs().mean())
    annual_turnover_for_forecast = avg_daily * 256
    return annual_turnover_for_forecast


def calc_trading_cost(instrument_code, rule_name, pooled_instruments, weights):
    # 单次交易成本，包括slippage和commission
    cost_per_trade = get_cost_per_trade(instrument_code)

    # transaction cost

    # 获取交易所有instr 年交易频率
    turnovers = [forecast_turnover_for_individual_instrument(instrument_code, rule_name)
                 for instrument_code in pooled_instruments]

    weighted_avg_turnover = calculate_weighted_average_with_nans(weights, turnovers)
    transaction_cost = cost_per_trade * weighted_avg_turnover
    '''
    交易频率和历史数据线性相关，历史数据越多，交易频率可以越高
    反之，即使根据Forecast计算应该频繁交易，但历史数据不足会导致交易频率受限
    更多的是一种自我设限的操作
    '''

    # holding cost
    roll_parameters = get_roll_parameters(instrument_code)
    hold_turnovers = roll_parameters.rolls_per_year_in_hold_cycle() * 2.0
    holding_cost = hold_turnovers * cost_per_trade

    '''
    期货合约进行换仓的时候，有卖出旧合约和卖出新合约两个操作，所以乘2
    然后乘上单次交易成本
    所以属于持仓成本
    '''

    trading_cost = transaction_cost + holding_cost
    return trading_cost


def generate_fit_end_list(start_date, end_date):
    start_dates_per_period = pd.date_range(end_date, start_date, freq='-365D').to_list()
    start_dates_per_period.reverse()
    end_list = start_dates_per_period[1:-1]
    return end_list



def calculate_forecast_weights(pnl_df, fit_end):
    number = len(pnl_df.columns)
    span = len(my_config.instruments) * 50000
    min_periods_corr = len(my_config.instruments) * 10
    min_periods = len(my_config.instruments) * 5
    norm_stdev, norm_factor, stdev_list = get_stdev_estimator_for_instrument_weight(pnl_df, fit_end, span, min_periods)
    mean_list = get_mean_estimator(pnl_df, fit_end, span, min_periods)
    norm_mean = [a / b for a, b in zip(mean_list, norm_factor)]
    corr = get_corr_estimator_for_instrument_weight(pnl_df, fit_end, span, min_periods_corr)  # Corr CLEARED
    weight = optimisation(number, corr, norm_mean, norm_stdev)
    return weight


def combine_instrument_pnl_df(weekly_ret):
    from itertools import chain
    all_indices_flattened = list(chain.from_iterable(data_item.index for data_item in weekly_ret))
    common_unique_index = sorted(set(all_indices_flattened))
    data_reindexed = [data_item.reindex(common_unique_index) for data_item in weekly_ret]
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

    norm_stdev, _ = get_stdev_estimator_for_instrument_weight(weekly_ret, fit_end, span, min_periods)
    norm_mean = [0.5 * asset_stdev for asset_stdev in norm_stdev]
    corr = get_corr_estimator_for_instrument_weight(weekly_ret, fit_end, span, min_periods)

    weight = optimisation(number, corr, norm_mean, norm_stdev)
    return weight


def calc_gross_instr_pnl(instrument, position_buffered, price):
    position_buffered = position_buffered.shift(1)
    pnl_in_points = calculate_daily_pnl_in_points_given_pos_prices(positions=position_buffered, prices=price)
    point_size = get_point_size(instrument)
    pnl_in_ccy = pnl_in_points * point_size
    gross_pnl_daily = pnl_in_ccy.resample("B").sum()
    return gross_pnl_daily


def div_mult_single_period(corr, weights, dm_max=2.5):
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


def get_turnover_for_list_of_rules(instrument_list, trading_rule_list):
    instrument_turnover_dict = dict()

    for instrument in instrument_list:
        instrument_turnover_dict[instrument] = {
            rule_name: forecast_turnover_for_individual_instrument(instrument, rule_name)
            for rule_name in trading_rule_list
        }

    return instrument_turnover_dict


def prepare_all_instr_data(all_instruments, trading_rule_list):
    all_instrument_data = {}
    for instrument in all_instruments:
        individual_instr_data = {}
        forecast_df = {}
        for rule_name in trading_rule_list:
            forecast = get_capped_forecast(instrument, rule_name)
            forecast_df[rule_name] = forecast
        forecast_df = pd.DataFrame(forecast_df)
        individual_instr_data['forecast_df'] = forecast_df

        turnover_dict = {}
        for rule_name in trading_rule_list:
            turnover = forecast_turnover_for_individual_instrument(instrument, rule_name)
            turnover_dict[rule_name] = turnover
        individual_instr_data['turnover_dict'] = turnover_dict


        price = get_daily_price(instrument)
        individual_instr_data['price'] = price

        point_size = get_point_size(instrument)
        individual_instr_data['point_size'] = point_size

        position_target = get_pos_target_from_risk_target(price, point_size, capital=1000000, risk_target=0.16)
        individual_instr_data['position_target'] = position_target

        all_instrument_data[instrument] = individual_instr_data
    return all_instrument_data

def process_instrument_pnl(instrument_code):
    all_instruments = my_config.instruments
    trading_rule_list = ['ewmac32', 'ewmac8']
    all_instrument_data = prepare_all_instr_data(all_instruments, trading_rule_list)

    turnovers = {instrument: all_instrument_data[instrument]['turnover_dict'] for instrument in all_instrument_data}
    gross_returns_dict = calc_gross_returns_dict_for_all_instr(all_instrument_data, all_instruments)

    # 用历史数据的多少来决定每个instrument的权重
    forecast_length = [len(all_instrument_data[instrument]['forecast_df']) for instrument in all_instruments]
    total_length = float(sum(forecast_length))
    weights = [forecast_length / total_length for forecast_length in forecast_length]

    dict_of_costs_gross_returns_ratio = {}
    for instrument in all_instruments:
        SR_dict = {}
        price = all_instrument_data[instrument]['price']
        point_size = all_instrument_data[instrument]['point_size']
        pos_target = all_instrument_data[instrument]['position_target']
        for trading_rule in trading_rule_list:
            forecast = all_instrument_data[instrument]['forecast_df'][trading_rule]
            pos_target = pos_target.reindex(forecast.index, method="ffill")
            annual_trading_cost = calc_trading_cost(instrument, trading_rule, all_instruments, weights)
            gross_returns_series = calc_gross_daily_pnl(forecast=forecast, point_size=point_size,
                                                        price=price, position_target=pos_target)
            cost_curve = calc_cost(pos_target=pos_target, price=price,
                                   point_size=point_size, trading_cost=annual_trading_cost)
            '''
            annual_cost_SR 算出交易成本与gross returns 波动的比例
            越高，说明成本越难以接受
            当annual_cost_SR等于1的时候，就算gross returns 总是赚的，也会被交易成本给消耗掉
            '''

            cost_curve.iloc[:11] = np.nan  # QUESTION: 为什么前11个数都是Nan
            if instrument == 'US10':
                cost_curve.iloc[:13] = np.nan  # QUESTION: 为什么到了US10是前13个数字
            cost_curve_mean = cost_curve.mean()
            gross_returns_series = gross_returns_series.replace(0, np.nan)
            gross_returns_std = gross_returns_series.std()
            annual_cost_SR = 16 * cost_curve_mean / gross_returns_std
            SR_dict[trading_rule] = annual_cost_SR
        dict_of_costs_gross_returns_ratio[instrument] = SR_dict

    #FIXME: 首先这个cost_per_turnover_this_asset 算的就很奇怪，毕竟分子并不是真正的cost, 而是个比值
    # 其次，cost_multiplier是2，没有解释
    cost_multiplier = 2
    dict_of_sr_costs = {}
    for rule in trading_rule_list:
        turnover = turnovers[instrument_code][rule]
        costs_gross_returns_ratio = dict_of_costs_gross_returns_ratio[instrument_code][rule]
        cost_per_turnover_this_asset = costs_gross_returns_ratio / turnover

        all_turnovers = [turnovers[instrument][rule] for instrument in all_instruments]
        average_turnover_across_assets = np.nanmean(all_turnovers)

        pooled_cost = cost_per_turnover_this_asset * average_turnover_across_assets * cost_multiplier
        dict_of_sr_costs[rule] = pooled_cost

    #TODO: 其实这里的步骤就是把第一个循环的内容重复反方向算了一遍而已，完全可以合并
    net_returns_dict = {}
    for instrument in gross_returns_dict.keys():
        gross_returns = gross_returns_dict[instrument]
        net_returns_single_instrument = {}
        for column_name in gross_returns.columns:
            gross_returns_daily_std = gross_returns[column_name].std()
            daily_sr_cost = dict_of_sr_costs[column_name] / 16
            daily_returns_cost = (daily_sr_cost * gross_returns_daily_std).item()
            net_returns_single_instrument_rule = gross_returns[column_name] + daily_returns_cost
            net_returns_single_instrument[column_name] = net_returns_single_instrument_rule
        net_returns_single_instrument = pd.DataFrame(net_returns_single_instrument)
        net_returns_dict[instrument] = net_returns_single_instrument  # CLEARED
    net_returns = single_resampled_set_of_returns(net_returns_dict, frequency='W')  # CLEARED

    start_date = net_returns.index[0]
    end_date = net_returns.index[-1]
    end_list = generate_fit_end_list(start_date, end_date)
    weight_df = pd.DataFrame([calculate_forecast_weights(net_returns, end) for end in end_list], index=end_list)
    # To add the initial weight
    column_num = len(weight_df.columns)
    initial_weight = {col: 1/column_num for col in weight_df.columns}
    initial_weight = pd.DataFrame(initial_weight, index=[start_date])
    weight_df = pd.concat([initial_weight, weight_df], axis=0)
    weight_df.columns = net_returns.columns

    weight_df = weight_df.reindex(all_instrument_data[instrument_code]['price'].index, method='ffill')
    weight_df = weight_df.fillna(1 / len(weight_df.columns))
    daily_forecast_weights_fixed_to_forecasts_unsmoothed = weight_df.resample('1B').mean()
    forecast_weights = daily_forecast_weights_fixed_to_forecasts_unsmoothed.ewm(span=125).mean()

    # 跳过一个weight normalisation to 1 的函数
    list_of_forecast = [all_instrument_data[instrument]['forecast_df'] for instrument in all_instruments]

    list_of_resampled_forecast = [forecast_df.resample('W').last() for forecast_df in list_of_forecast]
    pooled_forecast_data = combine_instrument_pnl_df(list_of_resampled_forecast)

    pooled_fdm = True
    ew_lookback = 250
    min_periods = 20

    if pooled_fdm == True:
        ew_lookback = ew_lookback * len(all_instruments)
        min_periods = min_periods * len(all_instruments)

    raw_correlations = pooled_forecast_data.ewm(span=ew_lookback, min_periods=min_periods,
                                                ignore_na=True).corr(pairwise=True)
    size_of_matrix = len(pooled_forecast_data)

    corr_list = []
    for fit_end in end_list:
        corr_matrix_values = (raw_correlations[raw_correlations.index.get_level_values(0) < fit_end]
                              .tail(size_of_matrix)
                              .values)
        corr_matrix_values = corr_matrix_values[-1]
        for corr_value in corr_matrix_values:
            if corr_value < 0:
                corr_value = 0
        corr_list.append(corr_matrix_values)
    # corr_list.insert(0, np.array([0.99, 1]))  # 为了让corr_list的element和end_list对齐，先不加起始默认matrix

    div_mult_vector = []
    for corrmatrix, start_of_period in zip(corr_list, end_list):
        weight_slice = forecast_weights[:start_of_period]
        if weight_slice.shape[0] == 0:
            div_mult_vector.append(1.0)
            continue

        weights_dict = np.array(weight_slice.iloc[-1])
        div_multiplier = div_mult_single_period(corrmatrix, weights_dict)
        div_mult_vector.append(div_multiplier)
    div_mult_df = pd.Series(div_mult_vector, index=end_list)
    div_mult_df_daily = div_mult_df.reindex(forecast_weights.index, method="ffill")
    div_mult_df_daily[div_mult_df_daily.isna()] = 1.0
    div_mult_df_smoothed = div_mult_df_daily.ewm(span=125).mean()

    instrument_forecast = all_instrument_data[instrument_code]['forecast_df']

    combined_forecast_without_cap = (forecast_weights * instrument_forecast).sum(axis=1) * div_mult_df_smoothed.ffill()


    combined_forecast = combined_forecast_without_cap.clip(20, -20)  # QUESTION: 小数点后8位开始对不上，暂时不管
    position_buffered = calc_instr_daily_pnl_from_buffered_pos(instrument_code, combined_forecast)

    raw_costs = get_raw_cost_data(instrument_code)
    price = all_instrument_data[instrument_code]['price']
    value_per_point = get_point_size(instrument_code)
    capital = 500000
    rolls_per_year = get_roll_parameters(instrument_code).rolls_per_year_in_hold_cycle()


    adjusted_pos_buffered = position_buffered.shift(1)
    gross_pnl_daily = calc_gross_instr_pnl(instrument_code, adjusted_pos_buffered, price)

    list_of_years = list(set([int(idx.year) for idx in adjusted_pos_buffered.index]))
    list_of_years.sort()
    fills_by_year = [pseudo_fills_for_year(year, rolls_per_year, price, adjusted_pos_buffered) for year in list_of_years]
    list_of_holding_fills = [item for sublist in fills_by_year for item in sublist]

    trades = adjusted_pos_buffered.diff()
    trades_without_na = trades[~trades.isna()]
    trades_without_zeros = trades_without_na[trades_without_na != 0]
    prices_aligned_to_trades = price.reindex(trades_without_zeros.index, method="ffill")

    trades_as_list = list(trades_without_zeros.values)
    prices_as_list = list(prices_aligned_to_trades.values)
    dates_as_list = list(prices_aligned_to_trades.index)

    list_of_trading_fills = [
        Fill(date, qty, price, price_requires_slippage_adjustment=True)
        for date, qty, price in zip(dates_as_list, trades_as_list, prices_as_list)
    ]

    list_of_all_fills = list_of_trading_fills + list_of_holding_fills

    instrument_currency_costs = [-calc_cost_instr_currency_for_a_fill(fill, value_per_point, raw_costs) for fill in list_of_all_fills]

    end = time.time()
    print('duration is: ', end - start)
    return gross_pnl_daily


def calc_cost_instr_currency_for_a_fill(fill, value_per_point, raw_costs):
    blocks_traded = fill.qty
    price = fill.price
    include_slippage = fill.price_requires_slippage_adjustment
    if include_slippage:
        slippage_costs = abs(blocks_traded) * value_per_point * raw_costs.price_slippage
    else:
        slippage_costs = 0

    '''
    三种Commission cost 的计算方式
    '''
    block_price_multiplier = value_per_point * price
    per_trade_commission = raw_costs.value_of_pertrade_commission
    block_commission = abs(blocks_traded) * raw_costs.value_of_block_commission
    perc_commission = abs(blocks_traded) * block_price_multiplier * raw_costs.percentage_cost

    commission_costs = max([per_trade_commission, block_commission, perc_commission])

    total_cost = slippage_costs + commission_costs
    return total_cost

def pseudo_fills_for_year(year, rolls_per_year, price, adjusted_pos_buffered):
    if rolls_per_year == 0:
        return []

    date_list = generate_equal_dates_within_year(year, rolls_per_year)
    avg_holding_within_a_yr = calc_avg_holding_within_a_year(year, rolls_per_year, adjusted_pos_buffered)
    price_series = price.ffill()
    last_date_with_positions = price.index[-1]
    multiply_roll_costs_by = 1

    ## We multiply the quantity rather than the actual costs, as the later
    ##   cost calculation doesn't distinguish between rolls and other trades

    opening_fills_this_year = [
        Fill(
            date=date,
            qty=qty * multiply_roll_costs_by,
            price=get_row_of_series_before_date(price_series, date),
            price_requires_slippage_adjustment=True,
        )
        for date, qty in zip(date_list, avg_holding_within_a_yr)
        if date <= last_date_with_positions and abs(qty) > 0
    ]

    closing_fills_this_year = [Fill(
            date=fill.date,
            qty=-fill.qty,
            price=fill.price) for fill in opening_fills_this_year]

    fills_this_year = opening_fills_this_year + closing_fills_this_year

    return fills_this_year


def get_row_of_series_before_date(data_series, relevant_date):
    if relevant_date == np.nan:
        data_at_date = data_series.values[-1]
    else:
        matching_index_size = data_series.index[data_series.index < relevant_date].size
        if matching_index_size == 0:
            index_point = None
        else:
            index_point = matching_index_size - 1
        data_at_date = data_series.values[index_point]
    return data_at_date

def calc_avg_holding_within_a_year(year, rolls_per_year, adjusted_pos_buffered):
    first_date = generate_equal_dates_within_year(year - 1, rolls_per_year)[-1]
    subsequent_dates = generate_equal_dates_within_year(year, rolls_per_year)
    all_dates = [first_date] + subsequent_dates
    list_of_average_holdings = []
    for date_index in range(len(subsequent_dates)):
        end_date = all_dates[date_index + 1]
        previous_date = all_dates[date_index]
        avg_holding = adjusted_pos_buffered[previous_date:end_date].abs().mean()
        if np.isnan(avg_holding):
            avg_holding = 0.0
        list_of_average_holdings.append(avg_holding)
    return list_of_average_holdings


def generate_equal_dates_within_year(year, rolls_per_year, false_start_of_year_align=False):
    days_between_periods = int(365 / rolls_per_year)
    start_of_year = datetime.datetime(year, 1, 1)
    if false_start_of_year_align:
        first_date = start_of_year
    else:
        half_period = int(days_between_periods / 2)
        half_period_increment = datetime.timedelta(days=half_period)
        first_date = start_of_year + half_period_increment
    delta_for_each_period = datetime.timedelta(days=days_between_periods)
    all_dates = [first_date + (delta_for_each_period * period_count)
                 for period_count in range(rolls_per_year)]
    return all_dates


def calc_net_returns_dict_for_all_instr(dict_of_sr_costs, gross_returns_dict):
    net_returns_dict = {}
    for instrument in gross_returns_dict.keys():
        gross_returns = gross_returns_dict[instrument]
        net_returns_single_instrument = {}
        for column_name in gross_returns.columns:
            gross_returns_daily_std = gross_returns[column_name].std()
            daily_sr_cost = dict_of_sr_costs[column_name] / 16
            daily_returns_cost = (daily_sr_cost * gross_returns_daily_std).item()
            net_returns_single_instrument_rule = gross_returns[column_name] + daily_returns_cost
            net_returns_single_instrument[column_name] = net_returns_single_instrument_rule
        net_returns_single_instrument = pd.DataFrame(net_returns_single_instrument)
        net_returns_dict[instrument] = net_returns_single_instrument  # CLEARED
    net_returns = single_resampled_set_of_returns(net_returns_dict, frequency='W')  # CLEARED
    return net_returns


def calc_gross_returns_dict_for_all_instr(all_instrument_data, all_instruments):
    gross_returns_dict = {}
    for instrument in all_instruments:
        price = all_instrument_data[instrument]['price']
        point_size = all_instrument_data[instrument]['point_size']
        forecast = all_instrument_data[instrument]['forecast_df']
        pos_target = all_instrument_data[instrument]['position_target']

        gross_returns_series = calc_gross_daily_pnl(forecast=forecast, point_size=point_size, price=price,
                                                    position_target=pos_target)
        gross_returns_single_instrument_df = gross_returns_series.replace(0, np.nan)
        gross_returns_dict[instrument] = gross_returns_single_instrument_df
    return gross_returns_dict


def calc_instr_daily_pnl_from_buffered_pos(instrument_code, combined_forecast):
    volatility_scalar = calculate_volatility_scalar(instrument_code, capital=500000, annual_percentage_volatility_target=0.25)
    volatility_scalar = volatility_scalar.reindex(combined_forecast.index, method="ffill")
    position_raw = volatility_scalar * combined_forecast / 10.0  # 小数点后8位开始对不上，暂时不管
    # position_raw[position_raw < 0] = 0
    # position_raw.fillna(0.0, inplace=True)
    position_buffered = apply_buffer(position_raw, volatility_scalar, 0.10)
    return position_buffered
process_instrument_pnl('CORN')

def main(my_config):
    instruments = my_config.instruments

    pnl_list = [process_instrument_pnl(instrument) for instrument in instruments]
    pnl_df = pd.concat(pnl_list, axis=1)
    pnl_df.columns = instruments

    weight = calculate_instrument_weights(pnl_df)
    print(weight)

    return


if __name__ == '__main__':
    main(my_config)


