import numpy as np
import pandas as pd
from copy import copy

from refactory.Fill import Fill
from refactory.apply_buffer_to_position import calc_buffered_pos_given_raw_pos
from refactory.functions import process_list_of_data, calc_subsystem_turnover, \
    calc_gross_instr_pnl, pseudo_fills_for_year, calc_cost_instr_currency_for_a_fill, \
    calc_gross_daily_pnl_dict_for_all_instr, calc_annual_trading_cost_per_contract, calc_cost, generate_fit_end_list, \
    calc_forecast_weights, reindex_and_stack_list_of_df, calc_div_mult_single_period
from refactory.prepare_all_instr_data import prepare_all_instr_data
from refactory.utils import optimisation, single_resampled_set_of_returns, calc_volatility_scalar


def calc_subsystem_position(instruments, instrument_code, all_instrument_data, trading_rule_list):
    all_instruments = [instrument for instrument in all_instrument_data.keys()]

    turnovers = {instrument: all_instrument_data[instrument]['turnover_dict'] for instrument in all_instrument_data}
    gross_daily_pnl_dict = calc_gross_daily_pnl_dict_for_all_instr(all_instrument_data, all_instruments)
    # 用历史数据的多少来决定每个instrument的权重
    forecast_length = [len(all_instrument_data[instrument]['forecast_df']) for instrument in all_instruments]
    total_length = float(sum(forecast_length))
    forecast_length_weights = [forecast_length / total_length for forecast_length in forecast_length]
    dict_of_instr_cost_sr = {}
    for instrument in all_instruments:
        cost_SR_dict = {}
        price = all_instrument_data[instrument]['price']
        point_size = all_instrument_data[instrument]['point_size']
        pos_target = all_instrument_data[instrument]['position_target']
        for trading_rule in trading_rule_list:
            forecast = all_instrument_data[instrument]['forecast_df'][trading_rule]
            pos_target = pos_target.reindex(forecast.index, method="ffill")

            # Annual trading cost is calculated using pooled instruments, hence "all_instruments" is passed
            # Trading cost is the sum of holding and transaction cost
            annual_trading_cost_per_contract = calc_annual_trading_cost_per_contract(instrument, trading_rule,
                                                                                     all_instruments,
                                                                                     forecast_length_weights)

            gross_daily_pnl_series = gross_daily_pnl_dict[instrument][trading_rule]

            ##PROBLEM: cost curve calc remains to be checked
            cost_curve = calc_cost(pos_target=pos_target, price=price,
                                   point_size=point_size, trading_cost=annual_trading_cost_per_contract)
            '''
            annual_cost_SR 算出交易成本与gross returns 波动的比例
            越高，说明成本越难以接受
            当annual_cost_SR等于1的时候，就算gross returns 总是赚的，也会被交易成本给消耗掉
            '''

            cost_curve.iloc[:11] = np.nan  # QUESTION: 为什么前11个数都是Nan
            if instrument == 'US10':
                cost_curve.iloc[:13] = np.nan  # QUESTION: 为什么到了US10是前13个数字
            cost_curve_mean = cost_curve.mean()

            gross_daily_pnl_series = gross_daily_pnl_series.replace(0, np.nan)
            gross_daily_pnl_std = gross_daily_pnl_series.std()
            annual_cost_SR = 16 * cost_curve_mean / gross_daily_pnl_std
            cost_SR_dict[trading_rule] = annual_cost_SR
        dict_of_instr_cost_sr[instrument] = cost_SR_dict

    # FIXME: 首先这个instr_cost_per_turnover 算的就很奇怪，毕竟分子并不是真正的cost, 而是个比值
    # 其次，cost_multiplier是2，没有解释
    cost_multiplier = 2
    dict_of_instr_cost_sr_with_pooling = {}
    for rule in trading_rule_list:
        turnover = turnovers[instrument_code][rule]
        instr_annual_cost_sr = dict_of_instr_cost_sr[instrument_code][rule]
        instr_cost_per_turnover = instr_annual_cost_sr / turnover

        all_turnovers = [turnovers[instrument][rule] for instrument in all_instruments]
        average_turnover_across_assets = np.nanmean(all_turnovers)

        pooled_cost = instr_cost_per_turnover * average_turnover_across_assets * cost_multiplier
        dict_of_instr_cost_sr_with_pooling[rule] = pooled_cost

    # TODO: 其实这里的步骤就是把第一个循环的内容重复反方向算了一遍而已，完全可以合并
    net_returns_of_rules_for_all_instr_dict = {}
    for instrument in gross_daily_pnl_dict.keys():
        gross_daily_pnl = gross_daily_pnl_dict[instrument]
        net_returns_single_instrument = {}

        # FIXME: dict_of_instr_cost_with_pooling is specific to the target instrument, how can it be applied widely
        for column_name in gross_daily_pnl.columns:
            gross_daily_pnl_std = gross_daily_pnl[column_name].std()
            daily_cost_sr = dict_of_instr_cost_sr_with_pooling[column_name] / 16
            daily_cost = (daily_cost_sr * gross_daily_pnl_std).item()

            net_returns_single_instrument_rule = gross_daily_pnl[column_name] + daily_cost
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
    universal_index = all_instrument_data[instrument_code]['price'].index
    column_num = len(weight_df.columns)
    initial_weight = pd.DataFrame({col: 1 / column_num for col in weight_df.columns}, index=[start_date])
    weight_df = pd.concat([initial_weight, weight_df], axis=0)

    # 把按年的Index ffill成按天的Index
    weight_df = weight_df.reindex(universal_index, method='ffill').fillna(1 / column_num)
    daily_forecast_weights_resampled_unsmoothed = weight_df.resample('1B').mean()
    forecast_weights_for_rules = daily_forecast_weights_resampled_unsmoothed.ewm(span=125).mean()
    # 跳过一个weight normalisation to 1 的函数
    list_of_forecast_df = [all_instrument_data[instrument]['forecast_df'] for instrument in all_instruments]
    list_of_resampled_forecast = [forecast_df.resample('W').last() for forecast_df in list_of_forecast_df]
    pooled_forecast_data = reindex_and_stack_list_of_df(list_of_resampled_forecast)

    pooled_fdm = True
    ew_lookback = 250
    min_periods = 20
    if pooled_fdm == True:
        ew_lookback = ew_lookback * len(all_instruments)
        min_periods = min_periods * len(all_instruments)
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
    instrument_forecast = all_instrument_data[instrument_code]['forecast_df']
    combined_forecast_without_cap = (forecast_weights_for_rules * instrument_forecast).sum(axis=1) * div_mult.ffill()
    combined_forecast = combined_forecast_without_cap.clip(20, -20)  # QUESTION: 小数点后8位开始对不上，暂时不管
    vol_scalar = calc_volatility_scalar(instrument_code, all_instrument_data,
                                        annual_perc_vol_target=0.25,
                                        capital=500000)
    vol_scalar = vol_scalar.reindex(universal_index, method="ffill")
    subsystem_position_raw = vol_scalar * combined_forecast / 10.0
    print('calc_subsystem_position')
    return subsystem_position_raw, vol_scalar


def calc_pnl_across_subsystem_for_indiv_instr(instruments, instrument_code, all_instrument_data, trading_rule_list):
    price = all_instrument_data[instrument_code]['price']
    rolls_per_year = all_instrument_data[instrument_code]['rolls_per_year']
    raw_costs = all_instrument_data[instrument_code]['raw_costs']
    value_per_point = all_instrument_data[instrument_code]['value_per_point']

    position_raw, vol_scalar = calc_subsystem_position(instruments, instrument_code, all_instrument_data,
                                                       trading_rule_list)
    position_buffered = calc_buffered_pos_given_raw_pos(position_raw, vol_scalar, 0.10)

    adjusted_pos_buffered = position_buffered.shift(1)
    gross_pnl_daily = calc_gross_instr_pnl(instrument_code, position_buffered, price)

    list_of_years = list(set([int(idx.year) for idx in adjusted_pos_buffered.index]))
    list_of_years.sort()
    fills_by_year = [pseudo_fills_for_year(year, rolls_per_year, price, adjusted_pos_buffered) for year in
                     list_of_years]
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
    instrument_currency_costs = [-calc_cost_instr_currency_for_a_fill(fill, value_per_point, raw_costs) for fill in
                                 list_of_all_fills]
    date_index = [fill.date for fill in list_of_all_fills]
    costs_as_pd_series = pd.Series(instrument_currency_costs, date_index)
    costs_as_pd_series = costs_as_pd_series.sort_index()
    costs_as_pd_series = costs_as_pd_series.groupby(costs_as_pd_series.index).sum()
    daily_price = price.resample("1B").ffill()
    daily_returns = daily_price.ffill().diff()
    vol_price = daily_returns.rolling(180, min_periods=3).std().ffill()
    final_vol = vol_price.iloc[-1]
    cost_deflator = vol_price / final_vol
    reindexed_deflator = cost_deflator.reindex(costs_as_pd_series.index, method="ffill")
    normalised_costs = reindexed_deflator * costs_as_pd_series
    net_pnl = gross_pnl_daily.add(normalised_costs, fill_value=0).resample('B').sum()
    print('calc_pnl_across_subsytem_for_indiv_instr')
    return net_pnl, gross_pnl_daily, normalised_costs


trading_instruments = ["CORN", "SOFR", "SP500_micro", 'US10']
trading_rule_list = ['ewmac32', 'ewmac8']
all_instrument_data = prepare_all_instr_data(trading_instruments, trading_rule_list)

net_instr_pnl_for_all_instr = {}
dict_of_gross_pandl = {}
dict_of_costs = {}

for instrument in trading_instruments:
    net_pnl, gross_instr_pnl, costs = calc_pnl_across_subsystem_for_indiv_instr(trading_instruments, instrument,
                                                                                all_instrument_data, trading_rule_list)
    net_instr_pnl_for_all_instr[instrument] = net_pnl
    dict_of_gross_pandl[instrument] = gross_instr_pnl
    dict_of_costs[instrument] = costs
df_of_gross_pandl = pd.DataFrame(dict_of_gross_pandl)
summed_gross_pandl = df_of_gross_pandl.sum(axis=1)
df_of_costs = pd.DataFrame(dict_of_costs)
summed_costs = df_of_costs.sum(axis=1)

net_PNL = summed_gross_pandl.add(summed_costs, fill_value=0)
net_PNL = net_PNL.resample('B').sum()

gross_pnl = process_list_of_data(data=df_of_gross_pandl)
costs = process_list_of_data(data=df_of_costs)

turnover_as_list = [
    calc_subsystem_turnover(trading_instruments, instrument_code, all_instrument_data, trading_rule_list) for
    instrument_code
    in trading_instruments]
turnover_as_dict = dict(
    [(instrument_code, turnover) for (instrument_code, turnover) in zip(trading_instruments, turnover_as_list)])
turnovers = {'asset': turnover_as_dict}

'''
df_of_gross_pandl.replace(0.0, np.nan) 后就是需要的gross curve
df_of_costs resample方式不同的"relevant curve", sum 都是一样的
'''

# SR 的Index 问题还是没有处理好，源代码为resample("B"), 现为很奇怪的resample
SR_dict = {}
for instrument in trading_instruments:
    cost_curve = df_of_costs[instrument]
    gross_pandl = df_of_gross_pandl[instrument]
    daily_returns = cost_curve.mean()
    daily_std = gross_pandl.std()
    annual_SR = 16 * daily_returns / daily_std
    SR_dict[instrument] = annual_SR

net_return_as_dict = {}
for instrument in trading_instruments:
    daily_gross_returns_for_asset = df_of_gross_pandl[instrument]
    daily_gross_return_std = daily_gross_returns_for_asset.std()
    daily_asset_sr_cost = SR_dict[instrument] / 16
    daily_returns_cost = daily_gross_return_std * daily_asset_sr_cost
    daily_returns_cost_as_list = [daily_returns_cost] * len(daily_gross_returns_for_asset.index)
    daily_returns_cost_as_ts = pd.Series(daily_returns_cost_as_list, daily_gross_returns_for_asset.index)
    net_returns = daily_gross_returns_for_asset + daily_returns_cost_as_ts
    net_return_as_dict[instrument] = net_returns

net_return_as_df = pd.DataFrame(net_return_as_dict)
net_return_dict = {'asset': net_return_as_df}
net_return = single_resampled_set_of_returns(net_return_dict, 'W')

start = net_return.index[0]
end = net_return.index[-1]

# sample method is INSAMPLE
fit_dates = f'Fit from {start} to {end}, use from {start} to {end}'
# print(calculate_instrument_weights(net_return))

corr = net_return.ewm(span=500000, min_periods=10, ignore_na=True).corr(pairwise=True)

size_of_matrix = len(corr.columns)
corr_matrix_values = (
    corr[corr.index.get_level_values(0) < end]
    .tail(size_of_matrix)
    .values)
corr_matrix_values[corr_matrix_values < 0.0] = 0.0

exponential_mean = net_return.ewm(span=50000, min_periods=5).mean()
matching_index_size = net_return.index[net_return.index < end].size
last_index = matching_index_size - 1
mean = exponential_mean.iloc[last_index]
mean = mean * 365.25 / 7.0  # Number of weeks in a year

exponential_std = net_return.ewm(span=50000, min_periods=5).std()
std = exponential_std.iloc[last_index]
std = std * (365.25 / 7.0) ** 0.5

data_length = len(net_return.index)
frequency = 'W'

# Shrinkage
corr_matrix_values = pd.DataFrame(corr_matrix_values, columns=corr.columns)
new_corr_values = copy(corr_matrix_values.values)
np.fill_diagonal(new_corr_values, np.nan)
avg_corr = np.nanmean(new_corr_values)
instruments_used = corr_matrix_values.columns

size_index = range(len(corr_matrix_values.columns))


def _od(i, j, offdiag, diag):
    if i == j:
        return diag
    else:
        return offdiag


corr_matrix_values_as_list = [
    [_od(i, j, offdiag=avg_corr, diag=1.0) for i in size_index] for j in size_index
]
corr_matrix_without_columns = np.array(corr_matrix_values_as_list)
prior_corr = pd.DataFrame(corr_matrix_without_columns, columns=instruments_used, index=instruments_used)

shrinkage_corr = 0.5
shrunk_corr_without_columns = (shrinkage_corr * prior_corr.values + (1 - shrinkage_corr) * corr_matrix_values.values)
shrunk_corr = pd.DataFrame(shrunk_corr_without_columns, columns=instruments_used, index=instruments_used)

shrinkage_sr = 0.9
target_sr = 0.5
sr_estimates = [asset_mean / asset_stdev for (asset_mean, asset_stdev) in zip(mean, std)]
post_sr_list = [(shrinkage_sr * target_sr) + (1 - shrinkage_sr) * estimatedSR for estimatedSR in sr_estimates]
shrunk_means_values = [asset_sr * asset_stdev for (asset_sr, asset_stdev) in zip(post_sr_list, std)]
shrunk_means = [(asset_name, mean_value) for (asset_name, mean_value) in zip(instruments_used, shrunk_means_values)]

## 这里相当于默认asset 的命名顺序不变，有风险

avg_std = np.nanmean(std)
norm_factor = [asset_stdev / avg_std for asset_stdev in std]
with np.errstate(invalid='ignore'):
    norm_means = [shrunk_means_values[i] / norm_factor[i] for (i, notUsed) in enumerate(shrunk_means)]
    norm_stdev = [std[i] / norm_factor[i] for (i, notUsed) in enumerate(std)]

mean_list = [target_sr * asset_stdev for asset_stdev in norm_stdev]

equalised_mean = {(asset_name, mean) for (asset_name, mean) in zip(instruments_used, mean_list)}
equalised_std = {(asset_name, std) for (asset_name, std) in zip(instruments_used, norm_stdev)}
weights = optimisation(len(instruments_used), corr=shrunk_corr.values, norm_mean=mean_list, norm_stdev=norm_stdev)
weights_dict = {asset_name: weight for (asset_name, weight) in zip(instruments_used, weights)}
## 在这里跳过clean weights 步骤
weight_index = [start]  ## 这里应该是list of starting dates
weights = pd.DataFrame(weights_dict, index=weight_index)

## 在这里跳过add zero


subsystem_positions = []

for instrument_code in instruments_used:
    raw_pos, vol_scalar = calc_subsystem_position(trading_instruments, instrument_code, all_instrument_data,
                                                  trading_rule_list)
    # pos_buffered = calc_buffered_pos_given_raw_pos(raw_pos, vol_scalar, buffer_size=0.1)
    subsystem_positions.append(raw_pos)
subsystem_positions = pd.concat(subsystem_positions, axis=1).ffill()
subsystem_positions.columns = instruments_used

position_or_forecast = subsystem_positions
pdm_ffill = position_or_forecast.ffill()

## Set leading all nan to zero so weights not set to zero
p_or_f_notnan = ~pdm_ffill.isna()
pdm_ffill[p_or_f_notnan.sum(axis=1) == 0] = 0

adj_weights = weights.groupby(level=0).last()
adj_weights = adj_weights.reindex(pdm_ffill.index, method="ffill")
instrument_weights = adj_weights[position_or_forecast.columns]
instrument_weights[np.isnan(pdm_ffill)] = 0.0
daily_unsmoothed_instr_weights = instrument_weights.resample('1B').mean()

smooth_weighting = 125
smoothed_instr_weights = daily_unsmoothed_instr_weights.ewm(span=smooth_weighting).mean()

sum_weights = smoothed_instr_weights.sum(axis=1)
zero_rows = sum_weights == 0.0
sum_weights[zero_rows] = 0.0001  ## avoid Inf
weight_multiplier = 1.0 / sum_weights
weight_multiplier_array = np.array([weight_multiplier] * len(smoothed_instr_weights.columns))
weight_values = smoothed_instr_weights.values

normalised_weights_np = weight_multiplier_array.transpose() * weight_values
normalised_weights = pd.DataFrame(normalised_weights_np, columns=smoothed_instr_weights.columns,
                                  index=smoothed_instr_weights.index)

print('END')
