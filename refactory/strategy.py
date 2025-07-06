import numpy as np
import pandas as pd
from copy import copy

from refactory.apply_buffer_to_position import calc_buffered_pos_given_raw_pos
from refactory.cost import calc_costs
from refactory.cost_forecast import annual_forecast_turnover, calculate_weighted_turnover, calc_turnover_weights
from refactory.cost_sr import calc_cost_SR
from refactory.data_source import get_instrument_info
from refactory.data_util import get_daily_price, get_raw_cost_data
from refactory.forecast import calc_forecasts
from refactory.functions import combine_forecast, calc_net_pnl
from refactory.gross_pnl import calc_gross, calc_gross_pnl
from refactory.target_volatility import calc_target_position
from refactory.turnover import turnover_x_y, calc_average_position
from refactory.utils import optimisation, single_resampled_set_of_returns, calc_volatility_scalar

instruments = ["CORN", "SOFR", "SP500_micro", 'US10']
rules = ['ewmac32', 'ewmac8']

info_ = get_instrument_info().loc[instruments]

price_ = pd.concat((get_daily_price(i)
                    for i in instruments), keys=instruments, names=['instrument', 'datetime'])

forecast_ = pd.concat((calc_forecasts(price_.loc[i])
                       for i in instruments), keys=instruments, names=['instrument', 'datetime'])

target_ = pd.concat((calc_target_position(price_.loc[i], info_.loc[i], capital=1000000, risk_target=0.16)
                     for i in instruments), keys=instruments, names=['instrument', 'datetime'])

gross_ = pd.concat((calc_gross(forecast_.loc[i], target_.loc[i], price_.loc[i], info_.loc[i])
                    for i in instruments), keys=instruments, names=['instrument', 'datetime'])

turnover_ = forecast_.groupby(level='instrument').apply(
    lambda x: x.reset_index(level='instrument', drop=True).apply(annual_forecast_turnover))
average_turnover_ = turnover_.apply(np.nanmean)
turnover_weight = calc_turnover_weights(forecast_)
weighted_turnover_ = turnover_.apply(lambda x: calculate_weighted_turnover(turnover_weight, x))

cost_sr_ = pd.DataFrame([calc_cost_SR(rules, average_turnover_, weighted_turnover_, gross_.loc[i], forecast_.loc[i],
                                      price_.loc[i], target_.loc[i], info_.loc[i])
                         for i in instruments], index=instruments, columns=rules)
net_ = pd.concat([calc_net_pnl(gross_.loc[i], cost_sr_.loc[i])
                  for i in instruments], keys=instruments, names=['instrument', 'datetime'])

net_dict = {}
gross_dict = {}
costs_dict = {}
turnover_dict = {}
subsystem_positions = []

for instrument in instruments:
    info = info_.loc[instrument]
    rolls_per_year = int(info['rolls_per_year'])  # TODO: 用【】取会自动转为浮点型，临时方案是强制给转成整型
    point_size = info['point_size']
    # spread_cost = info['spread_cost']
    # per_trade = info['per_trade']
    # per_block = info['per_block']
    # percentage = info['percentage']

    price = price_.loc[instrument]
    forecast = forecast_.loc[instrument]

    grouped = net_.groupby(level='instrument')
    net_pnl_all = {ins: group.reset_index(level='instrument', drop=True) for ins, group in grouped}

    combined_forecast, universal_index = combine_forecast(forecast, forecast_, net_pnl_all, price)

    vol_scalar = calc_volatility_scalar(price, point_size, 500000, 0.25)
    vol_scalar = vol_scalar.reindex(universal_index, method="ffill")
    subsystem_position_raw = vol_scalar * combined_forecast / 10.0
    print('calc_subsystem_position')

    subsystem_positions.append(subsystem_position_raw)
    position_buffered = calc_buffered_pos_given_raw_pos(subsystem_position_raw, vol_scalar, 0.10)
    position = position_buffered.shift(1)

    gross_pnl = calc_gross_pnl(position, price, point_size)

    raw_costs = get_raw_cost_data(instrument)
    normalised_costs = calc_costs(position, price, rolls_per_year, raw_costs, point_size)
    net_pnl = gross_pnl.add(normalised_costs, fill_value=0).resample('B').sum()

    net_dict[instrument] = net_pnl
    gross_dict[instrument] = gross_pnl
    costs_dict[instrument] = normalised_costs
    print('calc_pnl_across_subsytem_for_indiv_instr')

    daily_price = get_daily_price(instrument)
    average_position_for_turnover = calc_average_position(daily_price, point_size)
    subsystem_turnover = turnover_x_y(subsystem_position_raw, average_position_for_turnover)
    turnover_dict[instrument] = subsystem_turnover #TODO: Check the meaning of turnover
    print('calc_subsystem_turnover')

gross_pnl_df = pd.DataFrame(gross_dict)
cost_df = pd.DataFrame(costs_dict)

subsystem_positions = pd.concat(subsystem_positions, axis=1).ffill()
subsystem_positions.columns = instruments

# gross_pnl_sum = gross_pnl_df.sum(axis=1)
# cost_sum = cost_df.sum(axis=1)
# net_PNL = gross_pnl_sum.add(cost_sum, fill_value=0).resample('B').sum()
# def process_list_of_data(data):  # Rename the columns
#     resampled_data = data.resample('1B').sum()
#     resampled_data[resampled_data == 0.0] = np.nan
#     return resampled_data
# gross_pnl = process_list_of_data(data=gross_pnl_df)
# costs = process_list_of_data(data=cost_df)


# SR 的Index 问题还是没有处理好，源代码为resample("B"), 现为很奇怪的resample
def calc_sr_dict(instruments, cost_df, gross_pnl_df):
    net_return_as_dict = {}
    for instrument in instruments:
        index_used = gross_pnl_df[instrument].index
        daily_returns_cost_as_ts = pd.Series(cost_df[instrument].mean(), index_used)
        net_returns = gross_pnl_df[instrument] + daily_returns_cost_as_ts
        net_return_as_dict[instrument] = net_returns
    return net_return_as_dict
net_returns_dict = calc_sr_dict(instruments, cost_df, gross_pnl_df)

net_return_df_unresampled = pd.DataFrame(net_returns_dict)
net_return_dict = {'asset': net_return_df_unresampled}
net_return_df = single_resampled_set_of_returns(net_return_dict, 'W')

start = net_return_df.index[0]
end = net_return_df.index[-1]

# sample method is INSAMPLE
fit_dates = f'Fit from {start} to {end}, use from {start} to {end}'

corr = net_return_df.ewm(span=500000, min_periods=10, ignore_na=True).corr(pairwise=True)

size_of_matrix = len(corr.columns)
corr_matrix_values = (
    corr[corr.index.get_level_values(0) < end]
    .tail(size_of_matrix)
    .values)
corr_matrix_values[corr_matrix_values < 0.0] = 0.0

exponential_mean = net_return_df.ewm(span=50000, min_periods=5).mean()
matching_index_size = net_return_df.index[net_return_df.index < end].size
last_index = matching_index_size - 1
mean = exponential_mean.iloc[last_index]
mean = mean * 365.25 / 7.0  # Number of weeks in a year

exponential_std = net_return_df.ewm(span=50000, min_periods=5).std()
std = exponential_std.iloc[last_index]
std = std * (365.25 / 7.0) ** 0.5

data_length = len(net_return_df.index)
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
    norm_stdev = [std.iloc[i] / norm_factor[i] for (i, notUsed) in enumerate(std)]

mean_list = [target_sr * asset_stdev for asset_stdev in norm_stdev]

equalised_mean = {(asset_name, mean) for (asset_name, mean) in zip(instruments_used, mean_list)}
equalised_std = {(asset_name, std) for (asset_name, std) in zip(instruments_used, norm_stdev)}
weights = optimisation(len(instruments_used), corr=shrunk_corr.values, norm_mean=mean_list, norm_stdev=norm_stdev)
weights_dict = {asset_name: weight for (asset_name, weight) in zip(instruments_used, weights)}
## 在这里跳过clean weights 步骤
weight_index = [start]  ## 这里应该是list of starting dates
weights = pd.DataFrame(weights_dict, index=weight_index)

pdm_ffill = subsystem_positions.ffill()
## Set leading all nan to zero so weights not set to zero
p_or_f_notnan = ~pdm_ffill.isna()
pdm_ffill[p_or_f_notnan.sum(axis=1) == 0] = 0

adj_weights = weights.groupby(level=0).last()
adj_weights = adj_weights.reindex(pdm_ffill.index, method="ffill")
instrument_weights = adj_weights[subsystem_positions.columns]
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
