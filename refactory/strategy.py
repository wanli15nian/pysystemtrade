import numpy as np
import pandas as pd
from copy import copy

from refactory.functions import calc_subsystem_turnover, \
    calc_subsystem_position, calc_pnl_across_subsystem_for_indiv_instr
from refactory.prepare_all_instr_data import prepare_all_instr_data
from refactory.utils import optimisation, single_resampled_set_of_returns

trading_instruments = ["CORN", "SOFR", "SP500_micro", 'US10']

trading_rule_list = ['ewmac32', 'ewmac8']

all_instrument_data = prepare_all_instr_data(trading_instruments, trading_rule_list)

net_instr_pnl_for_all_instr = {}
dict_of_gross_pandl = {}
dict_of_costs = {}
dict_of_turnover = {}
for instrument in trading_instruments:
    net_pnl, gross_instr_pnl, costs = calc_pnl_across_subsystem_for_indiv_instr(trading_instruments, instrument,
                                                                                all_instrument_data, trading_rule_list)
    subsystem_turnover = calc_subsystem_turnover(trading_instruments, instrument, all_instrument_data,
                                                 trading_rule_list)
    net_instr_pnl_for_all_instr[instrument] = net_pnl
    dict_of_gross_pandl[instrument] = gross_instr_pnl
    dict_of_costs[instrument] = costs
    dict_of_turnover[instrument] = subsystem_turnover

gross_pnl_df = pd.DataFrame(dict_of_gross_pandl)
gross_pnl_sum = gross_pnl_df.sum(axis=1)

cost_df = pd.DataFrame(dict_of_costs)
cost_sum = cost_df.sum(axis=1)

net_PNL = gross_pnl_sum.add(cost_sum, fill_value=0).resample('B').sum()


def process_list_of_data(data):  # Rename the columns
    resampled_data = data.resample('1B').sum()
    resampled_data[resampled_data == 0.0] = np.nan
    return resampled_data


gross_pnl = process_list_of_data(data=gross_pnl_df)
costs = process_list_of_data(data=cost_df)

# turnover_as_list = [
#     calc_subsystem_turnover(trading_instruments, instrument_code, all_instrument_data, trading_rule_list) for
#     instrument_code
#     in trading_instruments]
# turnover_as_dict = dict(
#     [(instrument_code, turnover) for (instrument_code, turnover) in zip(trading_instruments, turnover_as_list)])
turnovers = {'asset': dict_of_turnover}

'''
df_of_gross_pandl.replace(0.0, np.nan) 后就是需要的gross curve
df_of_costs resample方式不同的"relevant curve", sum 都是一样的
'''

# SR 的Index 问题还是没有处理好，源代码为resample("B"), 现为很奇怪的resample
SR_dict = {}
for instrument in trading_instruments:
    cost_curve = cost_df[instrument]
    gross_pandl = gross_pnl_df[instrument]
    daily_returns = cost_curve.mean()
    daily_std = gross_pandl.std()
    annual_SR = 16 * daily_returns / daily_std
    SR_dict[instrument] = annual_SR

net_return_as_dict = {}
for instrument in trading_instruments:
    daily_gross_returns_for_asset = gross_pnl_df[instrument]
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
