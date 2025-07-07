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
from refactory.portfolio_weights import calc_corr_matrix
from refactory.target_volatility import calc_target_position
from refactory.turnover import calc_system_turnover
from refactory.utils import optimisation, calc_volatility_scalar, single_resampled_set_of_returns

instruments = ["CORN", "SOFR", "SP500_micro", 'US10']
rules = ['ewmac32', 'ewmac8']

info_ = get_instrument_info().loc[instruments]
size_ = info_['point_size']

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
    point_size = size_[instrument]
    # spread_cost = info['spread_cost']
    # per_trade = info['per_trade']
    # per_block = info['per_block']
    # percentage = info['percentage']

    price = price_.loc[instrument]
    forecast = forecast_.loc[instrument]

    grouped = net_.groupby(level='instrument')
    net_pnl_all = {ins: group.reset_index(level='instrument', drop=True) for ins, group in grouped}

    combined_forecast = combine_forecast(forecast, forecast_, net_pnl_all, price)
    vol_scalar = calc_volatility_scalar(price, point_size, 500000, 0.25)
    subsystem_position_raw = vol_scalar * combined_forecast / 10.0

    subsystem_positions.append(subsystem_position_raw)
    position_buffered = calc_buffered_pos_given_raw_pos(subsystem_position_raw, vol_scalar, 0.10)
    position = position_buffered.shift(1)
    gross_pnl = calc_gross_pnl(position, price, point_size)

    info = info_.loc[instrument]
    rolls_per_year = int(info['rolls_per_year'])  # TODO: 用【】取会自动转为浮点型，临时方案是强制给转成整型
    raw_costs = get_raw_cost_data(instrument)
    normalised_costs = calc_costs(position, price, rolls_per_year, raw_costs, point_size)
    net_pnl = gross_pnl.add(normalised_costs, fill_value=0).resample('B').sum()

    net_dict[instrument] = net_pnl
    gross_dict[instrument] = gross_pnl
    costs_dict[instrument] = normalised_costs
    print('calc_pnl_across_subsytem_for_indiv_instr')

    turnover_dict[instrument] = calc_system_turnover(subsystem_position_raw, price, point_size)

gross_pnl_df = pd.DataFrame(gross_dict)
cost_df = pd.DataFrame(costs_dict)

subsystem_positions = pd.concat(subsystem_positions, axis=1).ffill()
subsystem_positions.columns = instruments

gross_pnl_sum = gross_pnl_df.sum(axis=1)
cost_sum = cost_df.sum(axis=1)
net_PNL = gross_pnl_sum.add(cost_sum, fill_value=0).resample('B').sum()
print(net_PNL)


# def process_list_of_data(data):  # Rename the columns
#     resampled_data = data.resample('1B').sum()
#     resampled_data[resampled_data == 0.0] = np.nan
#     return resampled_data
# gross_pnl = process_list_of_data(data=gross_pnl_df)
# costs = process_list_of_data(data=cost_df)


def calc_avg_corr(corr_matrix_df):
    new_corr_values = copy(corr_matrix_df.values)
    np.fill_diagonal(new_corr_values, np.nan)
    return np.nanmean(new_corr_values)


def calc_avg_corr_matrix(instruments, avg_corr):
    n = len(instruments)
    corr_matrix = np.full((n, n), avg_corr)  # Fill entire matrix with avg_corr
    np.fill_diagonal(corr_matrix, 1.0)  # Set diagonals to 1.0
    return pd.DataFrame(corr_matrix, index=instruments, columns=instruments)


# net_return_dict = {}
# for instrument1 in instruments:
#     net_return_dict[instrument1] = gross_pnl_df[instrument1] + cost_df[instrument1].mean()
# net_return_df_unresampled = pd.DataFrame(net_return_dict)

net_return_raw = pd.DataFrame({inst: gross_pnl_df[inst] + cost_df[inst].mean() for inst in instruments})
net_return_df = single_resampled_set_of_returns({'asset': net_return_raw}, 'W')

# start = net_return_df.index[0]
end = net_return_df.index[-1]
matching_index_size = net_return_df.index[net_return_df.index < end].size
last_index = matching_index_size - 1
data_length = len(net_return_df.index)
frequency = 'W'

corr_matrix_df = calc_corr_matrix(net_return_df)

exponential_mean = net_return_df.ewm(span=50000, min_periods=5).mean()
mean = exponential_mean.iloc[last_index] * 365.25 / 7.0
annualised_return_mean = mean
exponential_std = net_return_df.ewm(span=50000, min_periods=5).std()
std = exponential_std.iloc[last_index] * (365.25 / 7.0) ** 0.5
annualised_return_std = std
avg_corr = calc_avg_corr(corr_matrix_df)
avg_corr_matrix = calc_avg_corr_matrix(instruments, avg_corr)


def calc_shrunk_corr(avg_corr_matrix, shrinkage_corr=0.5):
    shrunk_corr_without_columns = (
            shrinkage_corr * avg_corr_matrix.values + (1 - shrinkage_corr) * corr_matrix_df.values)
    return pd.DataFrame(shrunk_corr_without_columns, columns=instruments, index=instruments)


shrunk_corr = calc_shrunk_corr(avg_corr_matrix)


def calc_shrunk_means(annualised_return_mean, annualised_return_std, shrinkage_sr=0.9, target_sr=0.5):
    sr_estimates = (annualised_return_mean / annualised_return_std).to_list()
    post_sr_list = [(shrinkage_sr * target_sr) + (1 - shrinkage_sr) * estimatedSR for estimatedSR in sr_estimates]
    shrunk_means_values = (post_sr_list * annualised_return_std).to_list()
    shrunk_means = [(asset_name, mean_value) for (asset_name, mean_value) in zip(instruments, shrunk_means_values)]
    return shrunk_means


target_sr = 0.5
shrunk_means = calc_shrunk_means(annualised_return_mean, annualised_return_std)
norm_std = [annualised_return_std.mean()] * 4
mean_list = [target_sr * asset_stdev for asset_stdev in norm_std]
weights = optimisation(len(instruments), corr=shrunk_corr.values, norm_mean=mean_list, norm_stdev=norm_std)
weights_df = pd.DataFrame({asset_name: weight for (asset_name, weight) in zip(instruments, weights)}, index=[start])

## Set leading all nan to zero so weights not set to zero
subsystem_positions[(~subsystem_positions.isna()).sum(axis=1) == 0] = 0


def calc_smoothed_instr_weights(weights_df, smooth_weighting=125):
    instrument_weights = weights_df.reindex(subsystem_positions.index, method="ffill")
    instrument_weights[np.isnan(subsystem_positions)] = 0.0
    daily_unsmoothed_instr_weights = instrument_weights.resample('1B').mean()
    smoothed_instr_weights = daily_unsmoothed_instr_weights.ewm(span=smooth_weighting).mean()
    return smoothed_instr_weights


smoothed_instr_weights = calc_smoothed_instr_weights(weights_df)


def normalise_weights():
    sum_weights = smoothed_instr_weights.sum(axis=1)
    zero_rows = sum_weights == 0.0
    sum_weights[zero_rows] = 0.0001  ## avoid Inf
    weight_multiplier = 1.0 / sum_weights
    weight_multiplier_array = np.array([weight_multiplier] * len(smoothed_instr_weights.columns))
    normalised_weights_np = weight_multiplier_array.transpose() * smoothed_instr_weights.values
    normalised_weights = pd.DataFrame(normalised_weights_np, columns=smoothed_instr_weights.columns,
                                      index=smoothed_instr_weights.index)

    return normalised_weights


normalised_weights = normalise_weights()

print(normalised_weights)

print('END')
