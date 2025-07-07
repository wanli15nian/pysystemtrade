import numpy as np
import pandas as pd

from refactory.apply_buffer_to_position import calc_buffered_pos_given_raw_pos
from refactory.cost import calc_cost
from refactory.cost_forecast import annual_forecast_turnover, calculate_weighted_turnover, calc_turnover_weights
from refactory.cost_sr import calc_cost_SR
from refactory.data_source import get_instrument_info, get_daily_price
from refactory.forecast import ewmac, rescale_forecast, floor_vol, price_vol
from refactory.functions import combine_forecast, calc_net_pnl
from refactory.rule_pnl import calc_gross, calc_gross_pnl
from refactory.portfolio_weights import calc_portfolio_weights
from refactory.system_turnover import calc_system_turnover
from refactory.rule_pnl import calc_position_target
from refactory.utils import calc_volatility_scalar

instruments = ["CORN", "SOFR", "SP500_micro", 'US10']
# TODO 不应该用rules列表
rules = ['ewmac32', 'ewmac8']


def calc_forecasts(price):
    raw_ewmac32 = ewmac(price, 32, 128, 1)
    ewmac32 = rescale_forecast(raw_ewmac32 / floor_vol(price_vol(price)))
    raw_ewmac8 = ewmac(price, 8, 32, 1)
    ewmac8 = rescale_forecast(raw_ewmac8 / floor_vol(price_vol(price)))
    forecast_df = pd.DataFrame({'ewmac32': ewmac32, 'ewmac8': ewmac8})
    return forecast_df


info_ = get_instrument_info().loc[instruments]
size_ = info_['point_size']

price_list = (get_daily_price(i) for i in instruments)
price_ = pd.concat(price_list, keys=instruments, names=['instrument', 'datetime'])

forecast_list = (calc_forecasts(price_.loc[i]) for i in instruments)
forecast_ = pd.concat(forecast_list, keys=instruments, names=['instrument', 'datetime'])

target_list = (calc_position_target(price_.loc[i], size_.loc[i], capital=1000000, risk_target=0.16)
               for i in instruments)
target_ = pd.concat(target_list, keys=instruments, names=['instrument', 'datetime'])

gross_list = (calc_gross(forecast_.loc[i], target_.loc[i], price_.loc[i], info_.loc[i]) for i in instruments)
gross_ = pd.concat(gross_list, keys=instruments, names=['instrument', 'datetime'])

turnover_func = lambda x: x.reset_index(level='instrument', drop=True).apply(annual_forecast_turnover)
turnover_ = forecast_.groupby(level='instrument').apply(turnover_func)
average_turnover_ = turnover_.apply(np.nanmean)

turnover_weight = calc_turnover_weights(forecast_)
weighted_turnover_ = turnover_.apply(lambda x: calculate_weighted_turnover(turnover_weight, x))

# TODO: 为何计算cost时要用target_position?
cost_sr_list = (calc_cost_SR(rules, average_turnover_, weighted_turnover_, gross_.loc[i],
                             forecast_.loc[i], price_.loc[i], target_.loc[i], info_.loc[i]) for i in instruments)
cost_sr_ = pd.DataFrame(cost_sr_list, index=instruments, columns=rules)

net_list = [calc_net_pnl(gross_.loc[i], cost_sr_.loc[i]) for i in instruments]
net_ = pd.concat(net_list, keys=instruments, names=['instrument', 'datetime'])

print('calculate pnl for instrument and rule')

subsystem_positions_dict = {}
gross_dict = {}
costs_dict = {}
for instrument in instruments:
    point_size = size_[instrument]
    price = price_.loc[instrument]
    forecast = forecast_.loc[instrument]
    info = info_.loc[instrument]

    combined_forecast = combine_forecast(forecast, forecast_, net_, price)
    vol_scalar = calc_volatility_scalar(price, point_size, 500000, 0.25)
    subsystem_position_raw = vol_scalar * combined_forecast / 10.0

    # FIXME 确认一下是取最后的position
    # subsystem_positions_dict[instrument] = subsystem_position_raw

    position_buffered = calc_buffered_pos_given_raw_pos(subsystem_position_raw, vol_scalar, 0.10)
    position = position_buffered.shift(1)
    gross_pnl = calc_gross_pnl(position, price, point_size)
    normalised_costs = calc_cost(position, price, info)

    gross_dict[instrument] = gross_pnl
    costs_dict[instrument] = normalised_costs
    subsystem_positions_dict[instrument] = position
    print('calc_pnl_across_subsytem_for_indiv_instr')

subsystem_positions = pd.DataFrame(subsystem_positions_dict).ffill()
gross_pnl_df = pd.DataFrame(gross_dict)
cost_df = pd.DataFrame(costs_dict)

system_turnover_ = {i: calc_system_turnover(subsystem_positions[i], price_.loc[i], size_.loc[i]) for i in instruments}

net_return_raw = pd.DataFrame({inst: gross_pnl_df[inst] + cost_df[inst].mean() for inst in instruments})
portfolio_weights = calc_portfolio_weights(net_return_raw, subsystem_positions)
print(portfolio_weights)

print('END')
