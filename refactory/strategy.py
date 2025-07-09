import numpy as np
import pandas as pd

from refactory.combine_forecast import calc_weights_and_multiplier
from refactory.cost import calc_cost
from refactory.cost_sr import calc_annual_turnover, calc_turnover_weights, calc_weighted_turnover, calc_cost_sr
from refactory.data_source import get_instrument_info, get_daily_price, get_raw_price
from refactory.forecast import ewmac, rescale_forecast, floor_vol, price_vol
from refactory.portfolio_weights import calc_portfolio_weights
from refactory.position_pnl import calc_gross_pnl, calc_net_pnl, calc_position, calc_buffered_position, \
    calc_volatility_scalar1
from refactory.position_pnl import calc_position_target
from refactory.subsystem_turnover import calc_subsystem_turnover

instruments = ["CORN", "SOFR", "SP500_micro", 'US10']

info_ = get_instrument_info().loc[instruments]
size_ = info_['point_size']

price_list = (get_daily_price(i) for i in instruments)
price_ = pd.concat(price_list, keys=instruments, names=['instrument', 'datetime'])

raw_price_list = (get_raw_price(i) for i in instruments)
raw_price_ = pd.concat(raw_price_list, keys=instruments, names=['instrument', 'datetime'])


def calc_forecasts(price):
    raw_ewmac32 = ewmac(price, 32, 128, 1)
    ewmac32 = rescale_forecast(raw_ewmac32 / floor_vol(price_vol(price)))
    raw_ewmac8 = ewmac(price, 8, 32, 1)
    ewmac8 = rescale_forecast(raw_ewmac8 / floor_vol(price_vol(price)))
    forecast_df = pd.DataFrame({'ewmac32': ewmac32, 'ewmac8': ewmac8})
    return forecast_df


forecast_list = (calc_forecasts(price_.loc[i]) for i in instruments)
forecast_ = pd.concat(forecast_list, keys=instruments, names=['instrument', 'datetime'])

target_list = (calc_position_target(price_.loc[i], size_.loc[i], capital=1000000, annual_risk_target=0.16)
               for i in instruments)
target_ = pd.concat(target_list, keys=instruments, names=['instrument', 'datetime'])


def calc_gross(forecast, pos_target, price, point_size):
    position = calc_position(forecast, pos_target)
    return calc_gross_pnl(position, price, point_size)


gross_list = (calc_gross(forecast_.loc[i], target_.loc[i], price_.loc[i], size_.loc[i]) for i in instruments)
gross_ = pd.concat(gross_list, keys=instruments, names=['instrument', 'datetime'])

turnover_func = lambda x: x.reset_index(level='instrument', drop=True).apply(calc_annual_turnover)
turnover_ = forecast_.groupby(level='instrument').apply(turnover_func)
average_turnover_ = turnover_.apply(np.nanmean)

turnover_weight = calc_turnover_weights(forecast_)
weighted_turnover_ = turnover_.apply(lambda x: calc_weighted_turnover(turnover_weight, x))


def calc_cost_sr_rules(turnover, average_turnover_, weighted_turnover_, gross, price, position_target, info):
    rules = gross.columns.to_list()
    cost_SR_dict = {rule: calc_cost_sr(turnover[rule], average_turnover_[rule], weighted_turnover_[rule], gross[rule],
                                       price, position_target, info) for rule in rules}
    return cost_SR_dict


cost_sr_list = (
    calc_cost_sr_rules(turnover_.loc[i], average_turnover_, weighted_turnover_, gross_.loc[i], price_.loc[i],
                       target_.loc[i], info_.loc[i])
    for i in instruments)
cost_sr_ = pd.DataFrame(cost_sr_list, index=instruments, columns=gross_.columns)


def calc_net_pnl_rules(gross_pnl, cost_SR_dict):
    return pd.DataFrame({
        column_name: calc_net_pnl(gross_pnl[column_name], cost_SR_dict[column_name])
        for column_name in gross_pnl.columns
    })


net_list = [calc_net_pnl_rules(gross_.loc[i], cost_sr_.loc[i]) for i in instruments]
net_ = pd.concat(net_list, keys=instruments, names=['instrument', 'datetime'])

print('calculate pnl for instrument and rule')

forecast_weights, diversify_multiplier = calc_weights_and_multiplier(forecast_, net_)

subsystem_positions_dict = {}
gross_dict = {}
costs_dict = {}
for instrument in instruments:
    info = info_.loc[instrument]
    point_size = size_[instrument]

    price = price_.loc[instrument]
    raw_price = raw_price_.loc[instrument]
    forecast = forecast_.loc[instrument]

    combined_forecast = ((forecast_weights * forecast).sum(axis=1) * diversify_multiplier).clip(20, -20)
    # FIXME: 为何前面的risk_target是0.16，这里却用0.25?,risk_target应该作为一个常数在最前面设置。
    vol_scalar = calc_volatility_scalar1(raw_price, price, point_size, 500000, 0.25)

    subsystem_position_raw = vol_scalar * combined_forecast / 10.0
    subsystem_position_buffered = calc_buffered_position(subsystem_position_raw, vol_scalar, 0.10)
    position = subsystem_position_buffered.shift(1)
    gross_pnl = calc_gross_pnl(position, price, point_size)
    normalised_costs = calc_cost(position, price, info)

    gross_dict[instrument] = gross_pnl
    costs_dict[instrument] = normalised_costs
    subsystem_positions_dict[instrument] = position
    print('calc_pnl_across_subsytem_for_indiv_instr')

subsystem_positions = pd.DataFrame(subsystem_positions_dict).ffill()
gross_pnl_df = pd.DataFrame(gross_dict)
cost_df = pd.DataFrame(costs_dict)

subsystem_turnover_ = {
    i: calc_subsystem_turnover(subsystem_positions[i], raw_price_.loc[i], price_.loc[i], size_.loc[i])
    for i in instruments}

net_return_raw = pd.DataFrame({inst: gross_pnl_df[inst] + cost_df[inst].mean() for inst in instruments})
portfolio_weights = calc_portfolio_weights(net_return_raw, subsystem_positions)
print(portfolio_weights)

print('END')
