import numpy as np
import pandas as pd

from refactory.base import calc_gross_pnl, calc_net_pnl, calc_buffered_position, \
    calc_volatility_scalar, calc_position
from refactory.base import calc_position_target
from refactory.combine_forecast import calc_weights_daily, calc_div_mult_daily
from refactory.cost import calc_cost
from refactory.cost_sr import calc_annual_turnover, calc_turnover_weights, calc_weighted_turnover, calc_rule_daily_cost
from refactory.data_source import get_instrument_info, get_price, get_raw_price
from refactory.forecast import ewmac, rescale_forecast, floor_vol, price_vol
from refactory.portfolio_weights import calc_portfolio_weights
from refactory.subsystem_turnover import calc_subsystem_turnover

risk_target = 0.16
# instruments = ["CORN", "SOFR", "SP500_micro", 'US10']
instruments = ["US10", "SOFR", "CORN", "SP500_micro"]

info_ = get_instrument_info().loc[instruments]
size_ = info_['point_size']


def calc_forecasts(price):
    raw_ewmac32 = ewmac(price, 32, 128, 1)
    ewmac32 = rescale_forecast(raw_ewmac32 / floor_vol(price_vol(price)))
    raw_ewmac8 = ewmac(price, 8, 32, 1)
    ewmac8 = rescale_forecast(raw_ewmac8 / floor_vol(price_vol(price)))
    forecast_df = pd.DataFrame({'ewmac32': ewmac32, 'ewmac8': ewmac8})
    return forecast_df


price_list = (get_price(i) for i in instruments)
price_ = pd.concat(price_list, keys=instruments, names=['instrument', 'datetime'])

raw_price_list = (get_raw_price(i) for i in instruments)
raw_price_ = pd.concat(raw_price_list, keys=instruments, names=['instrument', 'datetime'])

forecast_list = (calc_forecasts(price_.loc[i]) for i in instruments)
forecast_ = pd.concat(forecast_list, keys=instruments, names=['instrument', 'datetime'])

target_list = (calc_position_target(price_.loc[i], size_.loc[i], capital=1000000, annual_risk_target=risk_target)
               for i in instruments)
target_ = pd.concat(target_list, keys=instruments, names=['instrument', 'datetime'])

position_list = [calc_position(forecast_.loc[i], target_.loc[i]) for i in instruments]
position_ = pd.concat(position_list, keys=instruments, names=['instrument', 'datetime'])

gross_list = (calc_gross_pnl(position_.loc[i], price_.loc[i], size_.loc[i]) for i in instruments)
gross_ = pd.concat(gross_list, keys=instruments, names=['instrument', 'datetime'])

turnover_func = lambda x: x.reset_index(level='instrument', drop=True).apply(calc_annual_turnover)
turnover_ = forecast_.groupby(level='instrument').apply(turnover_func)
average_turnover_ = turnover_.apply(np.nanmean)

turnover_weight = calc_turnover_weights(forecast_)
weighted_turnover_ = turnover_.apply(lambda x: calc_weighted_turnover(turnover_weight, x))

daily_cost_list = [
    pd.DataFrame({r: calc_rule_daily_cost(weighted_turnover_[r], price_.loc[i], target_.loc[i], info_.loc[i])
                  for r in (weighted_turnover_.index.to_list())})
    for i in instruments]
cost_ = pd.concat(daily_cost_list, keys=instruments, names=['instruments', 'datetime'])

net_list = [pd.DataFrame({r: calc_net_pnl(gross_.loc[i][r], cost_.loc[i][r])
                          for r in gross_.loc[i].columns})
            for i in instruments]
net_ = pd.concat(net_list, keys=instruments, names=['instrument', 'datetime'])

print('calculate pnl for instrument and rule')

forecast_weights = calc_weights_daily(net_)
forcast_div_mult = calc_div_mult_daily(forecast_weights, forecast_)

subsystem_positions_dict = {}
gross_dict = {}
costs_dict = {}
for instrument in instruments:
    info = info_.loc[instrument]
    point_size = size_[instrument]

    price = price_.loc[instrument]
    raw_price = raw_price_.loc[instrument]
    forecast = forecast_.loc[instrument]

    combined_forecast = ((forecast_weights * forecast).sum(axis=1) * forcast_div_mult).clip(20, -20)
    vol_scalar = calc_volatility_scalar(raw_price, price, point_size, 500000, risk_target)
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

print('END')

_info = info_.loc['US10']
_raw_price = raw_price_.loc['US10']
_price = price_.loc['US10']
_price_pnl = price_.loc['US10'].diff()

_forecast = forecast_.loc['US10']['ewmac32']

_position_target = target_.loc['US10']
_position = position_.loc['US10']['ewmac32']
_gross = gross_.loc['US10']['ewmac32']
_cost = cost_.loc['US10']['ewmac32']

_net = net_.loc['US10']['ewmac32']

print(_net)
