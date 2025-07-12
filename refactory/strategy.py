import numpy as np
import pandas as pd

from refactory.base import calc_gross_pnl, calc_net_pnl, calc_position, combine_forecast
from refactory.base import calc_vol_scalar
from refactory.cost_actual import calc_cost_actual
from refactory.cost_estimated import calc_cost_estimated, calc_cost_sr_rule
from refactory.data_source import get_instrument_info, get_price, get_raw_price
from refactory.forecast import ewmac, rescale_forecast, floor_vol, price_vol
from refactory.turnover import calc_turnover, estimate_weighted_turnover, estimate_turnover_annual
from refactory.weights_forecast import calc_forecast_weights, calc_div_mult_daily
from refactory.weights_portfolio import calc_portfolio_weights

# --------------------------------------------------------------------------------------------------------------------


risk_target = 0.16
instruments = ["US10", "SOFR", "CORN", "SP500_micro"]
rules = ['ewmac32', 'ewmac8']


def calc_forecasts(price):
    raw_ewmac32 = ewmac(price, 32, 128, 1)
    ewmac32 = rescale_forecast(raw_ewmac32 / floor_vol(price_vol(price)))
    raw_ewmac8 = ewmac(price, 8, 32, 1)
    ewmac8 = rescale_forecast(raw_ewmac8 / floor_vol(price_vol(price)))
    forecast_df = pd.DataFrame({'ewmac32': ewmac32, 'ewmac8': ewmac8})
    return forecast_df


# -------------------------------------------------------------------------------------------------------------------

def m(func, instruments=instruments):
    # 纵向组装。将func返回的dataset组装成muliIndex的dataset
    return pd.concat((func(i) for i in instruments),
                     keys=instruments,
                     names=['instrument', 'datetime'])


def c(func, instruments=instruments):
    # 横向组装。每个instrument拼成一列
    return pd.DataFrame({i: func(i) for i in instruments})


# -------------------------------------------------------------------------------------------------------------------

info_ = get_instrument_info().loc[instruments]
size_ = info_['point_size']
price_ = m(get_price)
raw_price_ = m(get_raw_price)

print('get price and info')

vol_scalar_ = m(lambda i: calc_vol_scalar(price_.loc[i], size_.loc[i], capital=1000000, risk_target=risk_target))
forecast_rule = m(lambda i: calc_forecasts(price_.loc[i]))
position_rule = m(lambda i: calc_position(forecast_rule.loc[i], vol_scalar_.loc[i]))
gross_rule = m(lambda i: calc_gross_pnl(position_rule.loc[i], price_.loc[i], size_.loc[i]))
turnover_weighted = estimate_weighted_turnover(forecast_rule)
cost_rule = m(lambda i: calc_cost_estimated(price_.loc[i], turnover_weighted, vol_scalar_.loc[i], info_.loc[i]))
net_rule = calc_net_pnl(gross_rule, cost_rule)

print('calculate pnl for instrument and rule')

turnover_full = estimate_turnover_annual(forecast_rule)
turnover_average = turnover_full.mean(axis=0)
cost_sr_rule = m(
    lambda i: calc_cost_sr_rule(gross_rule.loc[i], cost_rule.loc[i], turnover_full.loc[i], turnover_average))


def calc_net_rule(gross, cost_sr):
    gross = gross.replace(0.0, np.nan)
    vol = gross.std()
    cost_daily = cost_sr * (vol / 16)
    return gross + cost_daily


net_rule_fw = m(lambda i: calc_net_rule(gross_rule.loc[i], cost_sr_rule.loc[i]))

forecast_weights = calc_forecast_weights(net_rule_fw)

forcast_div_mult = calc_div_mult_daily(forecast_weights, forecast_rule)
forecast_inst = c(lambda i: combine_forecast(forecast_rule.loc[i], forecast_weights, forcast_div_mult))
position_inst = c(lambda i: calc_position(forecast_inst[i], vol_scalar_.loc[i], buffer_size=0.10))
gross_inst = c(lambda i: calc_gross_pnl(position_inst[i], price_.loc[i], size_.loc[i]))
cost_inst = c(lambda i: calc_cost_actual(position_inst[i], price_.loc[i], info_.loc[i]))

subsystem_turnover_ = {i: calc_turnover(position_inst[i], vol_scalar_.loc[i]) for i in instruments}

# TODO: 为什么是mean？
net_inst = pd.DataFrame({inst: gross_inst[inst] + cost_inst[inst].mean() for inst in instruments})
portfolio_weights = calc_portfolio_weights(net_inst, position_inst)
print(portfolio_weights)

print('calculate weightes for portfolio')

# --------------------------------------------------------------------------------------------------------------------

_info = info_.loc['US10']
_raw_price = raw_price_.loc['US10']
_price = price_.loc['US10']
_price_pnl = price_.loc['US10'].diff()

_vol_scalar = vol_scalar_.loc['US10']
_rule_forecast = forecast_rule.loc['US10']['ewmac32']
_rule_position = position_rule.loc['US10']['ewmac32']
_rule_gross = gross_rule.loc['US10']['ewmac32']
_rule_cost = cost_rule.loc['US10']['ewmac32']
_rule_net = net_rule.loc['US10']['ewmac32']

# print(_rule_net)


# --------------------------------------------------------------------------------------------------------------------


#
# price_list = (get_price(i) for i in instruments)
# price_ = pd.concat(price_list, keys=instruments, names=['instrument', 'datetime'])
#
# raw_price_list = (get_raw_price(i) for i in instruments)
# raw_price_ = pd.concat(raw_price_list, keys=instruments, names=['instrument', 'datetime'])
#
# forecast_list = (calc_forecasts(price_.loc[i]) for i in instruments)
# forecast_ = pd.concat(forecast_list, keys=instruments, names=['instrument', 'datetime'])
#
# target_list = (calc_volatility_scalar(price_.loc[i], size_.loc[i], capital=1000000, annual_risk_target=risk_target)
#                for i in instruments)
# target_ = pd.concat(target_list, keys=instruments, names=['instrument', 'datetime'])
#
# position_list = [calc_position(forecast_.loc[i], target_.loc[i]) for i in instruments]
# position_ = pd.concat(position_list, keys=instruments, names=['instrument', 'datetime'])
#
# gross_list = (calc_gross_pnl(position_.loc[i], price_.loc[i], size_.loc[i]) for i in instruments)
# gross_ = pd.concat(gross_list, keys=instruments, names=['instrument', 'datetime'])
#
# daily_cost_list = [calc_cost_daily(turnover_estimated, price_.loc[i], vol_scalar_.loc[i], info_.loc[i])
#                    for i in instruments]
# cost_ = pd.concat(daily_cost_list, keys=instruments, names=['instruments', 'datetime'])
#
# net_list = [pd.DataFrame({r: calc_net_pnl(gross_.loc[i][r], cost_.loc[i][r])
#                           for r in gross_.loc[i].columns})
#             for i in instruments]
# net_ = pd.concat(net_list, keys=instruments, names=['instrument', 'datetime'])
# subsystem_positions_dict = {}
# gross_dict = {}
# costs_dict = {}
#
# for instrument in instruments:
#     info = info_.loc[instrument]
#     point_size = size_[instrument]
#
#     price = price_.loc[instrument]
#     forecast = forecast_.loc[instrument]
#     vol_scalar = vol_scalar_.loc[instrument]
#
#     combined_forecast = combine_forecast(forecast, forecast_weights, forcast_div_mult)
#     position = calc_position_buffered(combined_forecast, vol_scalar)
#     gross_pnl = calc_gross_pnl(position, price, point_size)
#     normalised_costs = calc_cost(position, price, info)
#
#     gross_dict[instrument] = gross_pnl
#     costs_dict[instrument] = normalised_costs
#     subsystem_positions_dict[instrument] = position
#     print('calc_pnl_across_subsytem_for_indiv_instr')
#
# subsystem_positions = pd.DataFrame(subsystem_positions_dict)
# gross_pnl_df = pd.DataFrame(gross_dict)
# cost_df = pd.DataFrame(costs_dict)
