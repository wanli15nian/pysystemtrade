import pandas as pd

from refactory.base import calc_gross_pnl, calc_net_pnl, calc_position, combine_forecast, calc_net_rule
from refactory.base import calc_vol_scalar
from refactory.cost_actual import calc_cost_actual
from refactory.cost_estimated import calc_cost_estimated, calc_cost_sr_rule
from refactory.data_source import get_instrument_info, get_price, get_raw_price
from refactory.forecast import ewmac, rescale_forecast, floor_vol, price_vol
from refactory.turnover import estimate_weighted_turnover, estimate_turnover_annual, calc_turnover
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


def calc_forecast_weights_(gross, cost_sr):
    net = calc_net_rule(gross, cost_sr)
    forecast_weights = calc_forecast_weights(net)
    return forecast_weights


cost_sr_rule = m(lambda i: calc_cost_sr_rule(gross_rule.loc[i], cost_rule.loc[i], turnover_full.loc[i],
                                             turnover_average))
forecast_weights = m(lambda i: calc_forecast_weights_(gross_rule, cost_sr_rule[i]))
forecast_div_mult = m(lambda i: calc_div_mult_daily(forecast_weights.loc[i], forecast_rule.loc[i]))
forecast_inst = c(lambda i: combine_forecast(forecast_rule.loc[i], forecast_weights.loc[i], forecast_div_mult.loc[i]))
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
