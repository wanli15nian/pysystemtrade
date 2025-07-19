import pandas as pd

from refactory.base import calc_gross_pnl, calc_position, combine_forecast, calc_cost_sr, \
    calc_net
from refactory.base import calc_vol_scalar
from refactory.cost_actual import calc_cost_actual
from refactory.cost_estimated import calc_cost_sr_all
from refactory.data_source import get_instrument_info, get_price, get_raw_price
from refactory.forecast import ewmac, rescale_forecast, floor_vol, price_vol
from refactory.utils import bundle
from refactory.weights import calc_div_mult_daily, calc_forecast_weights, \
    calc_instrument_weights

# --------------------------------------------------------------------------------------------------------------------


risk_target = 0.16
instruments = ["CORN", "SOFR", "SP500_micro", 'US10']


def calc_forecasts(price):
    raw_ewmac32 = ewmac(price, 32, 128, 1)
    ewmac32 = rescale_forecast(raw_ewmac32 / floor_vol(price_vol(price)))
    raw_ewmac8 = ewmac(price, 8, 32, 1)
    ewmac8 = rescale_forecast(raw_ewmac8 / floor_vol(price_vol(price)))
    forecast_df = pd.DataFrame({'ewmac32': ewmac32, 'ewmac8': ewmac8})
    return forecast_df


# -------------------------------------------------------------------------------------------------------------------

m = lambda func: bundle(func, instruments=instruments)

info = get_instrument_info().loc[instruments]
size = info['point_size']

price = m(get_price)
raw_price = m(get_raw_price)
vol_scalar = m(lambda i: calc_vol_scalar(price.loc[i], size.loc[i], capital=1000000, risk_target=risk_target))

# TODO:考虑把rule后缀改为r，用rule容易有歧义，同样instrument后缀改为i
forecast_rule = m(lambda i: calc_forecasts(price.loc[i]))
position_rule = m(lambda i: calc_position(forecast_rule.loc[i], vol_scalar.loc[i]))
gross_rule = m(lambda i: calc_gross_pnl(position_rule.loc[i], price.loc[i], size.loc[i]))
cost_sr_rule = calc_cost_sr_all(forecast_rule, gross_rule, vol_scalar, price, info)

print('rule level finished')

forecast_weights = m(lambda i: calc_forecast_weights(gross_rule, cost_sr_rule.loc[i], i))
forecast_div_mult = m(lambda i: calc_div_mult_daily(forecast_weights.loc[i], forecast_rule))
forecast_inst = m(lambda i: combine_forecast(forecast_rule.loc[i], forecast_weights.loc[i], forecast_div_mult.loc[i]))
# position_inst_raw = m(lambda i: calc_raw_position(forecast_inst[i], vol_scalar.loc[i]))
# TODO: buffer操作后的position，会把没上市的品种的权重从na变为0，position的na该如何约定？
position_inst = m(lambda i: calc_position(forecast_inst[i], vol_scalar.loc[i], buffer_size=0.10))
gross_inst = m(lambda i: calc_gross_pnl(position_inst.loc[i], price.loc[i], size.loc[i]))
cost_inst = m(lambda i: calc_cost_actual(position_inst.loc[i], price.loc[i], info.loc[i]))
cost_sr_inst = pd.Series({i: calc_cost_sr(gross_inst.loc[i], cost_inst.loc[i], 1) for i in instruments})
net_daily_inst = m(lambda i: calc_net(gross_inst.loc[i], cost_sr_inst[i]))

print('instrument level finished')

portfolio_weights = calc_instrument_weights(net_daily_inst)

# 以下为意义不明变量
# net_inst = calc_net_pnl(gross_inst, cost_inst)
# subsystem_turnover_ = pd.Series({i: calc_turnover(forecast_inst.loc[i], vol_scalar.loc[i]) for i in instruments})


print(position_inst)
print(portfolio_weights)

print('portfolio level finished')

# --------------------------------------------------------------------------------------------------------------------

# _info = info.loc['US10']
# _raw_price = raw_price.loc['US10']
# _price = price.loc['US10']
# _price_pnl = price.loc['US10'].diff()
#
# _vol_scalar = vol_scalar.loc['US10']
# _rule_forecast = forecast_rule.loc['US10']['ewmac32']
# _rule_position = position_rule.loc['US10']['ewmac32']
# _rule_gross = gross_rule.loc['US10']['ewmac32']
# _rule_cost = cost_rule.loc['US10']['ewmac32']
# _rule_net = net_rule.loc['US10']['ewmac32']
# print(_rule_net)
