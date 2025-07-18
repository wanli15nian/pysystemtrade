import pandas as pd

from refactory.base import calc_gross_pnl, calc_position, combine_forecast, calc_net, \
    unstack_for_optimisation, stack_instr, calc_raw_position, calc_cost_sr, normalize_cost_sr
from refactory.base import calc_vol_scalar
from refactory.cost_actual import calc_cost_actual
from refactory.cost_estimated import calc_cost_estimated
from refactory.data_source import get_instrument_info, get_price, get_raw_price
from refactory.forecast import ewmac, rescale_forecast, floor_vol, price_vol
from refactory.turnover import estimate_turnover_all, estimate_weighted_turnover
from refactory.weights_forecast import calc_weights, calc_div_mult_daily

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

def m(func, instruments=instruments):
    # 纵向组装。将func返回的dataset组装成muliIndex的dataset
    return pd.concat((func(i) for i in instruments),
                     keys=instruments,
                     names=['instrument', 'datetime'])


# -------------------------------------------------------------------------------------------------------------------

info = get_instrument_info().loc[instruments]
size = info['point_size']
price = m(get_price)
raw_price = m(get_raw_price)

print('get price and info')

# TODO:考虑把rule后缀改为r，用rule容易有歧义，同样instrument后缀改为i
vol_scalar = m(lambda i: calc_vol_scalar(price.loc[i], size.loc[i], capital=1000000, risk_target=risk_target))
forecast_rule = m(lambda i: calc_forecasts(price.loc[i]))
position_rule = m(lambda i: calc_position(forecast_rule.loc[i], vol_scalar.loc[i]))
gross_rule = m(lambda i: calc_gross_pnl(position_rule.loc[i], price.loc[i], size.loc[i]))
print('calculate gross for instrument and rule')

turnover_all = estimate_turnover_all(forecast_rule)
turnover_weighted = estimate_weighted_turnover(turnover_all, forecast_rule)
# FIXME:这里应该传raw_price吧？
cost_rule = m(lambda i: calc_cost_estimated(price.loc[i], turnover_weighted, vol_scalar.loc[i], info.loc[i]))
cost_sr_rule = pd.DataFrame({i: calc_cost_sr(gross_rule.loc[i], cost_rule.loc[i], 2) for i in instruments}).transpose()
cost_sr_rule = normalize_cost_sr(cost_sr_rule, turnover_all)


def calc_forecast_weights_(gross, cost_sr, instrument_gross):
    # 注: 需要用m 函数以保证net 的正确计算
    net = m(lambda i: calc_net(gross.loc[i], cost_sr))
    net_weekly = stack_instr(net, 'W', 'sum')
    instruments_num = len(net.index.levels[0])
    config = {
        'corr_span': instruments_num * 50000,
        'corr_min_periods': instruments_num * 10,
        'multiple_span': instruments_num * 50000,
        'multiple_min_periods': instruments_num * 5,
        'shrinkage_corr': 0.5,
        'shrinkage_sr': 0.9,
        'sr_target': 0.5,
        'equalise_vol': True
    }
    forecast_weights = calc_weights(net_weekly, instrument_gross, config)
    return forecast_weights


forecast_weights = m(lambda i: calc_forecast_weights_(gross_rule, cost_sr_rule.loc[i], gross_rule.loc[i]))
forecast_div_mult = m(lambda i: calc_div_mult_daily(forecast_weights.loc[i], forecast_rule))
forecast_inst = m(lambda i: combine_forecast(forecast_rule.loc[i], forecast_weights.loc[i], forecast_div_mult.loc[i]))
# TODO: buffer操作后的position，会把没上市的品种的权重从na变为0，position的na该如何约定？
position_inst_raw = m(lambda i: calc_raw_position(forecast_inst[i], vol_scalar.loc[i]))
position_inst = m(lambda i: calc_position(forecast_inst[i], vol_scalar.loc[i], buffer_size=0.10))
gross_inst = m(lambda i: calc_gross_pnl(position_inst.loc[i], price.loc[i], size.loc[i]))
cost_inst = m(lambda i: calc_cost_actual(position_inst.loc[i], price.loc[i], info.loc[i]))


# 以下为意义不明变量
# net_inst = calc_net_pnl(gross_inst, cost_inst)
# subsystem_turnover_ = pd.Series({i: calc_turnover(forecast_inst.loc[i], vol_scalar.loc[i]) for i in instruments})

def calc_instrument_weights1(gross, cost_sr, subsystem_position):
    net_daily = m(lambda i: calc_net(gross.loc[i], cost_sr[i]))
    net = (net_daily.unstack(level=0)
           .resample('W').sum())
    subsystem_position1 = subsystem_position.unstack().T.ffill()
    config = {
        'corr_span': 500000,
        'corr_min_periods': 10,
        'multiple_span': 50000,
        'multiple_min_periods': 5,
        'shrinkage_corr': 0.5,
        'shrinkage_sr': 0.9,
        'sr_target': 0.5,
        'equalise_vol': True
    }
    weights = calc_weights(net, subsystem_position1, config)
    return weights


def calc_instrument_weights(gross_inst, cost_inst, position_inst_raw):
    gross_inst_ = unstack_for_optimisation(gross_inst)
    cost_inst_ = unstack_for_optimisation(cost_inst)
    cost_sr_inst = pd.Series({i: calc_cost_sr(gross_inst_.loc[i], cost_inst_.loc[i], 1) for i in instruments})
    net_daily = m(lambda i: calc_net(gross_inst_.loc[i], cost_sr_inst[i]))
    net = (net_daily.unstack(level=0)
           .resample('W').sum())
    position_unstacked = position_inst_raw.unstack().T.ffill()
    config = {
        'corr_span': 500000,
        'corr_min_periods': 10,
        'multiple_span': 50000,
        'multiple_min_periods': 5,
        'shrinkage_corr': 0.5,
        'shrinkage_sr': 0.9,
        'sr_target': 0.5,
        'equalise_vol': True
    }
    return calc_weights(net, position_unstacked, config)


portfolio_weights = calc_instrument_weights(gross_inst, cost_inst, position_inst_raw)

print(portfolio_weights)

print('calculate weightes for portfolio')

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
