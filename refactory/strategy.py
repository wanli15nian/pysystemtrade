import pandas as pd
import quantstats as qs

from refactory.core import calc_gross, calc_position, combine_forecast, calc_cost_sr, \
    calc_net, calc_net_, calc_weight_adjusted_position
from refactory.core import calc_vol_scalar
from refactory.cost_actual import calc_cost_actual
from refactory.cost_estimated import calc_cost_sr_all
from refactory.data_source import get_instrument_info, get_price, get_raw_price
from refactory.forecast import ewmac, rescale_forecast, floor_vol, price_vol
from refactory.stats import gaintolossratio, profitfactor, min, max, mean, median, std, skew, avg_losses, avg_gains, \
    ann_mean, ann_vol, sharpe, avg_drawdown, calmar, avg_return_to_drawdown
from refactory.utils import bundle
from refactory.weights import calc_rule_div_mult_daily, calc_forecast_weights, \
    calc_instrument_weights, calc_instrument_div_mult_daily

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

r_forecast = m(lambda i: calc_forecasts(price.loc[i]))
r_position = m(lambda i: calc_position(r_forecast.loc[i], vol_scalar.loc[i]))
r_gross = m(lambda i: calc_gross(r_position.loc[i], price.loc[i], size.loc[i]))
r_cost_sr = calc_cost_sr_all(r_forecast, r_gross, vol_scalar, price, info)

print('rule level finished')

forecast_weights = m(lambda i: calc_forecast_weights(r_gross, r_cost_sr.loc[i], i))
forecast_multiplier = m(lambda i: calc_rule_div_mult_daily(forecast_weights.loc[i], r_forecast))
i_forecast = m(lambda i: combine_forecast(r_forecast.loc[i], forecast_weights.loc[i], forecast_multiplier.loc[i]))
i_position = m(lambda i: calc_position(i_forecast[i], vol_scalar.loc[i], buffer_size=0.10))
i_gross = m(lambda i: calc_gross(i_position.loc[i], price.loc[i], size.loc[i]))
i_cost = m(lambda i: calc_cost_actual(i_position.loc[i], price.loc[i], info.loc[i]))
i_cost_sr = pd.Series({i: calc_cost_sr(i_gross.loc[i], i_cost.loc[i], 1) for i in instruments})

print('instrument level finished')

# 以下为意义不明变量
# net_inst = calc_net_pnl(gross_inst, cost_inst)
# subsystem_turnover_ = pd.Series({i: calc_turnover(forecast_inst.loc[i], vol_scalar.loc[i]) for i in instruments})


i_net = m(lambda i: calc_net(i_gross.loc[i], i_cost_sr[i]))
instrument_weights = calc_instrument_weights(i_net)
i_net2 = m(lambda i: calc_net_(i_gross.loc[i], i_cost.loc[i]))
instrument_multiplier = calc_instrument_div_mult_daily(instrument_weights, i_net2)
# p_position
# p_gross
# p_net
capital = 1000000
buffered_inst_pos = m(lambda i: calc_weight_adjusted_position(i, instrument_weights,
                                                              i_forecast.loc[i], instrument_multiplier,
                                                              vol_scalar.loc[i]))
portfolio_gross = m(lambda i: calc_gross(buffered_inst_pos.loc[i], raw_price.loc[i], size.loc[i]))
portfolio_gross_perc = portfolio_gross.unstack().sum() / capital
portfolio_cost = m(lambda i: calc_cost_actual(buffered_inst_pos.loc[i], price.loc[i], info.loc[i]))
portfolio_cost_perc = portfolio_cost.unstack().sum() / capital
portfolio_net = portfolio_gross_perc.add(portfolio_cost_perc, fill_value=0.0)


# portfolio_net = m(lambda i: calc_net_(portfolio_gross.loc[i], portfolio_cost.loc[i]))
# portfolio_net_ = portfolio_net.unstack().sum()


def portfolio_stat(net):
    stats_list = [
        "min",
        "max",
        "median",
        "mean",
        "std",
        "skew",
        "ann_mean",
        "ann_std",
        "sharpe",
        # "sortino",
        "avg_drawdown",
        # "time_in_drawdown",
        "calmar",
        "avg_return_to_drawdown",
        "avg_loss",
        "avg_gain",
        "gaintolossratio",
        "profitfactor",
        # "hitrate",
        # "t_stat",
        # "p_value",
    ]
    function_dict = {
        "min": min,
        "max": max,
        "median": median,
        "mean": mean,
        "std": std,
        "skew": skew,
        "ann_mean": ann_mean,
        "ann_std": ann_vol,
        "sharpe": sharpe,
        "avg_drawdown": avg_drawdown,
        "calmar": calmar,
        "avg_return_to_drawdown": avg_return_to_drawdown,
        "avg_loss": avg_losses,
        "avg_gain": avg_gains,
        "gaintolossratio": gaintolossratio,
        "profitfactor": profitfactor,
    }
    results = {stat: function_dict[stat](net) for stat in stats_list}
    return results


portfolio_stat = portfolio_stat(portfolio_net)
print(portfolio_net)
print('portfolio level finished')

qs.reports.html(portfolio_net, output='performance_portfolio.html', title='portfolio')

# print(instrument_weights)
# print(instrument_multiplier)

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

# FIXME: 检查price和raw price是否用错。price的本质不是价格是收益率，涉及到因子计算，波动率计算的用price。
# FIXME：检查resample日频是否正确。策略可以是日内分钟频率的，需要查一遍，resample成日频的地方对不对，cost都是日频的，那么net应该也是日频的，但是gross应该分为原始的和日频的
# FIXME：检查各阶段数据对于na和0的定义。比如position的na指的是什么，0指的是什么，如果两者所指一样，就都设为0.
# FIXME: r层和i层计算cost的逻辑为什么不一致？
# FIXME: p层计算weights和multiplier用的net为什么不一致？
