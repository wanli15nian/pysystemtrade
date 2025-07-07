import numpy as np
import pandas as pd

from refactory.apply_buffer_to_position import calc_buffered_pos_given_raw_pos
from refactory.cost import calc_cost
from refactory.cost_forecast import annual_forecast_turnover, calculate_weighted_turnover, calc_turnover_weights
from refactory.cost_sr import calc_cost_SR
from refactory.data_source import get_instrument_info, get_daily_price
from refactory.forecast import calc_forecasts
from refactory.functions import combine_forecast, calc_net_pnl
from refactory.gross_pnl import calc_gross, calc_gross_pnl
from refactory.portfolio_weights import calc_portfolio_weights
from refactory.system_turnover import calc_system_turnover
from refactory.target_volatility import calc_target_position
from refactory.utils import calc_volatility_scalar

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

    # Fixme:为什么不用buffer后的position？
    subsystem_positions_dict[instrument] = subsystem_position_raw

    position_buffered = calc_buffered_pos_given_raw_pos(subsystem_position_raw, vol_scalar, 0.10)
    position = position_buffered.shift(1)
    gross_pnl = calc_gross_pnl(position, price, point_size)
    normalised_costs = calc_cost(position, price, info)

    gross_dict[instrument] = gross_pnl
    costs_dict[instrument] = normalised_costs
    print('calc_pnl_across_subsytem_for_indiv_instr')

subsystem_positions = pd.DataFrame(subsystem_positions_dict).ffill()
gross_pnl_df = pd.DataFrame(gross_dict)
cost_df = pd.DataFrame(costs_dict)

system_turnover_ = {i: calc_system_turnover(subsystem_positions[i], price_.loc[i], size_.loc[i]) for i in instruments}

# 无用代码，仅为显示个结果以便核对重构是否成功
# gross_pnl_sum = gross_pnl_df.sum(axis=1)
# cost_sum = cost_df.sum(axis=1)
# net_PNL = gross_pnl_sum.add(cost_sum, fill_value=0).resample('B').sum()
# print(net_PNL)

net_return_raw = pd.DataFrame({inst: gross_pnl_df[inst] + cost_df[inst].mean() for inst in instruments})
portfolio_weights = calc_portfolio_weights(net_return_raw, subsystem_positions)
print(portfolio_weights)

print('END')
