import numpy as np
import pandas as pd

from refactory.utils import calc_mixed_volatility


def calc_position(forecast, vol_scalar, buffer_size=0):
    aligned_avg = vol_scalar.reindex(forecast.index, method='ffill')
    position = forecast.mul(aligned_avg, axis=0) / 10
    if buffer_size > 0:
        position = trans_buffered_position(position, vol_scalar, 0.10)
    # position = position.ffill()
    position = position.shift(1)
    return position


def calc_gross_pnl(position, price, point_size):
    # FIXME 源代码确实是shift 了两次，没看出来为什么
    position = position.shift(1).ffill()
    pnl_in_points = position.mul(price.ffill().diff(), axis=0).fillna(0)
    return pnl_in_points * point_size


def calc_net_pnl(gross_pnl, daily_costs):
    return gross_pnl.add(daily_costs, fill_value=0)


def calc_vol_scalar(price, point_size, capital=500000, risk_target=0.16):
    '''
    根据自行设置的risk target 所计算出的单一品种的目标仓位
    每个contract 能提供的cash vol 为ret_volatility * point_size (每手2500单位，每个单位的vol 为ret_volatility)
    '''
    # TODO: 加上下面这句结果会不一样，price为空值的时候意味什么？
    # price = price.ffill()
    pnl_vol = calc_mixed_volatility(price.diff(), slow_vol_years=10)  # ret_vol 不是百分比，而是绝对值
    risk_target = risk_target / (256 ** 0.5)
    position_target = (capital * risk_target) / (pnl_vol * point_size)
    return position_target


def combine_forecast(forecast, forecast_weights, forcast_div_mult):
    combined_forecast = ((forecast_weights * forecast).sum(axis=1) * forcast_div_mult).clip(20, -20)
    return combined_forecast


def trans_buffered_position(position, vol_scalar, buffer_size=0.10, trade_to_edge=True):
    # vol_scalar 的另一种理解是Avg pos of the subsystem level，就是说position 可以在avg pos的10% 区间内浮动
    buffer = vol_scalar * buffer_size
    top = (position + buffer).ffill().round()
    bottom = (position - buffer).ffill().round()
    position = position.ffill().round()

    last = 0.0
    buffered_position_list = []
    for index in range(len(position)):
        last = adjust_by_buffer(last, position.iloc[index], top.iloc[index], bottom.iloc[index], trade_to_edge)
        buffered_position_list.append(last)
    buffered_position = pd.Series(buffered_position_list, index=position.index)
    return buffered_position


def adjust_by_buffer(last, current, top, bottom, trade_to_edge=True):
    if np.isnan(top) or np.isnan(bottom) or np.isnan(current):
        return last
    if trade_to_edge:
        return min(max(last, bottom), top)  # 如果在buffer内则不调仓，调仓就调到buffer边缘，尽量减少调仓幅度
    else:
        return last if (bottom <= last <= top) else current  # 如果在buffer内则不调仓


def calc_cost_of_fill(price, info, quantity, include_slippage=True):
    commission_costs = calc_commission(price, quantity, info)
    slippage_costs = calc_slippage(quantity, info) if include_slippage else 0
    total_cost = slippage_costs + commission_costs
    return total_cost


def calc_commission(price, quantity, info):
    # 交易佣金，三种方式只会有一种，其他两种为零，可以用取最大值的方法
    point_size = info['point_size']
    per_trade = info['per_trade']
    per_block = info['per_block']
    percentage = info['percentage']
    block_price_multiplier = point_size * price
    per_block = (abs(quantity) * per_block)
    perc_commission = abs(quantity) * block_price_multiplier * percentage
    commission_costs = max([per_trade, per_block, perc_commission])
    return commission_costs


def calc_slippage(quantity, info):
    # 交易滑点，现在只考虑一个点，以后可以加上参数控制滑几个点
    slippage = info['spread_cost']
    point_size = info['point_size']
    slippage_ = (abs(quantity) * point_size * slippage)
    return slippage_
