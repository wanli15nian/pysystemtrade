import numpy as np
import pandas as pd

from refactory.utils import calc_mixed_volatility


def calc_position_target(price, point_size, capital=1000000, annual_risk_target=0.16):
    '''
    根据自行设置的risk target 所计算出的单一品种的目标仓位
    每个contract 能提供的cash vol 为ret_volatility * point_size (每手2500单位，每个单位的vol 为ret_volatility)
    '''
    pnl_vol = calc_mixed_volatility(price.diff(), slow_vol_years=10)  # ret_vol 不是百分比，而是绝对值
    risk_target = annual_risk_target / (256 ** 0.5)
    position_target = (capital * risk_target) / (pnl_vol * point_size)
    return position_target


def calc_position(forecast, pos_target):
    position = forecast.mul(pos_target, axis=0) / 10
    return position


def calc_gross_pnl(position, price, point_size):
    pnl_in_points = position.mul(price.ffill().mean(), axis=0)
    pnl_in_points[pnl_in_points.isna()] = 0.0
    pnl = pnl_in_points * point_size
    # TODO 换算成日频的，说明price可以是分钟级别的，后面需要详细检查一下在计算position之前不应限定只是日频的
    daily_pnl = pnl.resample("B").sum()
    daily_pnl = daily_pnl.replace(0, np.nan)
    return daily_pnl


def calc_net_pnl(gross_pnl, cost_SR):
    daily_cost_sr = cost_SR / 16
    daily_cost = (daily_cost_sr * gross_pnl.std()).item()
    net_pnl_rule = gross_pnl + daily_cost
    return net_pnl_rule


def calc_buffered_position(position_raw, vol_scalar, buffer_size=0.10, trade_to_edge=True):
    # vol_scalar 的另一种理解是Avg pos of the subsystem level，就是说position 可以在avg pos的10% 区间内浮动
    buffer = vol_scalar * buffer_size
    top = (position_raw + buffer).ffill().round()
    bottom = (position_raw - buffer).ffill().round()
    position = position_raw.ffill().round()

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


def calc_volatility_scalar(raw_price, price, block_move_value, capital=500000, risk_target=0.25, vol_mult=1.0):
    # raw_price, price = raw_price.align(price, join="inner")
    # raw_price.ffill(inplace=True)
    # price.ffill(inplace=True)

    pnl_vol = vol_mult * calc_mixed_volatility(price.diff(), slow_vol_years=10)
    vol_percent = 100.0 * (pnl_vol / raw_price.abs())

    block_value = block_move_value * raw_price * 0.01
    # TODO 这个到底起了什么作用？去掉了结果为什么会有差异？
    block_value, vol_percent = block_value.align(vol_percent, join="inner")

    currency_vol = block_value * vol_percent

    daily_currency_vol_target = capital * (risk_target / 16)
    volatility_scalar = daily_currency_vol_target / currency_vol

    volatility_scalar.ffill(inplace=True)

    return volatility_scalar
