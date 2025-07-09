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
    position = position.shift(1)
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


def calc_buffered_position(position_raw, vol_scalar, buffer_size=0.10):
    '''
    vol_scalar 的另一种理解是Avg pos of the subsystem level
    这么理解的话就是说position 可以在avg pos的10% 区间内浮动
    '''
    buffer = vol_scalar * buffer_size
    top_pos = (position_raw + buffer).ffill().round()
    bottom_pos = (position_raw - buffer).ffill().round()
    position_raw = position_raw.ffill().round()

    last = position_raw.values[0]
    if np.isnan(last):
        last = 0.0
    buffered_position_list = [last]

    # range(len(position_raw)) 跳过第一个Element
    for index in range(1, len(position_raw)):
        last = adjust_by_buffer(last, position_raw.values[index],
                                top_pos.values[index], bottom_pos.values[index])
        buffered_position_list.append(last)
    buffered_position = pd.Series(buffered_position_list, index=position_raw.index)
    # last = position_raw.shift(1).bfill()
    # df = pd.DataFrame({'last': last, 'current': position_raw, 'top': top_pos, 'bottom': bottom_pos})
    # buffered_position = df.apply(lambda x: adjust_by_buffer(x['last'], x['current'], x['top'], x['bottom']), axis=1)

    return buffered_position


def adjust_by_buffer(last, current, top, bottom, trade_to_edge=True):
    if np.isnan(top) or np.isnan(bottom) or np.isnan(current):
        return last

    if last > top:
        if trade_to_edge:
            return top
        else:
            return current
    elif last < bottom:
        if trade_to_edge:
            return bottom
        else:
            return current
    else:
        return last


def calc_volatility_scalar(raw_price, price, block_move_value, capital=500000, risk_target=0.25, vol_mult=1.0):
    raw_price, price = raw_price.align(price, join="inner")
    raw_price.ffill(inplace=True)
    price.ffill(inplace=True)
    pnl_vol = vol_mult * calc_mixed_volatility(price.diff(), slow_vol_years=10)
    vol_percent = 100.0 * (pnl_vol / raw_price.abs())
    block_value = block_move_value * raw_price * 0.01
    currency_vol = block_value * vol_percent
    daily_cash_vol_target = capital * (risk_target / 16)
    volatility_scalar = daily_cash_vol_target / currency_vol
    return volatility_scalar

# def calc_volatility_scalar(raw_price, price, block_move_value, capital, risk_target):
#     '''
#     Get ratio of required volatility vs volatility of instrument in instrument's own currency
#     Gets daily prices for use with % volatility
#     This won't always be the same as the normal 'price'
#     '''
#
#     block_value = raw_price.ffill() * 0.01 * block_move_value
#     block_value.ffill(inplace=True)
#
#     annualised_price_vol_points = calc_mixed_volatility(price.diff(), slow_vol_years=10)
#     annualised_price_vol_points.ffill(inplace=True)
#
#     # Align resampled carry price and annualised price volatility in points
#     resampled_carry_price = raw_price.resample('1B').last()
#     (resampled_carry_price, annualised_price_vol_points) = resampled_carry_price.align(annualised_price_vol_points,
#                                                                                        join='right')
#     percentage_vol = 100.0 * (annualised_price_vol_points / resampled_carry_price.ffill().abs())
#
#     (block_value, percentage_vol) = block_value.align(percentage_vol, join="inner")
#     # It is to multiply by fx_rate, which is taken to be 1 here
#     fx_rate = 1
#     currency_vol = (block_value * percentage_vol).ffill() * fx_rate
#
#     cash_vol_target = capital * risk_target / 16
#     vol_scalar = cash_vol_target / currency_vol
#     vol_scalar = vol_scalar.reindex(price.index, method="ffill")
#     return vol_scalar
