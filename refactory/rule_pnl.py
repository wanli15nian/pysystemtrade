import numpy as np

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


def calc_gross(forecast, pos_target, price, point_size):
    position = calc_rule_position(forecast, pos_target)
    return calc_rule_gross(position, price, point_size)


def calc_rule_position(forecast, pos_target):
    position = forecast.mul(pos_target, axis=0) / 10
    position = position.shift(1)
    return position


def calc_rule_gross(position, price, point_size):
    # TODO: 应该在计算position时做shift
    position = position.shift(1)

    pnl_in_points = position.mul(price.ffill().mean(), axis=0)
    pnl_in_points[pnl_in_points.isna()] = 0.0
    pnl = pnl_in_points * point_size

    # TODO 换算成日频的，说明price可以是分钟级别的，后面需要详细检查一下在计算position之前不应限定只是日频的
    daily_pnl = pnl.resample("B").sum()
    daily_pnl = daily_pnl.replace(0, np.nan)

    return daily_pnl
