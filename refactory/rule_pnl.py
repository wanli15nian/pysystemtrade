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


def calc_gross(forecast, pos_target, price, info):
    point_size = info['point_size']
    position = forecast.mul(pos_target, axis=0) / 10
    position = position.shift(1)
    return calc_gross_pnl(position, price, point_size)


def calc_gross_pnl(position, price, point_size):
    pnl_in_points = calc_daily_gross_pnl_in_points(positions=position, prices=price)
    pnl = pnl_in_points * point_size
    daily_pnl = pnl.resample("B").sum()
    daily_pnl = daily_pnl.squeeze()
    # TODO: 鉴于forecast是个两列的df, daily_pnl_gross也是个两列的df
    # 这就有问题了，应该如何理解这两列的实际持仓呢
    # 计算过程中，我们本质上是把每个rule当成了单独的portfolio来算的，所以才有了用position_target直接乘上去
    # 得出的daily_pnl_gross不能是直接相加吧，如果是的话就不合理了
    # 举例，两个forecast 给出了很弱的信号，所以实际持仓都是目标持仓的60%, 如果直接相加的，反而会导致最终持仓到了目标持仓的120%
    daily_pnl = daily_pnl.replace(0, np.nan)
    return daily_pnl


def calc_daily_gross_pnl_in_points(positions: pd.Series, prices: pd.Series):
    # TODO: 清理，这里为什么还要shift一次，可能会有问题
    '''
    持仓单位为 “手"
    实际持仓再往后调一天，然后乘以价格变化
    因为实际持仓和价格变化都是当天收盘之后算出来的
    所以第一天的实际持仓算出来后，第二天会那么持仓，然后吃满第二天的价格变化
    pnl也会算进第二天里
    需要去算具体金额的盈亏，还得乘以point_size, 也就是比如说一手多少吨
    '''
    pos_series = positions.groupby(positions.index).last()  # 得到当天最后的持仓
    pos_price_series = pd.concat([pos_series, prices], axis=1)
    if len(pos_price_series.columns) == 2:
        pos_price_series.columns = ["positions", "price"]
    pos_price_series = pos_price_series.ffill()
    daily_price_change = pos_price_series.price.diff()
    adjusted_pos_price_series = pos_price_series.loc[:, pos_price_series.columns != 'price'].shift(1)
    daily_pnl_in_points = adjusted_pos_price_series.mul(daily_price_change, axis=0)
    daily_pnl_in_points[daily_pnl_in_points.isna()] = 0.0
    return daily_pnl_in_points
