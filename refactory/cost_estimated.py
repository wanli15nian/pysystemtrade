import pandas as pd

from refactory.base import calc_cost_of_fill
from refactory.utils import calc_mixed_volatility


def calc_cost_estimated(price, turnover, vol_scalar, info):
    return pd.DataFrame({r: estimate_cost(price, turnover[r], vol_scalar, info)
                         for r in (turnover.index.to_list())})


def estimate_cost(price, turnover, vol_scalar, info):
    # 计算年夏普成本
    cost_sr_annual = get_cost_sr_annual(turnover, price, info)
    vol_annual = calc_mixed_volatility(price.diff(), slow_vol_years=10) * 16
    cost_annual = (-cost_sr_annual * vol_annual * vol_scalar).ffill()
    # 计算日成本
    vol_scalar = vol_scalar.shift(1)  # TODO：这个shift是必要的吗？
    cost_annual = (cost_annual.reindex(vol_scalar.index)
                   [~vol_scalar.isna()]
                   .reindex(price.index, method='ffill'))
    interval_as_year = cost_annual.index.to_series().diff().dt.total_seconds() / (365.25 * 24 * 60 * 60)
    point_size = info['point_size']
    cost_daily = cost_annual * interval_as_year * point_size
    return cost_daily


def get_cost_sr_annual(turnover, price, info):
    # A股股票必须是100股的整数倍，notional_blocks这个参数是这个100的意思吗？
    # 总成本 = 交易成本 + 移仓换月成本，都是以SR计算的。
    cost_sr_per = calc_cost_sr_per(price, info)
    rolls_per_year = int(info['rolls_per_year'])
    holding_cost = rolls_per_year * 2.0 * cost_sr_per
    transaction_cost = turnover * cost_sr_per
    cost_sr_annual = transaction_cost + holding_cost
    return cost_sr_annual


def calc_cost_sr_per(price, info):
    # TODO：这个应该是滚动计算的吧？不能只用当前最近一年的。
    point_size = info['point_size']
    average_price = price[price.index[-1] - pd.DateOffset(years=1):].mean()  # 过去一年的均价
    average_cost = calc_cost_of_fill(average_price, info, 1)

    vol = calc_mixed_volatility(price.diff(), slow_vol_years=10)
    average_vol_daily = vol[price.index[-1] - pd.DateOffset(years=1):].mean()  # 过去一年的平均波动率
    average_vol = average_vol_daily * 16 * point_size

    cost_sr_per = average_cost / average_vol
    return cost_sr_per

# FIXME 看看这些东西什么时候会被用上，目前处于无用状态
# 计算日均成本
# point_size = info['point_size']
# interval_as_year = cost_annual.index.to_series().diff().dt.total_seconds() / (365.25 * 24 * 60 * 60)
# cost_daily = cost_annual * interval_as_year * point_size
# cost_daily_mean = cost_daily.mean()
# # 计算年夏普成本
# pnl_vol_daily = pnl.std()
# cost_sr_annual = 16 * (cost_daily_mean / pnl_vol_daily)
# # 计算平均年夏普成本
# cost_sr = cost_sr_annual * (average_turnover / turnover_annual) * 2
