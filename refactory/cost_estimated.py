import pandas as pd

from refactory.base import calc_cost_of_fill, calc_mixed_volatility, calc_cost_sr
from refactory.turnover import estimate_turnover_all, estimate_weighted_turnover
from refactory.utils import bundle


def calc_cost_sr_all(forecast_rule, gross_rule, vol_scalar, price, info):
    turnover_all = estimate_turnover_all(forecast_rule)
    turnover_weighted = estimate_weighted_turnover(turnover_all, forecast_rule)

    instruments = gross_rule.index.levels[0]
    # FIXME:这里应该传raw_price吧？
    cost_rule = bundle(lambda i: calc_cost_estimated(price.loc[i], turnover_weighted, vol_scalar.loc[i], info.loc[i]),
                       instruments=instruments)

    cost_sr_rule = pd.DataFrame(
        {i: calc_cost_sr(gross_rule.loc[i], cost_rule.loc[i], 2) for i in instruments}).transpose()
    turnover_average = turnover_all.mean(axis=0)
    return cost_sr_rule * (turnover_average / turnover_all)


def calc_cost_estimated(price, turnover, vol_scalar, info):
    return pd.DataFrame({r: estimate_daily_cost(price, turnover[r], vol_scalar, info)
                         for r in (turnover.index.to_list())})


def estimate_daily_cost(price, turnover, vol_scalar, info):
    # 计算年夏普成本
    cost_sr_annual = get_cost_sr_annual(turnover, price, info)
    vol_annual = calc_mixed_volatility(price.diff(), slow_vol_years=10) * 16
    cost_annual = (-cost_sr_annual * vol_annual * vol_scalar).ffill()
    # 计算日成本
    vol_scalar = vol_scalar.shift(1)
    cost_annual = (cost_annual.reindex(vol_scalar.index)
                   [~vol_scalar.isna()]
                   .reindex(price.index, method='ffill'))
    interval_as_year = cost_annual.index.to_series().diff().dt.total_seconds() / (365.25 * 24 * 60 * 60)
    point_size = info['point_size']
    cost_daily = cost_annual * interval_as_year * point_size
    return cost_daily


def get_cost_sr_annual(turnover, price, info):
    # 总成本 = 移仓换月成本 + 交易成本，都是以SR计算的。
    cost_sr_per = calc_cost_sr_per(price, info)
    rolls_per_year = int(info['rolls_per_year'])
    holding_cost = rolls_per_year * 2.0 * cost_sr_per
    transaction_cost = turnover * cost_sr_per
    cost_sr_annual = transaction_cost + holding_cost
    return cost_sr_annual


def calc_cost_sr_per(price, info):
    # TODO：这个应该不能只用当前最近一年的，是滚动计算的吧？是因为估算就简单处理一下？
    point_size = info['point_size']
    average_price = price[price.index[-1] - pd.DateOffset(years=1):].mean()  # 过去一年的均价
    average_cost = calc_cost_of_fill(average_price, info, 1)

    vol = calc_mixed_volatility(price.diff(), slow_vol_years=10)
    average_vol_daily = vol[price.index[-1] - pd.DateOffset(years=1):].mean()  # 过去一年的平均波动率
    average_vol = average_vol_daily * 16 * point_size

    cost_sr_per = average_cost / average_vol
    return cost_sr_per
