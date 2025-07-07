import pandas as pd

from refactory.cost_forecast import annual_forecast_turnover
from refactory.utils import calc_mixed_volatility


def calc_cost_SR(average_turnover_, weighted_turnover_, gross, forecast, price, position_target, info):
    rules = gross.columns.to_list()
    cost_SR_dict = {rule: calc_cost_SR_by_rule(
        average_turnover_[rule],
        weighted_turnover_[rule],
        forecast[rule],
        gross[rule],
        price,
        position_target,
        info
    ) for rule in rules}
    return cost_SR_dict


def calc_cost_SR_by_rule(average_turnover, weighted_turnover, forecast_rule, gross_rule_pnl, price, pos_target, info):
    rolls_per_year = int(info['rolls_per_year'])
    point_size = info['point_size']

    cost_per_trade = calc_per_trade(price, info)
    annual_cost = calc_annual_cost(weighted_turnover, cost_per_trade, rolls_per_year)
    cost_curve = calc_cost(pos_target=pos_target, price=price,
                           point_size=point_size, trading_cost=annual_cost)
    # cost_SR_annual算出交易成本与gross returns 波动的比例,越高说明成本越难以接受
    # cost_curve.iloc[:11] = np.nan  # QUESTION: 为什么前11个数都是Nan
    # if instrument == 'US10':
    #     cost_curve.iloc[:13] = np.nan  # QUESTION: 为什么到了US10是前13个数字
    cost_curve_mean = cost_curve.mean()
    gross_daily_pnl_std = gross_rule_pnl.std()
    cost_SR_annual = 16 * cost_curve_mean / gross_daily_pnl_std
    turnover = annual_forecast_turnover(forecast_rule)
    instr_cost_per_turnover = cost_SR_annual / turnover
    cost_multiplier = 2
    pooled_cost = instr_cost_per_turnover * average_turnover * cost_multiplier
    return pooled_cost


def calc_per_trade(price, info):
    point_size = info['point_size']
    spread_cost = info['spread_cost']
    per_trade = info['per_trade']
    per_block = info['per_block']
    percentage = info['percentage']
    cost_per_trade = calc_cost_per_trade(price, per_block, per_trade, percentage, spread_cost, point_size)
    return cost_per_trade


def calc_cost(pos_target, price, point_size, trading_cost):
    # Actually output in price space to match gross returns
    # These will be annualised figure, make it a small loss every day
    # TODO: 完全没有看明白这个calc_cost的计算逻辑
    annualised_price_vol_points = calc_mixed_volatility(price.diff(), slow_vol_years=10)
    sr_cost_as_annualised_figure = (-trading_cost * pos_target * annualised_price_vol_points * 16).bfill()
    period_intervals_in_seconds = sr_cost_as_annualised_figure.index.to_series().diff().dt.total_seconds()
    costs_in_points = sr_cost_as_annualised_figure * period_intervals_in_seconds / (365.25 * 24 * 60 * 60)
    costs = costs_in_points * point_size  # 后续有个fx 的序列，但目前不加
    return costs


def calc_cost_per_trade(price, per_block, per_trade, percentage, price_slippage, point_size, notional_blocks_traded=1):
    # 例如：A股股票必须是100股的整数倍，这个参数是这个100的意思吗？
    blocks = notional_blocks_traded
    # 过去一年的均价
    average_price = float(price[price.index[-1] - pd.DateOffset(years=1):].mean())
    # 交易佣金，三种方式只会有一种，其他两种为零，可以用取最大值的方法
    commission_percentage = blocks * average_price * point_size * percentage
    commission_per_block = blocks * per_block
    commission = max([per_trade, commission_per_block, commission_percentage])
    # 交易滑点，这个应该可以加上参数控制滑几个点
    slippage = blocks * price_slippage * point_size
    # 单次交易成本，包括slippage和commission
    cost = commission + slippage
    # 年化波动率，前面乘了notional_blocks_traded，这里不应该乘吗？
    ann_vol = calc_ann_vol(price, point_size)
    # 夏普成本
    cost_sr = cost / ann_vol
    return cost_sr


def calc_ann_vol(price, point_size):
    vol = calc_mixed_volatility(price.diff(), slow_vol_years=10)
    vol_average = float(vol[price.index[-1] - pd.DateOffset(years=1):].mean())
    ann_vol = vol_average * 16 * point_size
    return ann_vol


def calc_annual_cost(turnover, cost_per_trade, rolls_per_year):
    transaction_cost = turnover * cost_per_trade
    holding_turnovers = rolls_per_year * 2.0
    holding_cost = holding_turnovers * cost_per_trade
    return transaction_cost + holding_cost
