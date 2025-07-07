from refactory.cost_forecast import get_cost_per_trade, calc_annual_cost, annual_forecast_turnover
from refactory.utils import calc_mixed_volatility


def calc_cost_SR(rules, average_turnover_, weighted_turnover_, pnl, forecast, price, pos_target, info):
    cost_SR_dict = {}
    for rule in rules:
        average_turnover = average_turnover_[rule]
        weighted_turnover = weighted_turnover_[rule]
        gross_pnl_rule = pnl[rule]
        forecast_rule = forecast[rule]
        pooled_cost = calc_cost_SR_by_rule(price, average_turnover, weighted_turnover, forecast_rule, gross_pnl_rule,
                                           pos_target, info)
        cost_SR_dict[rule] = pooled_cost
    # cost_SR_df = pd.DataFrame([cost_SR_dict])
    return cost_SR_dict


def calc_cost_SR_by_rule(price, average_turnover, weighted_turnover, forecast_rule, gross_rule_pnl, pos_target, info):
    rolls_per_year = int(info['rolls_per_year'])
    point_size = info['point_size']
    spread_cost = info['spread_cost']
    per_trade = info['per_trade']
    per_block = info['per_block']
    percentage = info['percentage']

    cost_per_trade = get_cost_per_trade(price, per_block, per_trade, percentage, spread_cost, point_size,
                                        notional_blocks_traded=1)
    annual_cost = calc_annual_cost(weighted_turnover, cost_per_trade, rolls_per_year)
    # pos_target = pos_target.reindex(forecast_rule.index, method="ffill")
    ##PROBLEM: cost curve calc remains to be checked
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
