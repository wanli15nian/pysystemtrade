from refactory.utils import calc_mixed_volatility


def calc_target_position(price, info, capital=1000000, risk_target=0.16):
    '''
    根据自行设置的risk target 所计算出的单一品种的目标仓位
    剩余资金的风险暴露应该是0，要不然就是使得整体的风险暴露大于risk target
    每个contract 能提供的cash vol 为ret_volatility * point_size (每手2500单位，每个单位的vol 为ret_volatility)
    '''
    point_size = info['point_size']
    ret_volatility = calc_mixed_volatility(price.diff(), slow_vol_years=10)  # ret_vol 不是百分比，而是绝对值
    daily_risk_target = risk_target / (256 ** 0.5)
    daily_cash_vol_target = daily_risk_target * capital  # 每天的目标vol target
    position_target = daily_cash_vol_target / (ret_volatility * point_size)
    return position_target

# def calc_daily_gross_pnl_in_points(positions: pd.Series, prices: pd.Series):
#     '''
#     持仓单位为 “手"
#     实际持仓再往后调一天，然后乘以价格变化
#     因为实际持仓和价格变化都是当天收盘之后算出来的
#     所以第一天的实际持仓算出来后，第二天会那么持仓，然后吃满第二天的价格变化
#     pnl也会算进第二天里
#     需要去算具体金额的盈亏，还得乘以point_size, 也就是比如说一手多少吨
#     '''
#     pos_series = positions.groupby(positions.index).last()  # 得到当天最后的持仓
#     pos_price_series = pd.concat([pos_series, prices], axis=1)
#     if len(pos_price_series.columns) == 2:
#         pos_price_series.columns = ["positions", "price"]
#     pos_price_series = pos_price_series.ffill()
#     daily_price_change = pos_price_series.price.diff()
#     adjusted_pos_price_series = pos_price_series.loc[:, pos_price_series.columns != 'price'].shift(1)
#     daily_pnl_in_points = adjusted_pos_price_series.mul(daily_price_change, axis=0)
#     daily_pnl_in_points[daily_pnl_in_points.isna()] = 0.0
#     print("calc_daily_pnl_in_points_given_pos_prices")
#     return daily_pnl_in_points
#
#
# def generate_fit_end_list(start_date, end_date):
#     '''
#     从结束日期开始倒推，然后reverse()
#     '''
#     start_dates_per_period = pd.date_range(end_date, start_date, freq='-365D').to_list()
#     start_dates_per_period.reverse()
#     end_list = start_dates_per_period[1:-1]
#     print('generate_fit_end_list')
#     return end_list
