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



# def calculate_instrument_weights(pnl_df):
#     daily_pnl = pnl_df.resample("1B").sum()
#     daily_pnl[daily_pnl == 0.0] = np.nan
#
#     number = len(daily_pnl.columns)
#     weekly_ret = daily_pnl.resample('W').sum()  # SP500_micro 的一些数值不对，其他的都能对的上。怀疑是不是一些nan被填充了
#     fit_end = weekly_ret.index[-1]
#     span = 500000
#     min_periods = 10
#
#     norm_stdev, _, = get_stdev_estim_for_instr_weight(weekly_ret, fit_end, span, min_periods)
#     norm_mean = [0.5 * asset_stdev for asset_stdev in norm_stdev]
#     corr = get_corr_estim_for_instr_weight(weekly_ret, fit_end, span, min_periods)
#
#     weight = optimisation(number, corr, norm_mean, norm_stdev)
#     return weight
#
#
# def get_turnover_for_list_of_rules(instrument_list, trading_rule_list):
#     instrument_turnover_dict = dict()
#     for instrument in instrument_list:
#         instrument_turnover_dict[instrument] = {
#             rule_name: forecast_turnover_for_indiv_instr(instrument, rule_name)
#             for rule_name in trading_rule_list
#         }
#     print('get_turnover_for_list_of_rules')
#     return instrument_turnover_dict
#
#
# def calc_net_returns_dict_for_all_instr(dict_of_sr_costs, gross_returns_dict):
#     net_returns_dict = {}
#     for instrument in gross_returns_dict.keys():
#         gross_returns = gross_returns_dict[instrument]
#         net_returns_single_instrument = {}
#         for column_name in gross_returns.columns:
#             gross_returns_daily_std = gross_returns[column_name].std()
#             daily_sr_cost = dict_of_sr_costs[column_name] / 16
#             daily_returns_cost = (daily_sr_cost * gross_returns_daily_std).item()
#             net_returns_single_instrument_rule = gross_returns[column_name] + daily_returns_cost
#             net_returns_single_instrument[column_name] = net_returns_single_instrument_rule
#         net_returns_single_instrument = pd.DataFrame(net_returns_single_instrument)
#         net_returns_dict[instrument] = net_returns_single_instrument  # CLEARED
#     net_returns = single_resampled_set_of_returns(net_returns_dict, frequency='W')  # CLEARED
#     print('calc_net_returns_dict_for_all_instr')
#     return net_returns
#
#
# def calc_buffered_pos_given_combined_forecast(volatility_scalar, position_raw):
#     # 小数点后8位开始对不上，暂时不管
#     # position_raw[position_raw < 0] = 0
#     # position_raw.fillna(0.0, inplace=True)
#     position_buffered = calc_buffered_pos_given_raw_pos(position_raw, volatility_scalar, 0.10)
#     return position_buffered

# def calc_shrunk_means(annualised_return_mean, annualised_return_std, shrinkage_sr=0.9, target_sr=0.5):
#     sr_estimates = (annualised_return_mean / annualised_return_std).to_list()
#     post_sr_list = [(shrinkage_sr * target_sr) + (1 - shrinkage_sr) * estimatedSR for estimatedSR in sr_estimates]
#     shrunk_means_values = (post_sr_list * annualised_return_std).to_list()
#     instruments = annualised_return_mean.index.to_list()
#     shrunk_means = [(asset_name, mean_value) for (asset_name, mean_value) in zip(instruments, shrunk_means_values)]
#     return shrunk_means
