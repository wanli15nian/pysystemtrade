import numpy as np
import pandas as pd
from scipy.optimize import minimize


def calc_position(forecast, vol_scalar, buffer_size=0):
    position_raw = calc_raw_position(forecast, vol_scalar)
    if buffer_size > 0:
        position_raw = trans_buffered_position(position_raw, vol_scalar, 0.10)
    # position = position.ffill()
    #FIXME: 检查这个shift(1) 是否适用于 subsystem position
    position = position_raw.shift(1)
    return position


def calc_raw_position(forecast, vol_scalar):
    aligned_avg = vol_scalar.reindex(forecast.index, method='ffill')
    position_raw = forecast.mul(aligned_avg, axis=0) / 10
    return position_raw


def calc_gross_pnl(position, price, point_size):
    # FIXME 源代码确实是shift 了两次，没看出来为什么
    position = position.shift(1).ffill()
    pnl_in_points = position.mul(price.ffill().diff(), axis=0).fillna(0)
    return pnl_in_points * point_size


def calc_net_pnl(gross_pnl, daily_costs):
    raw_net = gross_pnl.add(daily_costs, fill_value=0)
    net = raw_net.groupby(level=0).resample('B', level=1).sum()
    return net


def unstack_for_optimisation(multi_index_df):
    unstacked = multi_index_df.unstack(level=0)
    resampled = unstacked.resample('1B').sum()
    resampled[resampled == 0.0] = pd.NA
    resampled = resampled.T.stack(dropna=False).to_frame(name='')
    return resampled


def calc_vol_scalar(price, point_size, capital=500000, risk_target=0.16):
    '''
    根据自行设置的risk target 所计算出的单一品种的目标仓位
    每个contract 能提供的cash vol 为ret_volatility * point_size (每手2500单位，每个单位的vol 为ret_volatility)
    '''
    # TODO: 加上下面这句结果会不一样，price为空值的时候意味什么？
    # price = price.ffill()
    pnl_vol = calc_mixed_volatility(price.diff(), slow_vol_years=10)  # ret_vol 不是百分比，而是绝对值
    risk_target = risk_target / (256 ** 0.5)
    position_target = (capital * risk_target) / (pnl_vol * point_size)
    return position_target


def combine_forecast(forecast, forecast_weights, forecast_div_mult):
    combined_forecast = ((forecast_weights * forecast).sum(axis=1) * forecast_div_mult).clip(20, -20)
    return combined_forecast


def trans_buffered_position(position, vol_scalar, buffer_size=0.10, trade_to_edge=True):
    # vol_scalar 的另一种理解是Avg pos of the subsystem level，就是说position 可以在avg pos的10% 区间内浮动
    buffer = vol_scalar * buffer_size
    top = (position + buffer).ffill().round()
    bottom = (position - buffer).ffill().round()
    rounded_position = position.ffill().round()

    last = 0.0
    buffered_position_list = []
    for index in range(len(rounded_position)):
        last = adjust_by_buffer(last, rounded_position.iloc[index], top.iloc[index], bottom.iloc[index], trade_to_edge)
        buffered_position_list.append(last)
    buffered_position = pd.Series(buffered_position_list, index=rounded_position.index)
    return buffered_position


def adjust_by_buffer(last, current, top, bottom, trade_to_edge=True):
    if np.isnan(top) or np.isnan(bottom) or np.isnan(current):
        return last
    if trade_to_edge:
        return min(max(last, bottom), top)  # 如果在buffer内则不调仓，调仓就调到buffer边缘，尽量减少调仓幅度
    else:
        return last if (bottom <= last <= top) else current  # 如果在buffer内则不调仓


def calc_cost_of_fill(price, info, quantity, include_slippage=True):
    commission_costs = calc_commission(price, quantity, info)
    slippage_costs = calc_slippage(quantity, info) if include_slippage else 0
    total_cost = slippage_costs + commission_costs
    return total_cost


def calc_commission(price, quantity, info):
    # 交易佣金，三种方式只会有一种，其他两种为零，可以用取最大值的方法
    point_size = info['point_size']
    per_trade = info['per_trade']
    per_block = info['per_block']
    percentage = info['percentage']
    block_price_multiplier = point_size * price
    per_block = (abs(quantity) * per_block)
    perc_commission = abs(quantity) * block_price_multiplier * percentage
    commission_costs = max([per_trade, per_block, perc_commission])
    return commission_costs


def calc_slippage(quantity, info):
    # 交易滑点，现在只考虑一个点，以后可以加上参数控制滑几个点
    slippage = info['spread_cost']
    point_size = info['point_size']
    slippage_ = (abs(quantity) * point_size * slippage)
    return slippage_


def optimisation(corr, norm_mean, norm_stdev):
    def addem(weights):
        return 1.0 - sum(weights)

    def neg_SR(weights, sigma, mus):
        estimated_returns = np.dot(weights, mus)[0]
        stdev = weights.dot(sigma).dot(weights.transpose()) ** 0.5
        sr = -estimated_returns / stdev
        return sr

    number = len(corr)
    mus = np.array(norm_mean, ndmin=2).transpose()  # mus 没问题
    sigma = np.diag(norm_stdev).dot(corr).dot(np.diag(norm_stdev))
    start_weights = np.array([1 / number] * number)
    bounds = [(0.0, 1.0)] * number
    cdict = [{"type": "eq", "fun": addem}]
    ans = minimize(neg_SR, start_weights, (sigma, mus), method='SLSQP', constraints=cdict, bounds=bounds, tol=0.00001)
    weight = ans['x']
    return weight


def calc_mixed_volatility(data, days=35, min_periods=10, slow_vol_years=20,
                          perc_of_long_vol=0.3, vol_min=0.0000000001,
                          vol_multiplier=1.0):
    # 长期和短期波动进行权重处理
    short_vol = data.ewm(adjust=True, span=days, min_periods=min_periods).std()
    long_vol = short_vol.ewm(adjust=True, span=slow_vol_years * 256).mean()
    vol = perc_of_long_vol * long_vol + (1 - perc_of_long_vol) * short_vol
    vol[vol < vol_min] = vol_min
    vol = vol * vol_multiplier
    vol.ffill(inplace=True)
    return vol


def calc_net(gross, cost_sr):
    gross = gross.replace(0.0, np.nan)
    vol = gross.std()
    if len(cost_sr) == 1:
        cost_daily = cost_sr[0] * (vol / 16)
    else:
        cost_daily = cost_sr * (vol / 16)
    return gross + cost_daily


def stack_instr(data, freq, method):
    if method == 'sum':
        resampled = (data.groupby(level=0)
                      .resample(freq, level=1).sum())
    elif method == 'last':
        resampled = (data.groupby(level=0)
                     .resample(freq, level=1).last())
    else:
        return None
    stacked = (resampled.unstack(level=0)
                  .stack(dropna=False)
                  .droplevel('instrument')
                  .sort_index(ascending=True))
    return stacked
