import numpy as np
import pandas as pd
from scipy.optimize import minimize

# FIXME:原始价格序列中空值表示当天无交易？在最开始把价格序列ffill可能不行？ 后面算波动率时填充的价格会影响波动率计算，其他的呢？
def calc_vol_scalar(price, point_size, capital=1000000, risk_target=0.16):
    '''
    根据设置的risk target 计算出的单一品种的标准仓位，即forecast为均值10时的仓位，单位是手。
    每个contract能提供的cash vol为pnl_vol * point_size
    算每个合约的cash vol 方法为长期和短期的weighted average std
    '''
    # TODO: price有空值的时候,会导致空值前后的价格无用，有问题。但ffill会导致0出现，降低实际波动率。应该dropna再计算波动率？
    pnl_vol = calc_mixed_volatility(price.diff(), slow_vol_years=10)
    daily_risk_target = risk_target / (256 ** 0.5)
    vol_scalar = (capital * daily_risk_target) / (pnl_vol * point_size)
    return vol_scalar


def calc_position(forecast, vol_scalar, buffer_size=0):
    position_raw = calc_raw_position(forecast, vol_scalar)
    if buffer_size > 0:
        if isinstance(position_raw, pd.DataFrame):
            position_raw = position_raw.apply(lambda x: buffer_position(x, vol_scalar, buffer_size))
        else:
            position_raw = buffer_position(position_raw, vol_scalar, buffer_size)
    position = position_raw.shift(1)  # 当天的价格只能算出第二天的Position
    return position


def calc_raw_position(forecast, vol_scalar, forecast_scaling=10):
    # FIXME: 为什么除以10，forecast 的平均值并不是10，假设forecast 和10 差很多， 那岂不是永远达不到target position
    return forecast.mul(vol_scalar, axis=0) / forecast_scaling


def buffer_position(position, vol_scalar, buffer_size=0.10, trade_to_edge=True):
    # TODO: buffer操作后的position，会把没上市的品种的权重从na变为0，和原始的position不一致。position的na该如何约定？
    # vol_scalar 的另一种理解是Avg pos of the subsystem level，就是说position 可以在avg pos的10% 区间内浮动
    buffer = vol_scalar * buffer_size
    top = (position + buffer).ffill().round()
    bottom = (position - buffer).ffill().round()
    rounded = position.ffill().round()

    last = 0.0
    buffered_list = []
    for index in range(len(rounded)):
        last = adjust_by_buffer(last, rounded.iloc[index], top.iloc[index], bottom.iloc[index], trade_to_edge)
        buffered_list.append(last)
    buffered = pd.Series(buffered_list, index=rounded.index)
    return buffered


def adjust_by_buffer(last, current, top, bottom, trade_to_edge=True):
    if np.isnan(top) or np.isnan(bottom) or np.isnan(current):
        return last
    if trade_to_edge:
        return min(max(last, bottom), top)  # 如果在buffer内则不调仓，调仓就调到buffer边缘，尽量减少调仓幅度
    else:
        return last if (bottom <= last <= top) else current  # 如果在buffer内则不调仓


def calc_gross(position, price, point_size):
    # FIXME 源代码确实是shift 了两次，没看出来为什么
    position = position.shift(1)
    pnl_in_points = position.mul(price.ffill().diff(), axis=0).fillna(0)
    return pnl_in_points * point_size


def combine_forecast(forecast, forecast_weights, forecast_div_mult):
    combined_forecast = ((forecast_weights * forecast).sum(axis=1) * forecast_div_mult).clip(20, -20)
    return combined_forecast


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
    # 滑点目前当成一个Constant, 前提 1.高流动性市场 2.小规模交易
    slippage = info['spread_cost']
    point_size = info['point_size']
    return abs(quantity) * point_size * slippage


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
    # TODO:cost_sr是daily的，那么gross也必须是daily的，检查一下
    gross = gross.resample("1B").last()
    gross = gross.replace(0.0, np.nan)
    cost_daily = cost_sr * (gross.std() / 16)
    return gross + cost_daily


def calc_cost_sr(gross, cost, cost_multiplier=1):
    gross.replace(0.0, pd.NA, inplace=True)
    vol_daily = gross.std()
    costs_daily = cost.mean()
    cost_sr_daily = 16 * costs_daily / vol_daily
    cost_sr = cost_sr_daily * cost_multiplier
    return cost_sr


# def calc_net_pnl(gross_pnl, daily_costs):
#     raw_net = gross_pnl.add(daily_costs, fill_value=0)
#     net = raw_net.groupby(level=0).resample('B', level=1).sum()
#     return net

# def calc_cost_sr1(gross, cost, cost_multiplier=1, turnover=None, turnover_average=None):
#     cost_sr = calc_cost_sr(gross, cost, cost_multiplier)
#     cost_sr = cost_sr * (turnover_average / turnover)
#     return cost_sr
def calc_net_(gross, cost):
    net = gross.add(cost, fill_value=0)
    return net.resample('B').sum()


def calc_weight_adjusted_position(instrument, weights, forecast, div_mult, vol_scalar,
                                  buffer_size=0.1, trade_to_edge=True):
    subsystem_position = calc_raw_position(forecast, vol_scalar)
    instrument_weight = weights[instrument]
    position_index = subsystem_position.index
    weight_adjusted_position = (subsystem_position
                                * instrument_weight.reindex(position_index, method='ffill')
                                * div_mult.reindex(position_index, method='ffill'))

    buffer = (vol_scalar.reindex(position_index, method='ffill')
              * instrument_weight.reindex(position_index, method='ffill')
              * div_mult.reindex(position_index, method='ffill')
              * buffer_size)


    top = (weight_adjusted_position + buffer).ffill().round()
    bottom = (weight_adjusted_position - buffer).ffill().round()
    rounded = weight_adjusted_position.ffill().round()

    last = 0.0
    buffered_list = []
    for index in range(len(rounded)):
        last = adjust_by_buffer(last, rounded.iloc[index], top.iloc[index], bottom.iloc[index], trade_to_edge)
        buffered_list.append(last)
    buffered = pd.Series(buffered_list, index=rounded.index)

    return buffered


