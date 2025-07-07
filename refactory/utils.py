import numpy as np
import pandas as pd
from scipy.optimize import minimize


def calc_mixed_volatility(data, days=35, min_periods=10, slow_vol_years=20,
                          perc_of_long_vol=0.3, vol_min=0.0000000001,
                          vol_multiplier=1.0):
    # 长期和短期波动进行权重处理
    short_vol = data.ewm(adjust=True, span=days, min_periods=min_periods).std()
    long_vol = short_vol.ewm(adjust=True, span=slow_vol_years * 256).mean()
    vol = perc_of_long_vol * long_vol + (1 - perc_of_long_vol) * short_vol
    vol[vol < vol_min] = vol_min
    vol = vol * vol_multiplier
    return vol


def robust_vol_calc(daily_returns: pd.Series,
                    days: int = 35,
                    min_periods: int = 10,
                    vol_abs_min: float = 0.0000000001,
                    vol_floor: bool = True,
                    floor_min_quant: float = 0.05,
                    floor_min_periods: int = 100,
                    floor_days: int = 500,
                    backfill: bool = False, ):
    vol = daily_returns.ewm(adjust=True, span=days, min_periods=min_periods).std()
    vol[vol < vol_abs_min] = vol_abs_min

    if vol_floor:
        vol_min = vol.rolling(min_periods=floor_min_periods, window=floor_days).quantile(q=floor_min_quant)
        vol_min.iloc[0] = 0.0
        vol_min.ffill(inplace=True)
        vol = np.maximum(vol, vol_min)
    if backfill:
        # use the first vol in the past, sort of cheating
        vol_forward_fill = vol.ffill()
        vol = vol_forward_fill.bfill()
    return vol


def get_stdev_estimator_for_instrument_weight(data, fit_end, span=50000, min_periods=5):
    stdev_smoothed = data.ewm(span=span, min_periods=min_periods).std()
    last_index = data.index[data.index < fit_end].size - 1
    stdev = stdev_smoothed.iloc[last_index]
    stdev_list = stdev * ((365.25 / 7.0) ** 0.5)
    avg_stdev = np.nanmean(stdev_list)
    norm_stdev = [avg_stdev] * len(stdev_list)
    norm_factor = [stdev / avg_stdev for stdev in stdev_list]
    return norm_stdev, norm_factor


def get_mean_estimator(data, fit_end, span=50000, min_periods=10):
    mean = data.ewm(span=span, min_periods=min_periods).mean()  # 逻辑还是config 的4倍
    last_index = data.index[data.index < fit_end].size - 1
    mean = mean.iloc[last_index]
    annualised_mean_estimate = {}
    for rule_name, mean_value in mean.items():
        annualised_mean_estimate[rule_name] = mean_value * 365.25 / 7.0
    mean_list = [value for value in annualised_mean_estimate.values()]
    return mean_list


def get_corr_estimator_for_instrument_weight(data, fit_end, span=500000, min_periods=10):
    raw_corr = data.ewm(span=span, min_periods=min_periods, ignore_na=True).corr(
        pairwise=True)  # span 和min_periods 都是config 里面的4倍，因为4个instruments
    columns = data.columns
    size_of_matrix = len(columns)
    corr_matrix_values = (raw_corr[raw_corr.index.get_level_values(0) < fit_end].tail(
        size_of_matrix).values)  # 截取fit_period之前的数据
    corr_matrix_values = [[max(0, item) for item in sublist] for sublist in corr_matrix_values]
    return corr_matrix_values


def optimisation(number, corr, norm_mean, norm_stdev):
    def addem(weights):
        return 1.0 - sum(weights)

    def neg_SR(weights, sigma, mus):
        estimated_returns = np.dot(weights, mus)[0]
        stdev = weights.dot(sigma).dot(weights.transpose()) ** 0.5
        sr = -estimated_returns / stdev
        return sr

    mus = np.array(norm_mean, ndmin=2).transpose()  # mus 没问题
    sigma = np.diag(norm_stdev).dot(corr).dot(np.diag(norm_stdev))
    start_weights = np.array([1 / number] * number)
    bounds = [(0.0, 1.0)] * number
    cdict = [{"type": "eq", "fun": addem}]
    ans = minimize(neg_SR, start_weights, (sigma, mus), method='SLSQP', constraints=cdict, bounds=bounds, tol=0.00001)
    weight = ans['x']
    return weight


def single_resampled_set_of_returns(data_dict, frequency: str):
    data_resampled = [pnl.resample(frequency).sum() for pnl in data_dict.values()]

    all_indices = [data_item.index for data_item in data_resampled]
    flattened = [item for sublist in all_indices for item in sublist]
    common_index = list(set(flattened))
    common_index.sort()

    reindexed_data = [data_item.reindex(common_index) for data_item in data_resampled]

    for offset_value, data_item in enumerate(reindexed_data):
        data_item.index = data_item.index + pd.Timedelta("%dus" % offset_value)

    stacked_data = pd.concat(reindexed_data, axis=0).sort_index()
    return stacked_data


def calc_volatility_scalar(price, point_size, capital, annual_perc_vol_target):
    '''
    Get ratio of required volatility vs volatility of instrument in instrument's own currency

    Gets daily prices for use with % volatility
    This won't always be the same as the normal 'price'
    try:
        prices = self.get_instrument_raw_carry_data(instrument_code).PRICE
    except missingData:
        self.log.warning(
            "No carry data found for %s, using adjusted prices to calculate percentage returns"
            % instrument_code
        )
        return self.get_daily_prices(instrument_code)
    '''
    carry_price = price
    block_value = carry_price.ffill() * 0.01 * point_size
    block_value.ffill(inplace=True)
    # FIXME: When to use carry_price and when to use price, the logic of computation here is unknown
    resampled_carry_price = carry_price.resample('1B').last()
    annualised_price_vol_points = calc_mixed_volatility(price.diff(), slow_vol_years=10)
    annualised_price_vol_points.ffill(inplace=True)
    # Align resampled carry price and annualised price volatility in points
    (resampled_carry_price, annualised_price_vol_points) = resampled_carry_price.align(annualised_price_vol_points,
                                                                                       join='right')
    percentage_vol = 100.0 * (annualised_price_vol_points / resampled_carry_price.ffill().abs())
    (block_value, percentage_vol) = block_value.align(percentage_vol, join="inner")
    currency_vol = block_value * percentage_vol
    # It is to multiply by fx_rate, which is taken to be 1 here
    value_vol = currency_vol.ffill() * 1
    perc_vol_target = annual_perc_vol_target / 16
    cash_vol_target = capital * perc_vol_target
    vol_scalar = cash_vol_target / value_vol
    vol_scalar = vol_scalar.reindex(price.index, method="ffill")
    return vol_scalar


# Define a function called multi_to_list that takes in an input
def multi_to_list(input):
    grouped = input.groupby(level='instrument')
    result = [group.reset_index(level='instrument', drop=True) for instrument, group in grouped]
    return result
