import numpy as np
import pandas as pd

from refactory.data_util import get_daily_price
from refactory.forecast import ewmac, rescale_forecast, floor_vol, price_vol


def forecast_turnover_for_indiv_instr(instrument_code, rule_name):
    forecast_raw = get_capped_forecast(instrument_code, rule_name)
    forecast = forecast_raw.resample("1B").last()

    forecast_scalling = 10.0
    forecast_scalling_daily = pd.Series(np.full(forecast.shape[0], forecast_scalling), forecast.index)
    forecast_normalised = forecast / forecast_scalling_daily.ffill()

    avg_daily = float(forecast_normalised.diff().abs().mean())
    annual_turnover_for_forecast = avg_daily * 256

    print('forecast_turnover_for_individual_instrument')
    return annual_turnover_for_forecast


def get_capped_forecast(instrument, rule_name):
    '''
    Forecast 不是对当天价格的预判
    Forecast 根据包括当天在内的价格数据，对未来趋势进行判断
    究竟趋势如何就根据过去几天的价格变化
    '''
    price = get_daily_price(instrument)
    if rule_name == 'ewmac32':
        raw_ewmac32 = ewmac(price, 32, 128, 1)
        ewmac32 = rescale_forecast(raw_ewmac32 / floor_vol(price_vol(price)))
        ewmac32.rename('ewmac32', inplace=True)
        return ewmac32
    if rule_name == 'ewmac8':
        raw_ewmac8 = ewmac(price, 8, 32, 1)
        ewmac8 = rescale_forecast(raw_ewmac8 / floor_vol(price_vol(price)))
        ewmac8.rename('ewmac8', inplace=True)
        return ewmac8
    else:
        raise 'Rule not defined '
