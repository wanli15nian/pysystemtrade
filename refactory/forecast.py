import numpy as np
import pandas as pd


def ewmac(price, Lfast, Lslow, min_periods=1):
    fast = price.ewm(span=Lfast, min_periods=min_periods).mean()
    slow = price.ewm(span=Lslow, min_periods=min_periods).mean()
    result = fast - slow
    return result


def price_vol(price, span=35, min_periods=10, vol_abs_min=0.0000000001):
    vol = price.ewm(adjust=True, span=span, min_periods=min_periods).std()
    vol[vol < vol_abs_min] = vol_abs_min
    vol.ffill(inplace=True)
    return vol


def floor_vol(vol, floor_min_quant=0.05, floor_min_periods=100, floor_days=500):
    vol_min = vol.rolling(min_periods=floor_min_periods, window=floor_days).quantile(q=floor_min_quant)
    vol_min.iloc[0] = 0.0
    vol_min.ffill(inplace=True)
    return np.maximum(vol, vol_min)


def rescale_forecast(forecast, target_scaling=10, upper_cap=20, window=250000, min_period=500):
    average = forecast.abs().rolling(window=window, min_periods=min_period).mean()
    scalar = (target_scaling / average).bfill()  # FIXME:向过去填充是否有使用未来数据的问题，应该是向前填充？
    rescaled = scalar * forecast
    capped = rescaled.clip(lower=(-upper_cap), upper=upper_cap)
    return capped


def calculate_forecasts(price):
    raw_ewmac32 = ewmac(price, 32, 128, 1)
    ewmac32 = rescale_forecast(raw_ewmac32 / floor_vol(price_vol(price)))
    # ewmac32.rename('ewmac32', inplace=True)
    raw_ewmac8 = ewmac(price, 8, 32, 1)
    ewmac8 = rescale_forecast(raw_ewmac8 / floor_vol(price_vol(price)))
    # ewmac8.rename('ewmac8', inplace=True)
    forecast_df = pd.DataFrame({'ewmac32': ewmac32, 'ewmac8': ewmac8})
    return forecast_df
