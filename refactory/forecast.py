import numpy as np


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
    vol_min = vol.rolling(min_periods=floor_min_periods, window=floor_days).quantile(floor_min_quant)
    vol_min.iloc[0] = 0.0
    vol_min.ffill(inplace=True)
    return np.maximum(vol, vol_min)


def rescale_forecast(forecast, target_scaling=10, upper_cap=20, window=250000, min_period=500):
    average = forecast.abs().rolling(window=window, min_periods=min_period).mean()
    scalar = (target_scaling / average).bfill()  # TODO:向过去填充是否有使用未来数据的问题，应该是向前填充？
    rescaled = scalar * forecast
    capped = rescaled.clip(lower=(-upper_cap), upper=upper_cap)
    return capped
