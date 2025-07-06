import pandas as pd

from refactory.data_util import get_point_size, get_per_trade, get_per_block, get_percentage, get_spread_cost, \
    get_daily_price
from refactory.utils import calc_mixed_volatility


def get_cost_per_trade(instrument_code):
    # 单次交易成本，包括slippage和commission

    notional_blocks_traded = 1
    point_size = get_point_size(instrument_code)  # 指源代码中 get_value_of_block_price_move 返回的是point_size
    per_trade = get_per_trade(instrument_code)
    per_block = get_per_block(instrument_code)
    percentage = get_percentage(instrument_code)
    price_slippage = get_spread_cost(instrument_code)

    price = get_daily_price(instrument_code)

    # FIXME: 在这里作者使用了pd.DateOffset来进行年份计算，而在rolling window中是用365天，原因存疑
    average_price = float(price[price.index[-1] - pd.DateOffset(years=1):].mean())
    commission_percentage = notional_blocks_traded * average_price * point_size * percentage
    commission_per_block = notional_blocks_traded * per_block
    commission = max([per_trade, commission_per_block, commission_percentage])
    slippage = notional_blocks_traded * price_slippage * point_size
    cost = commission + slippage

    vol_daily = calc_mixed_volatility(price.diff(), slow_vol_years=10)
    vol_daily_average = float(vol_daily[price.index[-1] - pd.DateOffset(years=1):].mean())
    ann_std = vol_daily_average * 16 * point_size

    cost_per_trade = cost / ann_std

    return cost_per_trade
