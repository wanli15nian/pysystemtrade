import datetime
from dataclasses import dataclass

import numpy as np
import pandas as pd

from refactory.position_pnl import calc_trade_cost


def calc_cost(position, price, info, include_slippage=True):
    rolls_per_year = int(info['rolls_per_year'])  # TODO: 用【】取会自动转为浮点型，临时方案是强制给转成整型
    all_fills = calc_all_fills(position, price, rolls_per_year)
    cost_deflator = calc_cost_deflator(price)
    normalised_costs = calc_normalised_cost(info, all_fills, cost_deflator, include_slippage)
    return normalised_costs


@dataclass
class Fill:
    date: datetime.datetime
    qty: int
    price: float
    price_requires_slippage_adjustment: bool = False


def calc_normalised_cost(info, all_fills, cost_deflator, include_slippage):
    instrument_currency_costs = [-calc_trade_cost(fill.price, fill.qty, info, include_slippage)
                                 for fill in all_fills]
    date_index = [fill.date for fill in all_fills]
    costs_as_pd_series = pd.Series(instrument_currency_costs, date_index)
    costs_as_pd_series = costs_as_pd_series.sort_index()
    costs_as_pd_series = costs_as_pd_series.groupby(costs_as_pd_series.index).sum()
    reindexed_deflator = cost_deflator.reindex(costs_as_pd_series.index, method="ffill")
    normalised_costs = reindexed_deflator * costs_as_pd_series
    return normalised_costs


@dataclass
class Fill:
    date: datetime.datetime
    qty: int
    price: float


def calc_all_fills(position, price, rolls_per_year):
    list_of_years = list(set([int(idx.year) for idx in position.index]))
    list_of_years.sort()

    fills_by_year = [pseudo_fills_for_year(year, rolls_per_year, price, position) for year in
                     list_of_years]
    list_of_holding_fills = [item for sublist in fills_by_year for item in sublist]
    trades = position.diff()
    trades_without_na = trades[~trades.isna()]
    trades_without_zeros = trades_without_na[trades_without_na != 0]
    prices_aligned_to_trades = price.reindex(trades_without_zeros.index, method="ffill")
    trades_as_list = list(trades_without_zeros.values)
    prices_as_list = list(prices_aligned_to_trades.values)
    dates_as_list = list(prices_aligned_to_trades.index)
    list_of_trading_fills = [
        Fill(date, qty, price)
        for date, qty, price in zip(dates_as_list, trades_as_list, prices_as_list)
    ]
    list_of_all_fills = list_of_trading_fills + list_of_holding_fills
    return list_of_all_fills


def calc_cost_deflator(price):
    daily_price = price.resample("1B").ffill()
    vol = daily_price.diff().rolling(180, min_periods=3).std()
    return vol / vol.iloc[-1]


def pseudo_fills_for_year(year, rolls_per_year, price, positions):
    if rolls_per_year == 0:
        return []

    date_list = generate_equal_dates_within_year(year, rolls_per_year)


    # 计算每年的平均持有量
    first_date = generate_equal_dates_within_year(year - 1, rolls_per_year)[-1]
    subsequent_dates = date_list
    all_dates = [first_date] + subsequent_dates
    list_of_average_holdings = []
    for date_index in range(len(subsequent_dates)):
        end_date = all_dates[date_index + 1]
        previous_date = all_dates[date_index]
        avg_holding = positions[previous_date:end_date].abs().mean()
        if np.isnan(avg_holding):
            avg_holding = 0.0
        list_of_average_holdings.append(avg_holding)

    # 填充价格序列
    price_series = price.ffill()
    # 获取价格序列的最后一个日期
    last_date_with_positions = price.index[-1]
    multiply_roll_costs_by = 1

    ## We multiply the quantity rather than the actual costs, as the later
    ##   cost calculation doesn't distinguish between rolls and other trades

    opening_fills_this_year = [
        Fill(
            date=date,
            qty=qty * multiply_roll_costs_by,
            price=get_row_of_series_before_date(price_series, date),
        )
        for date, qty in zip(date_list, list_of_average_holdings)
        if date <= last_date_with_positions and abs(qty) > 0
    ]

    closing_fills_this_year = [Fill(
        date=fill.date,
        qty=-fill.qty,
        price=fill.price) for fill in opening_fills_this_year]

    fills_this_year = opening_fills_this_year + closing_fills_this_year

    return fills_this_year


def generate_equal_dates_within_year(year, rolls_per_year, align_to_start=True):
    days_of_roll = int(365 / rolls_per_year)
    first_date = datetime.datetime(year, 1, 1)
    if not align_to_start:
        first_date = first_date + datetime.timedelta(days=int(days_of_roll / 2))
    all_dates = [first_date + (datetime.timedelta(days=days_of_roll) * period_count)
                 for period_count in range(rolls_per_year)]
    return all_dates


def get_row_of_series_before_date(data_series, relevant_date):
    if relevant_date == np.nan:
        data_at_date = data_series.values[-1]
    else:
        matching_index_size = data_series.index[data_series.index < relevant_date].size
        if matching_index_size == 0:
            index_point = None
        else:
            index_point = matching_index_size - 1
        data_at_date = data_series.values[index_point]
    return data_at_date
