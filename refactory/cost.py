import datetime
from dataclasses import dataclass

import numpy as np
import pandas as pd

from refactory.position_pnl import calc_trade_cost


def calc_cost(position, price, info, include_slippage=True):
    rolls_per_year = int(info['rolls_per_year'])
    all_fills = calc_all_fills(position, price, rolls_per_year)
    cost_deflator = calc_cost_deflator(price)

    # all_fill_list = [Fill(row['date'], row['quantity'], row['price']) for _, row in all_fills.iterrows()]
    # instrument_currency_costs = [-calc_trade_cost(fill.price, fill.qty, info, include_slippage)
    #                              for fill in all_fill_list]
    # date_index = [fill.date for fill in all_fill_list]
    # raw_costs = pd.Series(instrument_currency_costs, date_index)

    all_fills['cost'] = -all_fills.apply(
        lambda row: calc_trade_cost(row['price'], row['quantity'], info, include_slippage), axis=1)
    raw_costs = pd.Series(all_fills['cost'].values, index=all_fills['date'])

    raw_costs = raw_costs.sort_index()
    raw_costs = raw_costs.groupby(raw_costs.index).sum()

    cost_deflator = cost_deflator.reindex(raw_costs.index, method="ffill")
    normalised_costs = cost_deflator * raw_costs

    return normalised_costs


@dataclass
class Fill:
    date: datetime.datetime
    qty: int
    price: float
    price_requires_slippage_adjustment: bool = False


@dataclass
class Fill:
    date: datetime.datetime
    qty: int
    price: float


def calc_all_fills(position, price, rolls_per_year):
    list_of_years = list(set([int(idx.year) for idx in position.index]))
    list_of_years.sort()

    holding_fills = pd.concat([pseudo_fills_for_year(year, rolls_per_year, price, position) for year in
                               list_of_years])

    trades = position.diff().dropna()  # 计算持仓变化并去除缺失值
    trades = trades[trades != 0]  # 去除交易量为0的行
    prices = price.reindex(trades.index, method="ffill")
    trading_fills = pd.DataFrame({
        'date': trades.index,
        'quantity': trades.values,
        'price': prices.values
    })

    all_fills = pd.concat([trading_fills, holding_fills]).sort_values(by='date').reset_index(drop=True)
    return all_fills


def calc_cost_deflator(price):
    daily_price = price.resample("1B").ffill()
    vol = daily_price.diff().rolling(180, min_periods=3).std()
    return vol / vol.iloc[-1]


def pseudo_fills_for_year(year, rolls_per_year, price, positions):
    if rolls_per_year == 0:
        return []

    date_list = generate_equal_dates_within_year(year, rolls_per_year)

    last_year_date = generate_equal_dates_within_year(year - 1, rolls_per_year)[-1]
    dl = [last_year_date] + date_list

    df = pd.DataFrame({
        'date': date_list,
        'quantity': [positions[dl[i]:dl[i + 1]].abs().mean() for i in range(len(date_list))]
    })
    df.fillna(0, inplace=True)

    last_date_with_positions = price.index[-1]
    df = df[(df['date'] <= last_date_with_positions) & (df['quantity'].abs() > 0)]
    df['price'] = df['date'].map(lambda date: get_row_of_series_before_date(price, date))
    df_fills = pd.concat([df, df]).sort_values(by='date').reset_index(drop=True)

    return df_fills

    # average_holdings_series = pd.Series(
    #     [positions[dl[i]:dl[i + 1]].abs().mean() for i in range(len(date_list))],
    #     index=date_list)
    # average_holdings_series = average_holdings_series.fillna(0)

    # list_of_average_holdings = average_holdings_series.values
    # # 获取价格序列的最后一个日期
    # opening_fills_this_year = [
    #     Fill(
    #         date=date,
    #         qty=qty,
    #         price=get_row_of_series_before_date(price, date),
    #     )
    #     for date, qty in zip(date_list, list_of_average_holdings)
    #     if date <= last_date_with_positions and abs(qty) > 0
    # ]

    # # 从 DataFrame 转换为 Fill 对象列表
    # opening_fills_this_year = [
    #     Fill(date=row['date'], qty=row['quantity'], price=row['price'])
    #     for _, row in df.iterrows()
    # ]
    #
    # closing_fills_this_year = [Fill(
    #     date=fill.date,
    #     qty=-fill.qty,
    #     price=fill.price) for fill in opening_fills_this_year]
    #
    # fills_this_year = opening_fills_this_year + closing_fills_this_year


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
