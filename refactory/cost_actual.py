import datetime

import pandas as pd

from refactory.base import calc_cost_of_fill


def calc_cost_actual(position, price, info, include_slippage=True):
    rolls_per_year = int(info['rolls_per_year'])
    all_fills = calc_all_fills(position, price, rolls_per_year)

    all_fills['cost'] = -all_fills.apply(
        lambda row: calc_cost_of_fill(row['price'], info, row['quantity'], include_slippage), axis=1)
    fill_cost = pd.Series(all_fills['cost'].values, index=all_fills['date']).sort_index()
    raw_costs = fill_cost.groupby(fill_cost.index).sum()

    daily_price = price.resample("1B").ffill()
    vol = daily_price.diff().rolling(180, min_periods=3).std()
    cost_deflator = vol / vol.iloc[-1]
    cost_deflator = cost_deflator.reindex(raw_costs.index, method="ffill")

    return cost_deflator * raw_costs


def calc_all_fills(position, price, rolls_per_year):
    list_of_years = list(set([int(idx.year) for idx in position.index]))
    list_of_years.sort()
    holding_fills = pd.concat([pseudo_holding_fills(year, rolls_per_year, price, position) for year in
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


def pseudo_holding_fills(year, rolls_per_year, price, positions):
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
    df['price'] = price.asof(df['date'])
    # df['price'] = df['date'].map(lambda date: get_row_of_series_before_date(price, date))

    df_fills = pd.concat([df, df]).sort_values(by='date').reset_index(drop=True)

    return df_fills


def generate_equal_dates_within_year(year, rolls_per_year, align_to_start=True):
    days_of_roll = int(365 / rolls_per_year)
    first_date = datetime.datetime(year, 1, 1)
    if not align_to_start:
        first_date = first_date + datetime.timedelta(days=int(days_of_roll / 2))
    all_dates = [first_date + (datetime.timedelta(days=days_of_roll) * period_count)
                 for period_count in range(rolls_per_year)]
    return all_dates

#
#
# def get_row_of_series_before_date(data_series, relevant_date):
#     if relevant_date == np.nan:
#         data_at_date = data_series.values[-1]
#     else:
#         matching_index_size = data_series.index[data_series.index < relevant_date].size
#         if matching_index_size == 0:
#             index_point = None
#         else:
#             index_point = matching_index_size - 1
#         data_at_date = data_series.values[index_point]
#     return data_at_date
