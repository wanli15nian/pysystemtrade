import pandas as pd


def get_daily_price(instrument_code):
    df = pd.read_csv('data/adjusted_prices_csv/' + instrument_code + '.csv', parse_dates=['DATETIME'],
                     index_col='DATETIME')
    daily_price = df.resample('1B').last()
    daily_price.columns = ['price']
    daily_price.index.name = 'datetime'
    return daily_price['price']


def get_instrument_info():
    i0 = load_instrument_config()
    i = i0[['Percentage', 'PerBlock', 'PerTrade', 'Pointsize']]
    i.columns = ['percentage', 'per_block', 'per_trade', 'point_size']

    r0 = load_roll_config()
    r = r0['HoldRollCycle'].apply(len).astype('Int64')
    r.name = 'rolls_per_year'

    c0 = load_spread_cost()
    c = c0[['SpreadCost']]
    c.columns = ['spread_cost']

    result = pd.concat([i, r, c], axis=1)
    return result


def load_instrument_config():
    file_path = 'data/csvconfig/instrumentconfig.csv'
    df = pd.read_csv(file_path, index_col='Instrument')
    df.index.name = 'instrument'
    return df


def load_spread_cost():
    file_path = 'data/csvconfig/spreadcosts.csv'
    df = pd.read_csv(file_path, index_col='Instrument')
    df.index.name = 'instrument'
    return df


def load_roll_config():
    file_path = 'data/csvconfig/rollconfig.csv'
    df = pd.read_csv(file_path, index_col='Instrument')
    df.index.name = 'instrument'
    return df


#
# def get_rolls_per_year(instrument_code, file_path='data/csvconfig/rollconfig.csv'):
#     df = pd.read_csv(file_path)
#     row = df[df['Instrument'] == instrument_code]
#
#     if row.empty:
#         raise ValueError(f"Instrument '{instrument_code}' not found in {file_path}")
#
#     rolls_per_year = len(row['HoldRollCycle'].values[0])
#     return pd.DataFrame({instrument_code: [rolls_per_year]}, index=['rolls_per_year'])
#
#
# def get_raw_cost_data(instrument_code):
#     instr_data = load_instrument_config(instrument_code)
#     spread_cost = get_spread_cost(instrument_code)
#     cost = instr_data.loc[['PerBlock', 'Percentage', 'PerTrade']]
#     cost = pd.concat([cost, spread_cost], axis=0)
#     raw_cost = cost
#     return raw_cost
#
#
# def get_spread_cost(instrument_code, file_path='data/csvconfig/spreadcosts.csv'):
#     df = pd.read_csv(file_path)
#     row = df[df['Instrument'] == instrument_code]
#
#     if row.empty:
#         raise ValueError(f"Instrument '{instrument_code}' not found in {file_path}")
#
#     spread_cost = row['SpreadCost'].values[0]
#     return pd.DataFrame({instrument_code: [spread_cost]}, index=['SpreadCost'])
#
#
# def get_instrument_config(instrument_code, file_path='data/csvconfig/instrumentconfig.csv'):
#     with open(file_path, newline='', encoding='utf-8') as csvfile:
#         reader = csv.DictReader(csvfile)
#         for row in reader:
#             if row['Instrument'] == instrument_code:
#                 metadata = {k: v for k, v in row.items() if k != 'Instrument'}
#                 info_df = pd.DataFrame.from_dict(metadata, orient='index', columns=[instrument_code])
#                 return info_df


if __name__ == '__main__':
    # a = get_daily_price("CORN")
    # b = data_util.get_daily_price('CORN')
    # c = a[~a.index.isin(b.index)]

    info = get_instrument_info()
    ins = 'CORN'
    corn = info.loc[ins]
    print(corn)
    import data_util as u

    print(u.get_percentage(ins), u.get_per_block(ins), u.get_per_trade(ins), u.get_point_size(ins),
          u.get_rolls_per_year(ins), u.get_spread_cost(ins))
