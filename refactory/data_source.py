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


if __name__ == '__main__':
    # a = get_daily_price("CORN")
    # b = data_util.get_daily_price('CORN')
    # c = a[~a.index.isin(b.index)]

    info = get_instrument_info()
    ins = 'CORN'
    corn = info.loc[ins]
    print(corn)
