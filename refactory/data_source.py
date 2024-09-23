import pandas as pd

from sysdata.sim.csv_futures_sim_data import csvFuturesSimData

source_data = csvFuturesSimData()


def get_instrument_info(instrument_code):
    return source_data.db_futures_instrument_data.get_instrument_data(instrument_code)


def get_daily_price(instrument_code):
    return source_data.daily_prices(instrument_code)


def get_spread_cost(instrument_code):
    return source_data.db_spread_cost_data.get_spread_cost(instrument_code)


def get_roll_parameters(instrument_code):
    return source_data.db_roll_parameters.get_roll_parameters(instrument_code)


def get_raw_carry_data(instrument_code):
    filename = '..\\data\\futures\\multiple_prices_csv\\' + instrument_code + '.csv'
    carry_data = pd.read_csv(filename)
    carry_price = carry_data['PRICE']
    carry_price.index = carry_data['DATETIME']
    carry_price.index = pd.to_datetime(carry_price.index)
    daily_carry_price = carry_price.resample('1B').last()
    return daily_carry_price
