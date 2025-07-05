import pandas as pd
import csv


'''
以下全部都是为了Instrument Info
'''
class Instrument_Meta_Data():
    def __init__(self, data_dict):
        for key, value in data_dict.items():
            setattr(self, key, value)

class Instrument_Info:
    def __init__(self, instrument, metadata_dict):
        self.instrument = instrument
        self.metadata = Instrument_Meta_Data(metadata_dict)

def get_instrument_info(instrument_code, file_path='data/csvconfig/instrumentconfig.csv'):
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['Instrument'] == instrument_code:
                metadata = {k:v for k, v in row.items() if k != 'Instrument'}
    return Instrument_Info(instrument_code, metadata)


###############################################################################
def get_daily_prices(instrument_code):
    file = pd.read_csv('data/adjusted_prices_csv/'+instrument_code+'.csv', parse_dates=['DATETIME'])
    df = file.set_index('DATETIME')
    daily_prices = df.resample('1B').last()
    return daily_prices





