import pandas as pd
import csv


def get_instrument_info(instrument_code, file_path='data/csvconfig/instrumentconfig.csv'):
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['Instrument'] == instrument_code:
                metadata = {k:v for k, v in row.items() if k != 'Instrument'}
                info_df = pd.DataFrame.from_dict(metadata, orient='index', columns=[instrument_code])
                return info_df


def get_daily_prices(instrument_code):
    file = pd.read_csv('data/adjusted_prices_csv/'+instrument_code+'.csv', parse_dates=['DATETIME'])
    df = file.set_index('DATETIME')
    daily_prices = df.resample('1B').last()
    return daily_prices


def get_spread_cost(instrument_code, file_path='data/csvconfig/spreadcosts.csv'):
    df = pd.read_csv(file_path)
    row = df[df['Instrument'] == instrument_code]

    if row.empty:
        raise ValueError(f"Instrument '{instrument_code}' not found in {file_path}")

    spread_cost = row['SpreadCost'].values[0]
    return pd.DataFrame({instrument_code: [spread_cost]}, index=['SpreadCost'])


def get_rolls_per_year(instrument_code, file_path='data/csvconfig/rollconfig.csv'):
    df = pd.read_csv(file_path)
    row = df[df['Instrument'] == instrument_code]

    if row.empty:
        raise ValueError(f"Instrument '{instrument_code}' not found in {file_path}")

    rolls_per_year = len(row['HoldRollCycle'].values[0])
    return pd.DataFrame({instrument_code: [rolls_per_year]}, index=['rolls_per_year'])


def get_raw_cost_data(instrument_code):
    instr_data = get_instrument_info(instrument_code)
    spread_cost = get_spread_cost(instrument_code)
    raw_cost = compile_raw_cost_data(instr_data, spread_cost)
    return raw_cost


def compile_raw_cost_data(instr_data, spread_cost):
    raw_cost = instr_data.loc[['PerBlock', 'Percentage', 'PerTrade']]
    raw_cost = pd.concat([raw_cost, spread_cost], axis=0)
    return raw_cost