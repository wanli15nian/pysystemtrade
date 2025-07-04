import pandas as pd

from refactory.data_source import get_daily_price, get_raw_carry_price, get_point_size, get_raw_cost_data, \
    get_roll_parameters
from refactory.utils import forecast_turnover_for_indiv_instr, get_capped_forecast
from refactory.temp import calc_pos_target_from_risk_target


def prepare_all_instr_data(all_instruments, trading_rule_list):
    all_instrument_data = {}
    for instrument in all_instruments:
        individual_instr_data = {}
        forecast_df = {}
        for rule_name in trading_rule_list:
            forecast = get_capped_forecast(instrument, rule_name)
            forecast_df[rule_name] = forecast
        forecast_df = pd.DataFrame(forecast_df)
        individual_instr_data['forecast_df'] = forecast_df

        turnover_dict = {}
        for rule_name in trading_rule_list:
            turnover = forecast_turnover_for_indiv_instr(instrument, rule_name)
            turnover_dict[rule_name] = turnover
        individual_instr_data['turnover_dict'] = turnover_dict

        price = get_daily_price(instrument)
        individual_instr_data['price'] = price

        carry_price = get_raw_carry_price(instrument)
        individual_instr_data['carry_price'] = carry_price

        point_size = get_point_size(instrument)
        individual_instr_data['point_size'] = point_size

        position_target = calc_pos_target_from_risk_target(price, point_size, capital=1000000, risk_target=0.16)
        individual_instr_data['position_target'] = position_target

        raw_costs = get_raw_cost_data(instrument)
        individual_instr_data['raw_costs'] = raw_costs

        value_per_point = get_point_size(instrument)
        individual_instr_data['value_per_point'] = value_per_point

        rolls_per_yr = get_roll_parameters(instrument).rolls_per_year_in_hold_cycle()
        individual_instr_data['rolls_per_year'] = rolls_per_yr

        all_instrument_data[instrument] = individual_instr_data
    return all_instrument_data
