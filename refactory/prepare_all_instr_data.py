from refactory.cost_forecast import instrument_forecast_turnover

from refactory.data_util import get_daily_price, get_raw_carry_price, get_point_size, get_raw_cost_data, \
    get_rolls_per_year
from refactory.forecast import calculate_forecasts
from refactory.target_volatility import calc_target_position


def prepare_all_instr_data(all_instruments, trading_rule_list):
    all_instrument_data = {}
    for instrument in all_instruments:
        individual_instr_data = {}

        price = get_daily_price(instrument)
        individual_instr_data['price'] = price

        forecast_df = calculate_forecasts(price)
        individual_instr_data['forecast_df'] = forecast_df

        turnover_dict = {}
        for rule_name in trading_rule_list:
            turnover = instrument_forecast_turnover(instrument, rule_name)
            turnover_dict[rule_name] = turnover
        individual_instr_data['turnover_dict'] = turnover_dict

        carry_price = get_raw_carry_price(instrument)
        individual_instr_data['carry_price'] = carry_price

        point_size = get_point_size(instrument)
        individual_instr_data['point_size'] = point_size

        position_target = calc_target_position(price, point_size, capital=1000000, risk_target=0.16)
        individual_instr_data['position_target'] = position_target

        raw_costs = get_raw_cost_data(instrument)
        individual_instr_data['raw_costs'] = raw_costs

        value_per_point = get_point_size(instrument)
        individual_instr_data['value_per_point'] = value_per_point

        rolls_per_yr = get_rolls_per_year(instrument)
        individual_instr_data['rolls_per_year'] = rolls_per_yr

        all_instrument_data[instrument] = individual_instr_data
    return all_instrument_data
