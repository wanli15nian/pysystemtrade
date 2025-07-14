from sysdata.config.configdata import Config
from sysdata.sim.csv_futures_sim_data import csvFuturesSimData
from systems.accounts.accounts_stage import Account
from systems.basesystem import System
from systems.forecast_combine import ForecastCombine
from systems.forecast_scale_cap import ForecastScaleCap
from systems.forecasting import Rules
from systems.positionsizing import PositionSizing
from systems.provided.rules.ewmac import ewmac_forecast_with_defaults as ewmac
from systems.rawdata import RawData
from systems.trading_rules import TradingRule

data = csvFuturesSimData()
my_config = Config()
my_account = Account()
combiner = ForecastCombine()
raw_data = RawData()
position_size = PositionSizing()
fcs = ForecastScaleCap()

ewmac_8 = TradingRule((ewmac, [], dict(Lfast=8, Lslow=32)))
ewmac_32 = TradingRule(dict(function=ewmac, other_args=dict(Lfast=32, Lslow=128)))
my_rules = Rules(dict(ewmac8=ewmac_8, ewmac32=ewmac_32))

my_system = System([my_account, fcs, my_rules, combiner, raw_data, position_size], data, my_config)

my_config.percentage_vol_target = 16
my_config.notional_trading_capital = 1000000
my_config.base_currency = "USD"
my_config.instruments = ["US10", "SOFR", "CORN", "SP500_micro"]
my_config.use_forecast_scale_estimates = True
my_config.use_forecast_weight_estimates = True
my_config.use_forecast_div_mult_estimates = True
my_config.forecast_scalar_estimate["pool_instruments"] = False


'''
fc = my_system.forecastScaleCap.get_forecast_scalar("US10", "ewmac32")
print(fc.tail(5))

capped_forecast = my_system.forecastScaleCap.get_capped_forecast('US10', 'ewmac32')

gr = my_system.accounts.pandl_for_instrument_forecast('US10', 'ewmac32')
print(gr.tail(5))

forecast_weights = my_system.combForecast.get_forecast_weights("US10")
print(forecast_weights.mean())

forecast_div_mult = my_system.combForecast.get_forecast_diversification_multiplier("US10")
print(forecast_div_mult.mean())
'''

subsystem_position = my_system.positionSize.get_subsystem_position("US10")
print(subsystem_position.mean())