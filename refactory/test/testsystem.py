
from sysdata.sim.csv_futures_sim_data import csvFuturesSimData

data = csvFuturesSimData()

from systems.provided.rules.ewmac import ewmac_forecast_with_defaults as ewmac

from systems.forecasting import Rules

from systems.basesystem import System

from systems.forecast_combine import ForecastCombine
from systems.accounts.accounts_stage import Account
from systems.rawdata import RawData
from systems.positionsizing import PositionSizing
from sysdata.config.configdata import Config

my_config = Config()
my_config


my_account = Account()
combiner = ForecastCombine()
raw_data = RawData()
position_size = PositionSizing()
possizer = PositionSizing()
my_config.percentage_vol_target = 25
my_config.notional_trading_capital = 500000
my_config.base_currency = "USD"


from systems.trading_rules import TradingRule
ewmac_8 = TradingRule((ewmac, [], dict(Lfast=8, Lslow=32)))
ewmac_32 = TradingRule(dict(function=ewmac, other_args=dict(Lfast=32, Lslow=128)))
my_rules = Rules(dict(ewmac8=ewmac_8, ewmac32=ewmac_32))

from systems.forecast_scale_cap import ForecastScaleCap

my_config.instruments = ["US10", "SOFR", "CORN", "SP500_micro"]
my_config.use_forecast_scale_estimates = True

fcs = ForecastScaleCap()
my_system = System([fcs, my_rules], data, my_config)
my_config.forecast_scalar_estimate["pool_instruments"] = False
print(my_system.forecastScaleCap.get_forecast_scalar("SOFR", "ewmac32").tail(5))

fcs = ForecastScaleCap()
my_system = System([fcs, my_rules], data, my_config)
print(my_system.forecastScaleCap.get_capped_forecast("SOFR", "ewmac32").tail(5))


my_config.forecast_weight_estimate = dict(method="one_period")
my_config.use_forecast_weight_estimates = True
my_config.use_forecast_div_mult_estimates = True

my_system = System(
    [my_account, fcs, my_rules, combiner, raw_data, position_size], data, my_config
)

print(my_system.combForecast.get_forecast_weights("US10").mean())
# print(my_system.combForecast.get_forecast_diversification_multiplier("US10").tail(5))