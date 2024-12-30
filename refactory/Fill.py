import datetime
from dataclasses import dataclass

@dataclass
class Fill:
    date: datetime.datetime
    qty: int
    price: float
    price_requires_slippage_adjustment: bool = False