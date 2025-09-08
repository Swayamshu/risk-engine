from datetime import datetime
from typing import List

from pydantic import BaseModel


class MarketDataResponse(BaseModel):
    symbol: str
    datetime: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    adjusted_close: float

class PriceHistory(BaseModel):
    symbol: str
    prices: List[MarketDataResponse]
    start_date: datetime
    end_date: datetime
