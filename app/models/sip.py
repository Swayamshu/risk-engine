from datetime import datetime
from typing import List

from pydantic import BaseModel

from app.models import Frequency


class SIPRequest(BaseModel):
    amount: float
    frequency: Frequency
    symbols: List[str]
    weights: List[float]
    start_date: datetime
    end_date: datetime


class SIPAnalysis(BaseModel):
    total_invested: float
    final_value: float
    cagr: float
    returns: float
    volatility: float
    sharpe_ratio: float
    investment_periods: int
    cost_averaging_benefit: float