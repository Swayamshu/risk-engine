from typing import List, Optional

from pydantic import BaseModel, Field


class Holding(BaseModel):
    symbol: str
    weight: float = Field(..., gt=0, le=1)
    quantity: Optional[float] = None

class PortfolioRequest(BaseModel):
    portfolio_id: str
    holdings: List[Holding]
    benchmark: str = "SPY"
    analysis_period: str = "1Y"