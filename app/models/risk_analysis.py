from datetime import datetime
from typing import Optional, List, Dict

from pydantic import BaseModel

from app.models import RiskLevel


class RiskMetrics(BaseModel):
    symbol: str
    volatility: Optional[float] = None
    beta: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    var_95: Optional[float] = None
    var_99: Optional[float] = None
    expected_shortfall_95: Optional[float] = None
    expected_shortfall_99: Optional[float] = None
    max_drawdown: Optional[float] = None
    sortino_ratio: Optional[float] = None


class PortfolioRiskAnalysis(BaseModel):
    portfolio_id: str
    analysis_date: datetime
    volatility: float
    beta: float
    sharpe_ratio: float
    var_95: float
    var_99: float
    expected_shortfall_95: float
    expected_shortfall_99: float
    max_drawdown: float
    sortino_ratio: float
    risk_score: float
    risk_level: RiskLevel
    individual_assets: List[RiskMetrics]
    correlation_matrix: Optional[Dict]
