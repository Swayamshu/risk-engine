from datetime import datetime
from typing import Dict, List

from pydantic import BaseModel


class SimulationRequest(BaseModel):
    portfolio_id: str
    simulation_days: int = 252
    num_simulations: int = 10000
    confidence_levels: List[float] = [0.95, 0.99]


class SimulationResponse(BaseModel):
    portfolio_id: str
    simulation_date: datetime
    num_simulations: int
    simulation_days: int
    confidence_intervals: Dict[str, Dict[str, float]]
    var_estimates: Dict[str, float]
    expected_returns: Dict[str, float]
    final_values: Dict[str, List[float]]
