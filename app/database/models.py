from sqlalchemy import Integer, Column, String, DateTime, Float, JSON, Text
from sqlalchemy.sql.functions import func

from app.database.connection import Base


class MarketData(Base):
    __tablename__ = 'market_data'

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), index=True, nullable=False)
    date = Column(DateTime, index=True, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)
    adjusted_close = Column(Float, nullable=False)
    created_at = Column(DateTime, server_default=func.now())

class RiskAnalysis(Base):
    __tablename__ = 'risk_analysis'

    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(String(50), index=True, nullable=False)
    analysis_date = Column(DateTime, nullable=False)
    volatility = Column(Float, nullable=True)
    beta = Column(Float, nullable=True)
    sharpe_ratio = Column(Float, nullable=True)
    var_95 = Column(Float, nullable=True)
    var_99 = Column(Float, nullable=True)
    expected_shortfall_95 = Column(Float, nullable=True)
    expected_shortfall_99 = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)
    sortino_ratio = Column(Float, nullable=True)
    risk_score = Column(Float, nullable=True)
    risk_level = Column(String(10), nullable=True)
    created_at = Column(DateTime, server_default=func.now())

class SimulationResult(Base):
    __tablename__ = 'simulation_results'

    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(String(50), index=True, nullable=False)
    simulation_date = Column(DateTime, nullable=False)
    num_simulations = Column(Integer, nullable=False)
    simulation_days = Column(Integer, nullable=False)
    confidence_intervals = Column(JSON, nullable=False)
    var_estimates = Column(JSON, nullable=True)
    expected_returns = Column(JSON, nullable=True)
    simulation_paths = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())

class PortfolioHolding(Base):
    __tablename__ = 'portfolio_holdings'

    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(String(50), index=True, nullable=False)
    symbol = Column(String(10), nullable=False)
    quantity = Column(Float, nullable=True)
    weight = Column(Float, nullable=False)
    price = Column(Float, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
