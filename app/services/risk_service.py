import numpy as np
import structlog

logger = structlog.get_logger()

class RiskCalculator:
    def __init__(self):
        self.trading_days_per_year = 252

    """Calculate volatility (standard deviation of returns)"""
    def calculate_volatility(self, returns: np.ndarray, annualize: bool = True) -> float:
        vol = np.std(returns, ddof=1)
        if annualize:
            vol *= np.sqrt(self.trading_days_per_year)
        return float(vol)

    """Calculate Beta coefficient of an asset relative to the market"""
    def calculate_beta(self, asset_returns: np.ndarray, market_returns: np.ndarray) -> float:
        covariance = np.cov(asset_returns, market_returns)[0][1]
        market_variance = np.var(market_returns, ddof=1)
        if market_variance == 0:
            return 0.0
        beta = covariance / market_variance
        return float(beta)

    def calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        excess_returns = returns - (risk_free_rate / self.trading_days_per_year)
        if np.std(excess_returns) == 0:
            return 0.0

        sharpe = float(np.mean(excess_returns)) / float(np.std(excess_returns, ddof=1))
        sharpe *= np.sqrt(self.trading_days_per_year) # annualize
        return float(sharpe)

    def calculate_sortino_ration(self, returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        excess_returns = returns - (risk_free_rate / self.trading_days_per_year)
        downside_returns = excess_returns[excess_returns < 0] # boolean indexing -> array of negative returns

        if len(downside_returns) == 0:
            return float('inf')

        downside_deviation = np.std(downside_returns, ddof=1)
        if downside_deviation == 0:
            return 0

        sortino = float(np.mean(excess_returns)) / float(downside_deviation)
        sortino *= np.sqrt(self.trading_days_per_year) # annualize
        return float(sortino)



