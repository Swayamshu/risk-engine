import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from scipy import stats
import structlog

from app.models import RiskLevel
from app.models.risk_analysis import RiskMetrics
from app.services.market_service import market_data_service


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

    def calculate_var(self, returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk using historical method"""
        return float(np.percentile(returns, (1 - confidence_level) * 100))

    def calculate_parametric_var(self, returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """Calculate parametric VaR assuming normal distribution"""
        mean = np.mean(returns)
        std = np.std(returns, ddof=1)
        z_score = stats.norm.ppf(1 - confidence_level)
        var = mean + z_score * std
        return float(var)

    def calculate_expected_shortfall(self, returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        var = self.calculate_var(returns, confidence_level)
        tail_losses = returns[returns <= var]
        if len(tail_losses) == 0:
            return var
        return float(np.mean(tail_losses))

    def calculate_max_drawdown(self, prices: List[float]) -> Tuple[float, int]:
        """Calculate maximum drawdown and recovery periods"""
        prices_array = np.array(prices)
        cumulative_returns = prices_array / prices_array[0]
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max

        max_dd = float(np.min(drawdown))

        # Calculate recovery time
        max_dd_idx = np.argmin(drawdown)
        recovery_idx = None
        for i in range(max_dd_idx + 1, len(cumulative_returns)):
            if cumulative_returns[i] >= running_max[max_dd_idx]:
                recovery_idx = i
                break

        recovery_days = recovery_idx - max_dd_idx if recovery_idx else len(prices) - max_dd_idx

        return max_dd, recovery_days

    def calculate_portfolio_volatility(self, weights: np.ndarray, cov_matrix: np.ndarray) -> float:
        """Calculate portfolio volatility using weights and covariance matrix"""
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        return float(np.sqrt(portfolio_variance * self.trading_days_per_year))

    def calculate_portfolio_beta(self, weights: np.ndarray, individual_betas: np.ndarray) -> float:
        """Calculate portfolio beta as weighted average"""
        return float(np.dot(weights, individual_betas))

    def calculate_diversification_ratio(self, weights: np.ndarray,
                                      individual_volatilities: np.ndarray,
                                      portfolio_volatility: float) -> float:
        """Calculate diversification ratio"""
        weighted_avg_vol = np.dot(weights, individual_volatilities)
        if portfolio_volatility == 0:
            return 1.0
        return float(weighted_avg_vol / portfolio_volatility)

    def calculate_concentration_risk(self, weights: np.ndarray) -> float:
        """Calculate Herfindahl-Hirschman Index for concentration"""
        hhi = np.sum(weights ** 2)
        return float(hhi)

    def calculate_correlation_matrix(self, returns_matrix: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix for assets"""
        return returns_matrix.corr()

    def calculate_risk_score(self, volatility: float, max_drawdown: float,
                           sharpe_ratio: float, var_95: float) -> float:
        """Calculate composite risk score (0-100)"""
        # Normalize components (higher risk = higher score)
        vol_score = min(volatility * 100, 50)  # Cap at 50
        dd_score = min(abs(max_drawdown) * 100, 30)  # Cap at 30
        sharpe_score = max(0, 20 - sharpe_ratio * 10)  # Lower Sharpe = higher risk
        var_score = min(abs(var_95) * 100, 20)  # Cap at 20

        total_score = vol_score + dd_score + sharpe_score + var_score
        return min(total_score, 100)

    def determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level based on risk score"""
        if risk_score <= 30:
            return RiskLevel.LOW
        elif risk_score <= 70:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.HIGH

    async def calculate_asset_risk_metrics(self, symbol: str, period: str = "1y") -> Optional[RiskMetrics]:
        """Calculate comprehensive risk metrics for a single asset"""
        try:
            # Get historical data
            price_history = await market_data_service.get_historical_data(symbol, period)
            if not price_history:
                return None

            # Get benchmark data for beta calculation
            benchmark_data = await market_data_service.get_market_benchmark("SPY", period)
            if not benchmark_data:
                return None

            # Get risk-free rate
            risk_free_rate = await market_data_service.get_risk_free_rate()

            # Extract prices and calculate returns
            prices = [p.close for p in price_history.prices]
            returns = market_data_service.calculate_returns(prices)

            benchmark_prices = [p.close for p in benchmark_data.prices]
            benchmark_returns = market_data_service.calculate_returns(benchmark_prices)

            # Align returns (in case of different lengths)
            min_length = min(len(returns), len(benchmark_returns))
            returns = returns[-min_length:]
            benchmark_returns = benchmark_returns[-min_length:]

            # Calculate metrics
            volatility = self.calculate_volatility(returns)
            beta = self.calculate_beta(returns, benchmark_returns)
            sharpe_ratio = self.calculate_sharpe_ratio(returns, risk_free_rate)
            sortino_ratio = self.calculate_sortino_ratio(returns, risk_free_rate)
            max_drawdown, _ = self.calculate_max_drawdown(prices)
            var_95 = self.calculate_var(returns, 0.95)
            var_99 = self.calculate_var(returns, 0.99)
            es_95 = self.calculate_expected_shortfall(returns, 0.95)
            es_99 = self.calculate_expected_shortfall(returns, 0.99)

            return RiskMetrics(
                symbol=symbol,
                volatility=volatility,
                beta=beta,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown,
                var_95=var_95,
                var_99=var_99,
                expected_shortfall_95=es_95,
                expected_shortfall_99=es_99
            )

        except Exception as e:
            logger.error(f"Failed to calculate risk metrics for {symbol}: {e}")
            return None

# Global instance
risk_calculator = RiskCalculator()
