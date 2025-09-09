import numpy as np
import pandas as pd
from typing import Dict, List
from datetime import datetime
import structlog

from app.services.market_service import market_data_service
from app.utils.calculations import monte_carlo_portfolio_simulation
from app.models.monte_carlo import SimulationResponse

logger = structlog.get_logger()

class SimulationService:
    def __init__(self):
        self.trading_days_per_year = 252

    async def run_monte_carlo_simulation(self,
                                       portfolio_id: str,
                                       holdings: List[Dict],
                                       simulation_days: int = 252,
                                       num_simulations: int = 10000,
                                       confidence_levels: List[float] = [0.95, 0.99],
                                       initial_value: float = 10000) -> SimulationResponse:
        """
        Run Monte Carlo simulation for portfolio risk forecasting
        """
        try:
            # Extract symbols and weights
            symbols = [holding['symbol'] for holding in holdings]
            weights = np.array([holding['weight'] for holding in holdings])

            # Get historical data
            market_data = await market_data_service.get_multiple_symbols_data(symbols, "2y")

            if not market_data or len(market_data) != len(symbols):
                raise ValueError("Unable to fetch complete market data for simulation")

            # Calculate historical returns
            returns_matrix = pd.DataFrame()
            for symbol in symbols:
                prices = [p.close for p in market_data[symbol].prices]
                returns = market_data_service.calculate_log_returns(prices)
                returns_matrix[symbol] = returns

            # Calculate expected returns and covariance matrix
            expected_returns = returns_matrix.mean().values * self.trading_days_per_year
            cov_matrix = returns_matrix.cov().values * self.trading_days_per_year

            # Run Monte Carlo simulation
            portfolio_paths = monte_carlo_portfolio_simulation(
                expected_returns=expected_returns,
                cov_matrix=cov_matrix,
                weights=weights,
                initial_value=initial_value,
                days=simulation_days,
                num_simulations=num_simulations
            )

            # Calculate final portfolio values
            final_values = portfolio_paths[:, -1]

            # Calculate returns
            final_returns = (final_values / initial_value) - 1

            # Calculate confidence intervals
            confidence_intervals = {}
            for conf_level in confidence_levels:
                lower_percentile = (1 - conf_level) / 2 * 100
                upper_percentile = (1 + conf_level) / 2 * 100

                confidence_intervals[f"{conf_level:.0%}"] = {
                    "lower": float(np.percentile(final_returns, lower_percentile)),
                    "upper": float(np.percentile(final_returns, upper_percentile))
                }

            # Calculate VaR estimates
            var_estimates = {}
            for conf_level in confidence_levels:
                var_estimates[f"{conf_level:.0%}"] = float(
                    np.percentile(final_returns, (1 - conf_level) * 100)
                )

            # Calculate expected returns statistics
            expected_returns_stats = {
                "mean": float(np.mean(final_returns)),
                "median": float(np.median(final_returns)),
                "std": float(np.std(final_returns))
            }

            # Organize final values by percentiles
            final_values_dict = {
                "5th_percentile": float(np.percentile(final_values, 5)),
                "25th_percentile": float(np.percentile(final_values, 25)),
                "50th_percentile": float(np.percentile(final_values, 50)),
                "75th_percentile": float(np.percentile(final_values, 75)),
                "95th_percentile": float(np.percentile(final_values, 95))
            }

            return SimulationResponse(
                portfolio_id=portfolio_id,
                simulation_date=datetime.now(),
                num_simulations=num_simulations,
                simulation_days=simulation_days,
                confidence_intervals=confidence_intervals,
                var_estimates=var_estimates,
                expected_returns=expected_returns_stats,
                final_values=final_values_dict
            )

        except Exception as e:
            logger.error(f"Monte Carlo simulation failed: {e}")
            raise e

    def calculate_scenario_analysis(self,
                                  returns_matrix: pd.DataFrame,
                                  weights: np.array,
                                  scenarios: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate portfolio performance under different market scenarios
        """
        results = {}

        for scenario_name, market_shock in scenarios.items():
            # Apply shock to all assets (simplified approach)
            shocked_returns = returns_matrix.mean() + market_shock
            portfolio_return = np.dot(weights, shocked_returns)
            results[scenario_name] = float(portfolio_return)

        return results

    def stress_test_portfolio(self,
                            returns_matrix: pd.DataFrame,
                            weights: np.array) -> Dict[str, float]:
        """
        Perform stress testing based on historical crisis scenarios
        """
        stress_scenarios = {
            "2008_financial_crisis": -0.37,  # S&P 500 declined ~37% in 2008
            "covid_crash_2020": -0.34,       # Market declined ~34% in March 2020
            "dot_com_bubble_2000": -0.49,    # NASDAQ declined ~49% in 2000-2002
            "black_monday_1987": -0.22,      # Single day decline of 22%
            "mild_recession": -0.15,         # Moderate economic downturn
            "severe_recession": -0.30        # Severe economic downturn
        }

        return self.calculate_scenario_analysis(returns_matrix, weights, stress_scenarios)

# Global instance
simulation_service = SimulationService()
