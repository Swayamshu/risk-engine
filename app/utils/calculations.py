import numpy as np
from typing import List


def calculate_cagr(start_value: float, end_value: float, years: float) -> float:
    """ Calculate Compound Annual Growth Rate (CAGR) """
    if start_value <= 0 or years <= 0:
        return 0
    return (end_value / start_value) ** (1 / years) - 1


def calculate_xirr(cash_flow: List[float], dates: List, guess: float = 0.1) -> float:
    """
    Calculate XIRR (Extended Internal Rate of Return) for irregular cash flows
    Uses Newton-Raphson method for optimization
    """
    try:
        def xnpv(rate, cash_flows, dates):
            """Net Present Value calculation"""
            return sum([cf / (1 + rate) ** ((date - dates[0]).days / 365.0)
                        for cf, date in zip(cash_flows, dates)])

        def xnpv_derivative(rate, cash_flows, dates):
            """Derivative of NPV for Newton-Raphson"""
            return sum([-cf * ((date - dates[0]).days / 365.0) /
                        (1 + rate) ** (((date - dates[0]).days / 365.0) + 1)
                        for cf, date in zip(cash_flows, dates)])

        # Newton-Raphson iteration
        rate = guess
        for _ in range(100):  # Max iterations
            npv = xnpv(rate, cash_flow, dates)
            npv_derivative = xnpv_derivative(rate, cash_flow, dates)

            if abs(npv_derivative) < 1e-10:
                break

            new_rate = rate - npv / npv_derivative

            if abs(new_rate - rate) < 1e-10:
                break

            rate = new_rate

        return rate
    except:
        return 0.0


def calculate_rolling_volatility(returns: np.ndarray, window: int = 30) -> np.ndarray:
    """Calculate rolling volatility"""
    rolling_vol = []
    for i in range(len(returns)):
        if i < window - 1:
            rolling_vol.append(np.nan)
        else:
            window_returns = returns[i - window + 1:i + 1]
            vol = np.std(window_returns, ddof=1) * np.sqrt(252)
            rolling_vol.append(vol)
    return np.array(rolling_vol)


def calculate_information_ratio(portfolio_returns: np.ndarray,
                                benchmark_returns: np.ndarray) -> float:
    """Calculate Information Ratio (excess return / tracking error)"""
    excess_returns = portfolio_returns - benchmark_returns
    tracking_error = np.std(excess_returns, ddof=1)

    if tracking_error == 0:
        return 0

    return np.mean(excess_returns) / tracking_error * np.sqrt(252)


def calculate_calmar_ratio(returns: np.ndarray, prices: List[float]) -> float:
    """Calculate Calmar Ratio (CAGR / Max Drawdown)"""
    from app.services.risk_service import risk_calculator

    # Calculate CAGR
    years = len(returns) / 252
    total_return = (prices[-1] / prices[0] - 1) if len(prices) > 1 else 0
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    # Calculate max drawdown
    max_drawdown, _ = risk_calculator.calculate_max_drawdown(prices)

    if max_drawdown == 0:
        return float('inf')

    return cagr / abs(max_drawdown)


def monte_carlo_portfolio_simulation(expected_returns: np.ndarray,
                                     cov_matrix: np.ndarray,
                                     weights: np.ndarray,
                                     initial_value: float,
                                     days: int,
                                     num_simulations: int) -> np.ndarray:
    """
    Perform Monte Carlo simulation for portfolio values
    Returns array of shape (num_simulations, days+1)
    """
    # Convert annual parameters to daily
    daily_returns = expected_returns / 252
    daily_cov = cov_matrix / 252

    # Cholesky decomposition for correlated random numbers
    L = np.linalg.cholesky(daily_cov)

    # Generate random numbers
    random_numbers = np.random.normal(0, 1, (num_simulations, len(weights), days))

    # Initialize portfolio values
    portfolio_values = np.zeros((num_simulations, days + 1))
    portfolio_values[:, 0] = initial_value

    for sim in range(num_simulations):
        for day in range(days):
            # Generate correlated returns
            uncorrelated_returns = random_numbers[sim, :, day]
            correlated_returns = daily_returns + L @ uncorrelated_returns

            # Calculate portfolio return
            portfolio_return = np.dot(weights, correlated_returns)

            # Update portfolio value
            portfolio_values[sim, day + 1] = portfolio_values[sim, day] * (1 + portfolio_return)

    return portfolio_values
