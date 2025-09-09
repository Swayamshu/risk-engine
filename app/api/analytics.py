from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import structlog

from app.config import settings
from app.database.connection import get_db
from app.main import app
from app.models.sip import SIPRequest, SIPAnalysis
from app.models.monte_carlo import SimulationRequest, SimulationResponse
from app.services.market_service import market_data_service
from app.services.risk_service import risk_calculator
from app.services.simulation import simulation_service
from app.database.models import SimulationResult as DBSimulationResult

logger = structlog.get_logger()
router = APIRouter()

app.include_router(router, prefix=f"{settings.API_V1_STR}/analytics", tags=["analytics"])

@router.post("/sip", response_model=SIPAnalysis)
async def analyze_sip(sip_request: SIPRequest):
    """Analyze SIP (Systematic Investment Plan) performance"""
    try:
        # Validate weights
        if len(sip_request.symbols) != len(sip_request.weights):
            raise HTTPException(
                status_code=400,
                detail="Number of symbols must match number of weights"
            )

        if abs(sum(sip_request.weights) - 1.0) > 0.01:
            raise HTTPException(
                status_code=400,
                detail="Weights must sum to 1.0"
            )

        # Calculate investment frequency
        if sip_request.frequency == "MONTHLY":
            freq_days = 30
        elif sip_request.frequency == "WEEKLY":
            freq_days = 7
        else:  # DAILY
            freq_days = 1

        # Get historical data for all symbols
        period_years = (sip_request.end_date - sip_request.start_date).days / 365.25
        period_str = f"{max(1, int(period_years + 1))}y"

        market_data = await market_data_service.get_multiple_symbols_data(
            sip_request.symbols, period_str
        )

        if not market_data:
            raise HTTPException(
                status_code=404,
                detail="Unable to fetch market data for SIP analysis"
            )

        # Create date range for investments
        investment_dates = []
        current_date = sip_request.start_date
        while current_date <= sip_request.end_date:
            investment_dates.append(current_date)
            current_date += timedelta(days=freq_days)

        # Calculate SIP performance
        total_invested = 0
        portfolio_units = {symbol: 0 for symbol in sip_request.symbols}

        for inv_date in investment_dates:
            # Find closest trading day for each symbol
            investment_amount_per_symbol = sip_request.amount / len(sip_request.symbols)

            for i, symbol in enumerate(sip_request.symbols):
                weight = sip_request.weights[i]
                symbol_investment = sip_request.amount * weight

                # Find price on investment date
                symbol_data = market_data[symbol]
                closest_price = None
                min_date_diff = float('inf')

                for price_data in symbol_data.prices:
                    date_diff = abs((price_data.date.date() - inv_date).days)
                    if date_diff < min_date_diff:
                        min_date_diff = date_diff
                        closest_price = price_data.close

                if closest_price:
                    units_bought = symbol_investment / closest_price
                    portfolio_units[symbol] += units_bought
                    total_invested += symbol_investment

        # Calculate final portfolio value
        final_value = 0
        for symbol in sip_request.symbols:
            # Get final price (latest available)
            symbol_data = market_data[symbol]
            final_price = symbol_data.prices[-1].close
            final_value += portfolio_units[symbol] * final_price

        # Calculate returns and metrics
        total_returns = (final_value - total_invested) / total_invested if total_invested > 0 else 0

        # Calculate CAGR
        years = max(0.1, (sip_request.end_date - sip_request.start_date).days / 365.25)
        cagr = ((final_value / total_invested) ** (1 / years) - 1) if total_invested > 0 else 0

        # Calculate portfolio returns for volatility and Sharpe ratio
        # Build returns matrix
        returns_matrix = pd.DataFrame()
        for symbol in sip_request.symbols:
            prices = [p.close for p in market_data[symbol].prices]
            returns = market_data_service.calculate_returns(prices)
            returns_matrix[symbol] = returns

        # Calculate weighted portfolio returns
        weights = np.array(sip_request.weights)
        portfolio_returns = np.dot(returns_matrix.values, weights)

        volatility = risk_calculator.calculate_volatility(portfolio_returns)

        # Get risk-free rate for Sharpe ratio
        risk_free_rate = await market_data_service.get_risk_free_rate()
        sharpe_ratio = risk_calculator.calculate_sharpe_ratio(portfolio_returns, risk_free_rate)

        # Calculate cost averaging benefit (compare with lump sum)
        lump_sum_value = total_invested
        for symbol in sip_request.symbols:
            weight = sip_request.weights[sip_request.symbols.index(symbol)]
            symbol_data = market_data[symbol]
            start_price = symbol_data.prices[0].close
            end_price = symbol_data.prices[-1].close
            symbol_return = (end_price / start_price - 1)
            lump_sum_value += total_invested * weight * symbol_return

        cost_averaging_benefit = (final_value - lump_sum_value) / lump_sum_value if lump_sum_value > 0 else 0

        return SIPAnalysis(
            total_invested=total_invested,
            final_value=final_value,
            returns=total_returns,
            cagr=cagr,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            investment_periods=len(investment_dates),
            cost_averaging_benefit=cost_averaging_benefit
        )

    except Exception as e:
        logger.error(f"SIP analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/simulation", response_model=SimulationResponse)
async def monte_carlo_simulation(
    simulation_request: SimulationRequest,
    db: Session = Depends(get_db)
):
    """Perform Monte Carlo simulation for portfolio risk forecasting"""
    try:
        # Get portfolio holdings from database or request
        # For now, we'll use a sample portfolio structure
        # In production, you'd fetch actual holdings from the database
        sample_holdings = [
            {"symbol": "AAPL", "weight": 0.3},
            {"symbol": "GOOGL", "weight": 0.25},
            {"symbol": "MSFT", "weight": 0.25},
            {"symbol": "SPY", "weight": 0.2}
        ]

        # Run the actual Monte Carlo simulation
        simulation_result = await simulation_service.run_monte_carlo_simulation(
            portfolio_id=simulation_request.portfolio_id,
            holdings=sample_holdings,
            simulation_days=simulation_request.simulation_days,
            num_simulations=simulation_request.num_simulations,
            confidence_levels=simulation_request.confidence_levels
        )

        # Save simulation results to database
        db_simulation = DBSimulationResult(
            portfolio_id=simulation_request.portfolio_id,
            simulation_date=simulation_result.simulation_date,
            num_simulations=simulation_request.num_simulations,
            simulation_days=simulation_request.simulation_days,
            confidence_intervals=simulation_result.confidence_intervals,
            var_estimates=simulation_result.var_estimates,
            expected_returns=simulation_result.expected_returns
        )

        db.add(db_simulation)
        db.commit()

        return simulation_result

    except Exception as e:
        logger.error(f"Monte Carlo simulation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stress-test")
async def stress_test_portfolio(portfolio_request: dict):
    """Perform stress testing on portfolio"""
    try:
        symbols = [holding["symbol"] for holding in portfolio_request["holdings"]]
        weights = np.array([holding["weight"] for holding in portfolio_request["holdings"]])

        # Get historical data
        market_data = await market_data_service.get_multiple_symbols_data(symbols, "2y")

        # Build returns matrix
        returns_matrix = pd.DataFrame()
        for symbol in symbols:
            prices = [p.close for p in market_data[symbol].prices]
            returns = market_data_service.calculate_returns(prices)
            returns_matrix[symbol] = returns

        # Run stress tests
        stress_results = simulation_service.stress_test_portfolio(returns_matrix, weights)

        return {
            "portfolio_id": portfolio_request.get("portfolio_id"),
            "stress_test_date": datetime.now(),
            "scenarios": stress_results,
            "worst_case_scenario": min(stress_results.values()),
            "best_case_scenario": max(stress_results.values())
        }

    except Exception as e:
        logger.error(f"Stress testing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance/{symbol}")
async def get_asset_performance(symbol: str, period: str = "1y"):
    """Get comprehensive performance metrics for an asset"""
    try:
        # Get historical data
        price_history = await market_data_service.get_historical_data(symbol, period)
        if not price_history:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for symbol {symbol}"
            )

        prices = [p.close for p in price_history.prices]
        returns = market_data_service.calculate_returns(prices)

        # Calculate performance metrics
        total_return = (prices[-1] / prices[0] - 1) if len(prices) > 1 else 0

        # Calculate CAGR
        days = len(prices)
        years = days / 252  # Trading days per year
        cagr = ((prices[-1] / prices[0]) ** (1 / years) - 1) if years > 0 and len(prices) > 1 else 0

        volatility = risk_calculator.calculate_volatility(returns)
        max_drawdown, recovery_days = risk_calculator.calculate_max_drawdown(prices)

        # Get risk-free rate
        risk_free_rate = await market_data_service.get_risk_free_rate()
        sharpe_ratio = risk_calculator.calculate_sharpe_ratio(returns, risk_free_rate)

        return {
            "symbol": symbol,
            "period": period,
            "total_return": total_return,
            "cagr": cagr,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "recovery_days": recovery_days,
            "start_price": prices[0],
            "end_price": prices[-1],
            "data_points": len(prices)
        }

    except Exception as e:
        logger.error(f"Performance analysis failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
