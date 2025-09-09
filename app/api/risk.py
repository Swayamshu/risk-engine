from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
import structlog
import numpy as np
import pandas as pd

from app.config import settings
from app.database.connection import get_db
from app.main import app
from app.models.portfolio import PortfolioRequest
from app.models.risk_analysis import PortfolioRiskAnalysis, RiskMetrics
from app.services.risk_service import risk_calculator
from app.services.market_service import market_data_service
from app.database.models import RiskAnalysis as DBRiskAnalysis

logger = structlog.get_logger()
router = APIRouter()

app.include_router(router, prefix=f"{settings.API_V1_STR}/risk", tags=["risk"])

@router.post("/portfolio", response_model=PortfolioRiskAnalysis)
async def analyze_portfolio_risk(
    portfolio_request: PortfolioRequest,
    db: Session = Depends(get_db)
):
    """Analyze comprehensive risk metrics for a portfolio"""
    try:
        # Validate weights sum to 1
        total_weight = sum(holding.weight for holding in portfolio_request.holdings)
        if abs(total_weight - 1.0) > 0.01:
            raise HTTPException(
                status_code=400,
                detail=f"Portfolio weights must sum to 1.0, got {total_weight}"
            )

        # Get historical data for all holdings
        symbols = [holding.symbol for holding in portfolio_request.holdings]
        market_data = await market_data_service.get_multiple_symbols_data(
            symbols, portfolio_request.analysis_period
        )

        if not market_data:
            raise HTTPException(
                status_code=404,
                detail="Unable to fetch market data for portfolio holdings"
            )

        # Get benchmark data
        benchmark_data = await market_data_service.get_market_benchmark(
            portfolio_request.benchmark, portfolio_request.analysis_period
        )

        if not benchmark_data:
            raise HTTPException(
                status_code=404,
                detail=f"Unable to fetch benchmark data for {portfolio_request.benchmark}"
            )

        # Calculate individual asset risk metrics
        individual_metrics = []
        returns_matrix = pd.DataFrame()
        weights = np.array([holding.weight for holding in portfolio_request.holdings])

        for holding in portfolio_request.holdings:
            if holding.symbol not in market_data:
                continue

            asset_metrics = await risk_calculator.calculate_asset_risk_metrics(
                holding.symbol, portfolio_request.analysis_period
            )

            if asset_metrics:
                individual_metrics.append(asset_metrics)

                # Build returns matrix for portfolio calculations
                prices = [p.close for p in market_data[holding.symbol].prices]
                returns = market_data_service.calculate_returns(prices)
                returns_matrix[holding.symbol] = returns

        if returns_matrix.empty:
            raise HTTPException(
                status_code=500,
                detail="Unable to calculate returns for any portfolio holdings"
            )

        # Calculate portfolio-level metrics
        portfolio_returns = np.dot(returns_matrix.values, weights)
        benchmark_prices = [p.close for p in benchmark_data.prices]
        benchmark_returns = market_data_service.calculate_returns(benchmark_prices)

        # Align lengths
        min_length = min(len(portfolio_returns), len(benchmark_returns))
        portfolio_returns = portfolio_returns[-min_length:]
        benchmark_returns = benchmark_returns[-min_length:]

        # Get risk-free rate
        risk_free_rate = await market_data_service.get_risk_free_rate()

        # Calculate portfolio metrics
        portfolio_volatility = risk_calculator.calculate_volatility(portfolio_returns)
        portfolio_beta = risk_calculator.calculate_beta(portfolio_returns, benchmark_returns)
        portfolio_sharpe = risk_calculator.calculate_sharpe_ratio(portfolio_returns, risk_free_rate)
        portfolio_sortino = risk_calculator.calculate_sortino_ratio(portfolio_returns, risk_free_rate)

        # Calculate portfolio prices for drawdown
        portfolio_prices = []
        initial_value = 10000  # Starting with $10,000
        portfolio_prices.append(initial_value)

        for ret in portfolio_returns:
            portfolio_prices.append(portfolio_prices[-1] * (1 + ret))

        max_drawdown, _ = risk_calculator.calculate_max_drawdown(portfolio_prices)

        var_95 = risk_calculator.calculate_var(portfolio_returns, 0.95)
        var_99 = risk_calculator.calculate_var(portfolio_returns, 0.99)
        es_95 = risk_calculator.calculate_expected_shortfall(portfolio_returns, 0.95)
        es_99 = risk_calculator.calculate_expected_shortfall(portfolio_returns, 0.99)

        # Calculate risk score and level
        risk_score = risk_calculator.calculate_risk_score(
            portfolio_volatility, max_drawdown, portfolio_sharpe, var_95
        )
        risk_level = risk_calculator.determine_risk_level(risk_score)

        # Calculate correlation matrix
        correlation_matrix = None
        if len(returns_matrix.columns) > 1:
            corr_matrix = risk_calculator.calculate_correlation_matrix(returns_matrix)
            correlation_matrix = corr_matrix.to_dict()

        # Create response
        analysis = PortfolioRiskAnalysis(
            portfolio_id=portfolio_request.portfolio_id,
            analysis_date=pd.Timestamp.now(),
            volatility=portfolio_volatility,
            beta=portfolio_beta,
            sharpe_ratio=portfolio_sharpe,
            sortino_ratio=portfolio_sortino,
            var_95=var_95,
            var_99=var_99,
            expected_shortfall_95=es_95,
            expected_shortfall_99=es_99,
            max_drawdown=max_drawdown,
            risk_score=risk_score,
            risk_level=risk_level,
            individual_assets=individual_metrics,
            correlation_matrix=correlation_matrix
        )

        # Save to database
        db_analysis = DBRiskAnalysis(
            portfolio_id=portfolio_request.portfolio_id,
            analysis_date=analysis.analysis_date,
            volatility=portfolio_volatility,
            beta=portfolio_beta,
            sharpe_ratio=portfolio_sharpe,
            var_95=var_95,
            var_99=var_99,
            expected_shortfall_95=es_95,
            expected_shortfall_99=es_99,
            max_drawdown=max_drawdown,
            sortino_ratio=portfolio_sortino,
            risk_score=risk_score,
            risk_level=risk_level.value
        )

        db.add(db_analysis)
        db.commit()

        logger.info(f"Completed risk analysis for portfolio {portfolio_request.portfolio_id}")

        return analysis

    except Exception as e:
        logger.error(f"Portfolio risk analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/asset/{symbol}", response_model=RiskMetrics)
async def get_asset_risk(symbol: str, period: str = "1y"):
    """Get risk metrics for an individual asset"""
    try:
        metrics = await risk_calculator.calculate_asset_risk_metrics(symbol, period)
        if not metrics:
            raise HTTPException(
                status_code=404,
                detail=f"Unable to calculate risk metrics for {symbol}"
            )
        return metrics
    except Exception as e:
        logger.error(f"Asset risk calculation failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/portfolio/{portfolio_id}/history")
async def get_portfolio_risk_history(portfolio_id: str, db: Session = Depends(get_db)):
    """Get historical risk analysis for a portfolio"""
    try:
        analyses = db.query(DBRiskAnalysis).filter(
            DBRiskAnalysis.portfolio_id == portfolio_id
        ).order_by(DBRiskAnalysis.analysis_date.desc()).limit(50).all()

        return [
            {
                "analysis_date": analysis.analysis_date,
                "volatility": analysis.volatility,
                "beta": analysis.beta,
                "sharpe_ratio": analysis.sharpe_ratio,
                "var_95": analysis.var_95,
                "risk_score": analysis.risk_score,
                "risk_level": analysis.risk_level
            }
            for analysis in analyses
        ]
    except Exception as e:
        logger.error(f"Failed to get risk history for portfolio {portfolio_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
