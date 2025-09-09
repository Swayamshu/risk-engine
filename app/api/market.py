from fastapi import APIRouter, HTTPException
import structlog

from app.config import settings
from app.main import app
from app.models.market_data import PriceHistory
from app.services.market_service import market_data_service

logger = structlog.get_logger()
router = APIRouter()

app.include_router(router, prefix=f"{settings.API_V1_STR}/market", tags=["market"])

@router.get("/price/{symbol}")
async def get_current_price(symbol: str):
    """Get current price for a symbol"""
    try:
        price = await market_data_service.get_current_price(symbol)
        if price is None:
            raise HTTPException(
                status_code=404,
                detail=f"Price not found for symbol {symbol}"
            )

        return {
            "symbol": symbol,
            "price": price,
            "timestamp": "real-time"
        }
    except Exception as e:
        logger.error(f"Failed to get price for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history/{symbol}", response_model=PriceHistory)
async def get_price_history(symbol: str, period: str = "1y"):
    """Get historical price data for a symbol"""
    try:
        history = await market_data_service.get_historical_data(symbol, period)
        if not history:
            raise HTTPException(
                status_code=404,
                detail=f"Historical data not found for symbol {symbol}"
            )

        return history
    except Exception as e:
        logger.error(f"Failed to get history for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/benchmark/{index}")
async def get_benchmark_data(index: str = "SPY", period: str = "1y"):
    """Get benchmark index data"""
    try:
        benchmark = await market_data_service.get_market_benchmark(index, period)
        if not benchmark:
            raise HTTPException(
                status_code=404,
                detail=f"Benchmark data not found for {index}"
            )

        return benchmark
    except Exception as e:
        logger.error(f"Failed to get benchmark data for {index}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/batch")
async def get_batch_prices(symbols: str, period: str = "1y"):
    """Get historical data for multiple symbols (comma-separated)"""
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]

        if len(symbol_list) > 50:  # Limit batch size
            raise HTTPException(
                status_code=400,
                detail="Maximum 50 symbols allowed per batch request"
            )

        data = await market_data_service.get_multiple_symbols_data(symbol_list, period)

        return {
            "symbols": symbol_list,
            "data": data,
            "found": len(data),
            "requested": len(symbol_list)
        }
    except Exception as e:
        logger.error(f"Failed to get batch data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/risk-free-rate")
async def get_risk_free_rate():
    """Get current risk-free rate (10-Year Treasury)"""
    try:
        rate = await market_data_service.get_risk_free_rate()
        return {
            "risk_free_rate": rate,
            "rate_percent": rate * 100,
            "source": "10-Year Treasury"
        }
    except Exception as e:
        logger.error(f"Failed to get risk-free rate: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/vix")
async def get_vix():
    """Get current VIX (volatility index) value"""
    try:
        vix = await market_data_service.get_vix_data()
        if vix is None:
            raise HTTPException(
                status_code=404,
                detail="VIX data not available"
            )

        return {
            "vix": vix,
            "market_sentiment": "High Fear" if vix > 30 else "Low Fear" if vix < 15 else "Moderate Fear"
        }
    except Exception as e:
        logger.error(f"Failed to get VIX data: {e}")
        raise HTTPException(status_code=500, detail=str(e))
