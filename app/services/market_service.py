import yfinance as yf
import numpy as np
from typing import List, Optional, Dict
import requests
import structlog
from alpha_vantage.alphavantage import AlphaVantage
from app.config import settings
from app.models.market_data import MarketDataResponse, PriceHistory

logger = structlog.get_logger()

class MarketDataService:
    def __init__(self):
        self.alpha_vantage = AlphaVantage(key=settings.ALPHA_VANTAGE_API_KEY, output_format='pandas') if settings.ALPHA_VANTAGE_API_KEY else None

    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info.get('currentPrice') or info.get('previousClose')
        except Exception as e:
            logger.error(f"Failed to get current price for {symbol}: {e}")
            return None

    async def get_historical_data(self, symbol: str, period: str = "1y") -> Optional[PriceHistory]:
        """Get historical price data for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)

            if hist.empty:
                logger.warning(f"No historical data found for {symbol}")
                return None

            prices = []
            for date, row in hist.iterrows():
                price_data = MarketDataResponse(
                    symbol=symbol,
                    date=date,
                    open=float(row['Open']),
                    high=float(row['High']),
                    low=float(row['Low']),
                    close=float(row['Close']),
                    volume=int(row['Volume']),
                    adjusted_close=float(row['Close'])  # yfinance already provides adjusted close
                )
                prices.append(price_data)

            return PriceHistory(
                symbol=symbol,
                prices=prices,
                start_date=hist.index[0],
                end_date=hist.index[-1]
            )
        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            return None

    async def get_multiple_symbols_data(self, symbols: List[str], period: str = "1y") -> Dict[str, PriceHistory]:
        """Get historical data for multiple symbols"""
        results = {}
        for symbol in symbols:
            data = await self.get_historical_data(symbol, period)
            if data:
                results[symbol] = data
        return results

    async def get_risk_free_rate(self) -> float:
        """Get current risk-free rate (10-Year Treasury)"""
        try:
            ticker = yf.Ticker("^TNX")
            info = ticker.info
            rate = info.get('previousClose', settings.DEFAULT_RISK_FREE_RATE * 100)
            return rate / 100  # Convert percentage to decimal
        except Exception as e:
            logger.error(f"Failed to get risk-free rate: {e}")
            return settings.DEFAULT_RISK_FREE_RATE

    async def get_market_benchmark(self, benchmark: str = "SPY", period: str = "1y") -> Optional[PriceHistory]:
        """Get benchmark data (default SPY)"""
        return await self.get_historical_data(benchmark, period)

    def calculate_returns(self, prices: List[float]) -> np.ndarray:
        """Calculate simple returns from price series"""
        prices_array = np.array(prices)
        returns = np.diff(prices_array) / prices_array[:-1]
        return returns

    def calculate_log_returns(self, prices: List[float]) -> np.ndarray:
        """Calculate logarithmic returns from price series"""
        prices_array = np.array(prices)
        log_returns = np.diff(np.log(prices_array))
        return log_returns

    async def get_vix_data(self) -> Optional[float]:
        """Get current VIX (volatility index) value"""
        try:
            ticker = yf.Ticker(settings.VIX_SYMBOL)
            info = ticker.info
            return info.get('previousClose')
        except Exception as e:
            logger.error(f"Failed to get VIX data: {e}")
            return None

# Global instance
market_data_service = MarketDataService()
