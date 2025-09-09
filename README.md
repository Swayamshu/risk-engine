# Risk Engine Service

A comprehensive Python-based Risk Engine microservice for quantitative finance calculations, risk analysis, and market data integration.

## Features

### Phase 1 Implementation ✅
- **Market Data Integration**: Yahoo Finance and Alpha Vantage API support
- **Risk Calculations**: Volatility, Beta, Sharpe Ratio, VaR, Expected Shortfall, Max Drawdown
- **Portfolio Analytics**: Portfolio-level risk metrics and correlation analysis
- **SIP Analytics**: Systematic Investment Plan performance analysis
- **Monte Carlo Simulation**: Risk forecasting with confidence intervals
- **REST API**: FastAPI-based high-performance endpoints
- **Database Integration**: PostgreSQL for persistent storage
- **Caching**: Redis for market data caching˛

## Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.11+ (for local development)
- Alpha Vantage API key (optional, for enhanced data)

### Environment Setup
1. Copy environment template:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your configuration:
   - Set `ALPHA_VANTAGE_API_KEY` for enhanced market data
   - Adjust database and Redis URLs if needed

### Running with Docker (Recommended)
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f risk-engine

# Stop services
docker-compose down
```

### Running Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Start PostgreSQL and Redis (or use Docker)
docker-compose up -d db redis

# Run the application
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### Market Data
- `GET /api/v1/market/price/{symbol}` - Current price
- `GET /api/v1/market/history/{symbol}` - Historical data
- `GET /api/v1/market/risk-free-rate` - Risk-free rate
- `GET /api/v1/market/vix` - Volatility index

### Risk Analysis
- `POST /api/v1/risk/portfolio` - Portfolio risk analysis
- `GET /api/v1/risk/asset/{symbol}` - Individual asset risk metrics
- `GET /api/v1/risk/portfolio/{portfolio_id}/history` - Risk history

### Analytics
- `POST /api/v1/analytics/sip` - SIP analysis
- `POST /api/v1/analytics/simulation` - Monte Carlo simulation
- `POST /api/v1/analytics/stress-test` - Stress testing
- `GET /api/v1/analytics/performance/{symbol}` - Asset performance

## Example Usage

### Portfolio Risk Analysis
```python
import requests

portfolio_data = {
    "portfolio_id": "portfolio-123",
    "holdings": [
        {"symbol": "AAPL", "weight": 0.4},
        {"symbol": "GOOGL", "weight": 0.3},
        {"symbol": "MSFT", "weight": 0.3}
    ],
    "benchmark": "SPY",
    "analysis_period": "1Y"
}

response = requests.post(
    "http://localhost:8000/api/v1/risk/portfolio",
    json=portfolio_data
)
risk_metrics = response.json()
```

### SIP Analysis
```python
from datetime import datetime, timedelta

sip_data = {
    "amount": 5000,
    "frequency": "MONTHLY",
    "symbols": ["AAPL", "SPY"],
    "weights": [0.6, 0.4],
    "start_date": (datetime.now() - timedelta(days=365)).isoformat(),
    "end_date": datetime.now().isoformat()
}

response = requests.post(
    "http://localhost:8000/api/v1/analytics/sip",
    json=sip_data
)
```

## Testing

```bash
# Run tests with Docker
docker-compose --profile test run test

# Run tests locally
pytest tests/ -v
```

## API Documentation

Access interactive API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Architecture

```
├── app/
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration settings
│   ├── api/                 # API endpoints
│   │   ├── risk.py         # Risk analysis endpoints
│   │   ├── market.py       # Market data endpoints
│   │   └── analytics.py    # Portfolio analytics
│   ├── services/           # Business logic
│   │   ├── market_data.py  # Market data fetching
│   │   ├── risk_calculator.py # Risk calculations
│   │   └── simulation.py   # Monte Carlo simulations
│   ├── database/           # Database models
│   │   ├── models.py       # SQLAlchemy models
│   │   └── connection.py   # Database connection
│   └── utils/              # Utilities
│       ├── calculations.py # Financial calculations
│       └── cache.py        # Redis caching
```

## Risk Metrics Implemented

1. **Volatility (σ)** - Standard deviation of returns
2. **Beta (β)** - Market sensitivity coefficient
3. **Sharpe Ratio** - Risk-adjusted return measure
4. **Sortino Ratio** - Downside risk-adjusted returns
5. **Value at Risk (VaR)** - Potential loss at confidence levels
6. **Expected Shortfall** - Average loss beyond VaR
7. **Maximum Drawdown** - Largest peak-to-trough decline
8. **Correlation Matrix** - Asset correlation analysis

## Integration with Spring Boot

The service is designed to integrate seamlessly with your Kotlin Spring Boot backend:

- REST API endpoints match your `RiskMetricResult` entity structure
- PostgreSQL database shared between services
- JSON response formats compatible with Spring Boot DTOs
- Error handling and logging for microservice architecture

## Next Steps (Phase 2 & 3)

- Enhanced Monte Carlo simulation with custom scenarios
- Advanced portfolio optimization algorithms
- Real-time market data streaming
- Machine learning-based risk predictions
- Performance optimization and caching strategies
- Comprehensive integration testing with Spring Boot backend

## Performance

- FastAPI for high-performance async operations
- Redis caching for frequently accessed market data
- Optimized NumPy/Pandas operations for calculations
- Database connection pooling for concurrent requests
