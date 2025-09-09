import structlog
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # App settings
    APP_NAME: str = "Risk Engine"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # API Configuration
    API_V1_STR: str = "/api/risk-engine/v1"
    PROJECT_NAME: str = "Risk Engine Service"
    DESCRIPTION: str = "API for Portfolio risk analysis and financial calculations"
    VERSION: str = "0.1.0"

    # Database settings
    DATABASE_URL: str = "postgresql://user:password@localhost/dbname"
    ENABLE_DATABASE: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

settings = Settings()

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter,
        structlog.processors.JSONRenderer,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer,
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)
