import structlog

from app.config import settings
from app.main import app

logger = structlog.get_logger(__name__)

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Risk Engine API",
        "version": settings.VERSION,
        "description": settings.DESCRIPTION,
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "database": "enabled" if settings.ENABLE_DATABASE else "disabled"
    }
