import structlog
from fastapi import APIRouter

from app.config import settings
from app.main import app

logger = structlog.get_logger(__name__)
router = APIRouter

app.include_router(router, prefix=f"{settings.API_V1_STR}/risk", tags=["risk"])
