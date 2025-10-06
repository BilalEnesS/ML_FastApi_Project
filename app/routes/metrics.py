from __future__ import annotations

from fastapi import APIRouter

from app.utils.logger import get_logger
from app.services.metrics import MetricsService


# This router is used to read the metrics (accuracy, f1, etc.) from the disk and return them via the API after training.
router = APIRouter(prefix="/metrics", tags=["Metrics"])
logger = get_logger()

# When a request is received, log the request and load the metrics from MetricsService and return them
@router.get("")
def get_metrics():
    logger.info("Metrics requested")
    return MetricsService().load()


