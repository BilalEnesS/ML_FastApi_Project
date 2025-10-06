from __future__ import annotations
from fastapi import APIRouter, HTTPException
from starlette import status
from app.utils.config_store import ConfigStore
from app.utils.logger import get_logger
from app.services.training import TrainingService


router = APIRouter(prefix="/train", tags=["Train"])
logger = get_logger()
store = ConfigStore()

# Model training endpoint
@router.post("")
def start_training():
    
    cfg = store.load()
    algorithm = cfg.get("algorithm")
    if not algorithm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No algorithm configured. Call /config first.",
        )

    logger.info("Training job triggered")
    metrics = TrainingService().train()
    return {"message": f"Training completed with algorithm: {algorithm}", "metrics": metrics}


