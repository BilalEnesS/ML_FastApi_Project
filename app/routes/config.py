from __future__ import annotations
from fastapi import APIRouter, Body
from app.models.config import ConfigPayload
from app.utils.config_store import ConfigStore
from app.utils.logger import get_logger


router = APIRouter(prefix="/config", tags=["Config"])
logger = get_logger()
store = ConfigStore()


#Our goal here is to validate and store the algorithm and hyperparameter settings received from the /config endpoint.
@router.post("")
def set_config(
    payload: ConfigPayload = Body(...)
):
    data = payload.model_dump()
    store.save(data)
    logger.info("Config updated: {data}", data=data)
    return {"message": "Config saved", "config": data}


