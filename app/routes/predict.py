from __future__ import annotations
from fastapi import APIRouter, HTTPException
from app.models.predict import PredictPayload
from app.utils.logger import get_logger
from app.services.prediction import PredictionService


# This router, API receives a single prediction request, validates it against the schema and uses the trained model to calculate the prediction and return it as JSON.
router = APIRouter(tags=["Predict"])
logger = get_logger()

# Our goal here is to validate the input data with PredictPayload and pass it to PredictionService to calculate the prediction for a single example and return it as JSON.
@router.post("/predict")
def predict(payload: PredictPayload):
    try:
        pred = PredictionService().predict_one(payload)
        return {"prediction": pred}
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Model not trained yet. Run /train.")


