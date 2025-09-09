from __future__ import annotations
from typing import Dict, Any
import pandas as pd
from app.models.predict import PredictPayload
from app.services.training import TrainingService


# Our goal here is to load the trained model(pipeline) from the disk and predict the target_type for the single input and return it as a dictionary to the API layer
class PredictionService:
    def __init__(self) -> None:
        self.pipeline = TrainingService.load_model()

    def predict_one(self, payload: PredictPayload) -> Dict[str, Any]:
        # Convert Pydantic payload to DataFrame and predict using pipeline.predict
        df = pd.DataFrame([payload.model_dump()])
        pred = self.pipeline.predict(df)
        # Return predicted target_type among {seller, customer, account}
        return {"target_type": str(pred[0])}


