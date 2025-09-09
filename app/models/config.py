from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, ConfigDict


# Supported ML algorithms
class AlgorithmEnum(str, Enum):
    regression = "regression"
    classification = "classification"
    linearsvc = "linearsvc"
    randomforest = "randomforest"
    ann = "ann"


# Our goal here is to validate and transfer the configuration data received from the /config endpoint.
# Algorithm: The model family selected by the user (limited by enum).
# Hyperparameters: The dictionary of optional hyperparameters for the relevant algorithm.   
class ConfigPayload(BaseModel):
    algorithm: AlgorithmEnum = Field(..., description="Selected algorithm type")
    hyperparameters: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional hyperparameters for the algorithm"
    )

    # We add extra information to the schema to make it visible in Swagger/Docs.
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "algorithm": "regression",
                    "hyperparameters": {"C": 1.0, "penalty": "l2"}
                },
                {
                    "algorithm": "classification",
                    "hyperparameters": {"C": 1.0, "penalty": "l2"}
                },
                {
                    "algorithm": "linearsvc",
                    "hyperparameters": {"C": 1.0, "max_iter": 5000}
                },
                {
                    "algorithm": "randomforest",
                    "hyperparameters": {"n_estimators": 400, "max_depth": None}
                },
                {
                    "algorithm": "ann",
                    "hyperparameters": {"hidden_layer_sizes": [128, 64], "learning_rate": "adaptive"}
                },
            ]
        }
    )


