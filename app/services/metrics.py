from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any


# Our goal here is to load the metrics(JSON) saved on the disk and return them as a dictionary to the API layer
class MetricsService:
    def __init__(self, metrics_path: str = "app/models/metrics.json") -> None:
        self.path = Path(metrics_path)

    def load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {"message": "Metrics will be available after training"}
        with self.path.open("r", encoding="utf-8") as f:
            return json.load(f)


