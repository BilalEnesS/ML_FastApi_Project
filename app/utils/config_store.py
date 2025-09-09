from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict
from app.utils.logger import get_logger

# This file saves and loads the algorithm and hyperparameter settings selected by the user to a JSON file.
# The settings received from the /config endpoint are saved to app/config.json, and read during training.

logger = get_logger()


# This class saves the machine learning model configuration (algorithm, hyperparameters) to a JSON file.
class ConfigStore:
    def __init__(self, path: str = "app/config.json") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # This method saves the algorithm and hyperparameter settings selected by the user to a JSON file.
    def save(self, config: Dict[str, Any]) -> None:
        try:
            with self.path.open("w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            logger.info("Config saved to {path}", path=str(self.path))
        except Exception as exc:
            logger.exception("Failed to save config: {error}", error=str(exc))
            raise

    # This method loads the saved configuration from the JSON file, returns an empty dict if the file does not exist.
    def load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            with self.path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.exception("Failed to load config: {error}", error=str(exc))
            raise


