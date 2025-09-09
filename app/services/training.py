from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Tuple
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from app.services.preparation import FEATURE_COLUMNS, TARGET_COLUMNS, PreparationService
from app.utils.config_store import ConfigStore
from app.utils.logger import get_logger

# ML model training and evaluation service


logger = get_logger()


# Training service for ML models
class TrainingService:
    def __init__(self,
                 model_dir: str = "app/models",
                 metrics_path: str = "app/models/metrics.json") -> None:
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = Path(metrics_path)
        self.config = ConfigStore().load()

    # This method creates a machine learning pipeline based on the algorithm and hyperparameters chosen by the user.
    # It filters the valid parameters for each algorithm and merges them with safe default values.
    def _build_pipeline(self, algorithm: str, hyperparameters: Dict[str, Any], X_train: pd.DataFrame) -> Pipeline:
        prep = PreparationService()
        preprocessor = prep.build_preprocessor(X_train)

        # Algorithm selection with filtered hyperparameters
        clf: Any
        if algorithm.lower() in {"regression", "logreg", "logistic", "logisticregression"}:
            # Filter valid LogisticRegression parameters
            valid_params = {k: v for k, v in hyperparameters.items() 
                          if k in ["C", "penalty", "solver", "max_iter", "random_state"]}
            clf = LogisticRegression(max_iter=2000, class_weight='balanced', n_jobs=None, **valid_params)
        elif algorithm.lower() in {"linearsvc", "svm"}:
            # Filter valid LinearSVC parameters
            valid_params = {k: v for k, v in hyperparameters.items() 
                          if k in ["C", "penalty", "max_iter", "random_state"]}
            clf = LinearSVC(class_weight='balanced', max_iter=5000, **valid_params)
        elif algorithm.lower() in {"randomforest", "rf"}:
            # Filter valid RandomForest parameters
            valid_params = {k: v for k, v in hyperparameters.items() 
                          if k in ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "random_state"]}
            clf = RandomForestClassifier(
                n_estimators=400, max_depth=None, min_samples_split=2,
                random_state=42, n_jobs=-1, **valid_params
            )
        elif algorithm.lower() in {"ann", "neural", "mlp"}:
            # Filter valid MLP parameters and merge with safe defaults
            valid_keys = {
                "hidden_layer_sizes", "learning_rate", "max_iter", "random_state",
                "activation", "solver", "alpha", "batch_size"
            }
            valid_params = {k: v for k, v in hyperparameters.items() if k in valid_keys}

            default_params = {
                "hidden_layer_sizes": (128, 64),
                "activation": "relu",
                "solver": "adam",
                "alpha": 1e-4,
                "batch_size": "auto",
                "learning_rate": "adaptive",
                "max_iter": 400,
                "random_state": 42,
            }
            # User-specified params override defaults; avoids passing the same kw twice
            default_params.update(valid_params)
            clf = MLPClassifier(**default_params)
        else:
            clf = LogisticRegression(max_iter=2000, class_weight='balanced')

        logger.info("Selected algorithm: {algorithm}, classifier: {clf}, params: {params}", 
                   algorithm=algorithm, clf=type(clf).__name__, params=hyperparameters)

        pipe = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("clf", clf),
        ])
        return pipe

    # Train model with cross-validation and comprehensive metrics
    def train(self) -> Dict[str, Any]:
        prep = PreparationService()
        df = prep.load_raw()
        X_train, X_test, y_train, y_test = prep.prepare_and_split(df)

        algorithm = self.config.get("algorithm", "classification")
        hyperparameters = self.config.get("hyperparameters", {})

        pipeline = self._build_pipeline(algorithm, hyperparameters, X_train)
        pipeline.fit(X_train, y_train["target_type"])  # multiclass target

        y_pred = pipeline.predict(X_test)
        
        # Basic metrics with different averaging methods
        acc = float(accuracy_score(y_test["target_type"], y_pred))
        f1_macro = float(f1_score(y_test["target_type"], y_pred, average="macro"))
        f1_weighted = float(f1_score(y_test["target_type"], y_pred, average="weighted"))
        precision_macro = float(precision_score(y_test["target_type"], y_pred, average="macro", zero_division=0))
        precision_weighted = float(precision_score(y_test["target_type"], y_pred, average="weighted", zero_division=0))
        recall_macro = float(recall_score(y_test["target_type"], y_pred, average="macro", zero_division=0))
        recall_weighted = float(recall_score(y_test["target_type"], y_pred, average="weighted", zero_division=0))
        
        # Confusion matrix
        cm = confusion_matrix(y_test["target_type"], y_pred)
        cm_dict = {
            "matrix": cm.tolist(),
            "labels": sorted(y_test["target_type"].unique().tolist())
        }
        
        class_report = classification_report(y_test["target_type"], y_pred, output_dict=True, zero_division=0)
        
        # Cross-validation metrics
        cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_acc = cross_val_score(pipeline, X_train, y_train["target_type"], scoring='accuracy', cv=cv_splitter)
        cv_f1 = cross_val_score(pipeline, X_train, y_train["target_type"], scoring='f1_macro', cv=cv_splitter)
        
        metrics = {
            "accuracy": acc,
            "f1_score_macro": f1_macro,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "confusion_matrix": cm_dict,
            "cross_validation": {
                "cv_accuracy_mean": float(cv_acc.mean()),
                "cv_accuracy_std": float(cv_acc.std()),
                "cv_f1_macro_mean": float(cv_f1.mean()),
                "cv_f1_macro_std": float(cv_f1.std())
            }
        }

        joblib.dump(pipeline, self.model_dir / "model.pkl")
        with self.metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        logger.info("Model trained with advanced pipeline. Metrics: {metrics}", metrics=metrics)
        return metrics

    # This static method loads the trained model from the disk and allows the prediction service to use it.
    @staticmethod
    def load_model(model_dir: str = "app/models") -> Pipeline:
        model_path = Path(model_dir) / "model.pkl"
        if not model_path.exists():
            raise FileNotFoundError("Trained model not found. Run /train first.")
        return joblib.load(model_path)


