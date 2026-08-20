import pandas as pd

from src.data.loader import load_dataset
from src.data.preprocessing import prepare_features, scale_features
from src.data.splitter import split_data
from src.evaluation.metrics import (
    calculate_classification_metrics,
    calculate_metrics,
)
from src.models.model_factory import ModelFactory
from src.services.trainer import Trainer


class Pipeline:
    """Coordinate the machine learning workflow."""

    def __init__(
        self,
        dataset_path: str,
        target: str,
        features: list[str],
        model_name: str,
        model_params: dict | None = None,
    ):
        self.dataset_path = dataset_path
        self.target = target
        self.features = features
        self.model_name = model_name
        self.model_params = model_params or {}

    def run(self):
        """Run the complete machine learning workflow."""

        # 1. Load dataset
        df = load_dataset(self.dataset_path)

        # 2. Prepare features and target
        X, y = prepare_features(
            df,
            target=self.target,
            features=self.features,
        )

        # 3. Split into training and testing sets
        X_train, X_test, y_train, y_test = split_data(X, y)

        # 4. Scale features
        X_train_scaled, X_test_scaled, scaler = scale_features(
            X_train,
            X_test,
        )

        # 5. Create model
        model = ModelFactory.create(
            self.model_name,
            **self.model_params,
        )

        # 6. Train model
        trainer = Trainer(model)
        trained_model = trainer.train(
            X_train_scaled,
            y_train,
        )

        # 7. Generate predictions
        predictions = trained_model.predict(X_test_scaled)

        # 8. Evaluate model
        if self.model_name == "logistic_regression":
            metrics = calculate_classification_metrics(
                y_test,
                predictions,
            )
        else:
            metrics = calculate_metrics(
                y_test,
                predictions,
            )

        return {
            "model": trained_model,
            "predictions": predictions,
            "metrics": metrics,
            "scaler": scaler,
        }