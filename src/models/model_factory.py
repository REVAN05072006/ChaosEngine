from src.models.base_model import BaseModel
from src.models.classification.logistic_regression import LogisticRegression
from src.models.regression.linear_regression import LinearRegression


class ModelFactory:
    """Create machine learning model instances."""

    @staticmethod
    def create(
        model_name: str,
        **model_params,
    ) -> BaseModel:
        """Create a model based on its name and parameters."""

        if model_name == "linear_regression":
            return LinearRegression(**model_params)

        elif model_name == "logistic_regression":
            return LogisticRegression(**model_params)

        raise ValueError(f"Unknown model: {model_name}")