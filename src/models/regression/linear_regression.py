import numpy as np
import pandas as pd

from src.models.base_model import BaseModel

class LinearRegression(BaseModel):
    """Linear Regression implemented using Gradient Descent."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
    ):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

        self.weights = None
        self.bias = 0.0
        self.cost_history = []

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> None:

        X_values = X.to_numpy()
        y_values = y.to_numpy()

        n_samples, n_features = X_values.shape

        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iterations):

            # 1. Make predictions
            predictions = X_values @ self.weights + self.bias

            # 2. Calculate errors
            errors = predictions - y_values

            # 3. Calculate cost (MSE)
            cost = np.mean(errors ** 2)
            self.cost_history.append(cost)

            # 4. Calculate gradients
            weight_gradients = (
                (2 / n_samples)
                * (X_values.T @ errors)
            )

            bias_gradient = (
                (2 / n_samples)
                * np.sum(errors)
            )

            # 5. Update parameters
            self.weights -= (
                self.learning_rate * weight_gradients
            )

            self.bias -= (
                self.learning_rate * bias_gradient
            )

    def predict(
        self,
        X: pd.DataFrame,
    ) -> pd.Series:

        X_values = X.to_numpy()

        predictions = X_values @ self.weights + self.bias

        return pd.Series(
            predictions,
            index=X.index,
            name="prediction",
        )