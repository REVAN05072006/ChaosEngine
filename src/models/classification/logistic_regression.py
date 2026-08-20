import numpy as np
import pandas as pd

from src.models.base_model import BaseModel


class LogisticRegression(BaseModel):
    """Logistic Regression classifier trained with Gradient Descent."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        threshold: float = 0.5,
    ):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.threshold = threshold

        self.weights = None
        self.bias = 0.0
        self.cost_history = []

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Convert raw scores into probabilities."""

        return 1 / (1 + np.exp(-z))

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> None:
        """Train the model using Gradient Descent."""

        X_values = X.to_numpy()
        y_values = y.to_numpy()

        n_samples, n_features = X_values.shape

        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.cost_history = []

        for _ in range(self.n_iterations):

            z = X_values @ self.weights + self.bias

            probabilities = self._sigmoid(z)

            epsilon = 1e-15
            probabilities = np.clip(
                probabilities,
                epsilon,
                1 - epsilon,
            )

            cost = -np.mean(
                y_values * np.log(probabilities)
                + (1 - y_values) * np.log(1 - probabilities)
            )

            self.cost_history.append(cost)

            errors = probabilities - y_values

            weight_gradients = (
                1 / n_samples
            ) * (X_values.T @ errors)

            bias_gradient = (
                1 / n_samples
            ) * np.sum(errors)

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
        """Generate class predictions."""

        X_values = X.to_numpy()

        z = X_values @ self.weights + self.bias

        probabilities = self._sigmoid(z)

        predictions = (probabilities >= self.threshold).astype(int)

        return pd.Series(predictions, index=X.index)