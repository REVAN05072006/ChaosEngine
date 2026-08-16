from abc import ABC, abstractmethod

import pandas as pd


class BaseModel(ABC):
    """Base interface for machine learning models."""

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> None:
        """Train the model."""
        pass

    @abstractmethod
    def predict(
        self,
        X: pd.DataFrame,
    ) -> pd.Series:
        """Generate predictions."""
        pass