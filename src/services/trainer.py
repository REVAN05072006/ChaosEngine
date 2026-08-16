from src.models.base_model import BaseModel

import pandas as pd


class Trainer:
    """Train machine learning models."""

    def __init__(self, model: BaseModel):
        self.model = model

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> BaseModel:
        """Train the model and return the trained model."""

        self.model.fit(X_train, y_train)

        return self.model