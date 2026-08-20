import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


def calculate_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict[str, float]:
    """
    Calculate regression evaluation metrics.

    Args:
        y_true: Actual target values.
        y_pred: Model predictions.

    Returns:
        Dictionary containing MAE, MSE, RMSE, and R².
    """

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
    }


def calculate_classification_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict[str, float]:
    """
    Calculate binary classification evaluation metrics.

    Args:
        y_true: Actual target classes.
        y_pred: Predicted target classes.

    Returns:
        Dictionary containing accuracy, precision, recall, and F1 score.
    """

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )
    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )
    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }