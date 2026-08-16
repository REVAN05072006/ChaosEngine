import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
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