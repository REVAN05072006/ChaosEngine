from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split features and target into training and testing sets.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        test_size: Proportion of data reserved for testing.
        random_state: Seed for reproducible splitting.

    Returns:
        X_train, X_test, y_train, y_test.
    """

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )