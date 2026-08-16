import pandas as pd
from sklearn.preprocessing import StandardScaler


def prepare_features(
    df: pd.DataFrame,
    target: str,
    features: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate features and target from a dataset.

    Args:
        df: Input dataset.
        target: Name of the target column.
        features: Names of columns to use as model features.

    Returns:
        A tuple containing X and y.

    Raises:
        ValueError: If the target or any requested feature is missing.
    """
    required_columns = features + [target]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Columns not found in dataset: {missing_columns}"
        )

    X = df[features].copy()
    y = df[target].copy()

    return X, y


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Scale training and testing features using StandardScaler.

    The scaler is fitted only on the training data to prevent
    data leakage.

    Args:
        X_train: Training features.
        X_test: Testing features.

    Returns:
        A tuple containing scaled X_train, scaled X_test,
        and the fitted scaler.
    """
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=X_train.columns,
        index=X_train.index,
    )

    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=X_test.columns,
        index=X_test.index,
    )

    return X_train_scaled, X_test_scaled, scaler