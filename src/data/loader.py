from pathlib import Path

import pandas as pd


def load_dataset(path: str) -> pd.DataFrame:
    """
    Load a CSV dataset.

    Args:
        path (str): Path to the CSV dataset.

    Returns:
        pd.DataFrame: Loaded dataset.

    Raises:
        FileNotFoundError: If the dataset does not exist.
        ValueError: If the dataset is empty.
    """
    dataset_path = Path(path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}"
        )

    df = pd.read_csv(dataset_path)

    if df.empty:
        raise ValueError(
            f"The dataset is empty: {dataset_path}"
        )

    return df


def show_columns(df: pd.DataFrame) -> None:
    """Display all the column names."""
    print("\nColumns:")
    for col in df.columns:
        print(f"- {col}")


def show_shape(df: pd.DataFrame) -> None:
    """Display the shape of the dataset."""
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")


def show_info(df: pd.DataFrame) -> None:
    """Display dataset information."""
    print("Dataset Information:")
    print("-" * 30)
    df.info()

    print("\nMissing Values:")
    print("-" * 30)
    print(df.isnull().sum())