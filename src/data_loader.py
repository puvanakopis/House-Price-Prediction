from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
from src.config import RAW_DATA_PATH, TARGET_COLUMN, COLUMNS_TO_DROP


def load_raw_data(data_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Loads the raw housing dataset from disk.

    Args:
        data_path: Optional Path to the raw CSV file. Defaults to RAW_DATA_PATH.

    Returns:
        pd.DataFrame: Loaded dataset.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If dataset is empty or missing target column.
    """
    path = data_path or RAW_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Please verify the file path or run data download script."
        )

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Dataset at {path} is completely empty.")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' is missing from the dataset. Found columns: {list(df.columns)}"
        )

    return df


def clean_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops unnecessary identifier and date metadata columns without touching feature distributions.

    Args:
        df: Input DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame with metadata columns removed.
    """
    df_cleaned = df.copy()
    existing_drop_cols = [c for c in COLUMNS_TO_DROP if c in df_cleaned.columns]
    if existing_drop_cols:
        df_cleaned = df_cleaned.drop(columns=existing_drop_cols)
    return df_cleaned


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """
    Returns high-level metadata and integrity summary of the dataset.

    Args:
        df: Input DataFrame.

    Returns:
        dict: Summary statistics including row count, column count, duplicates, and missing values.
    """
    return {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "total_missing_values": int(df.isna().sum().sum()),
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }
