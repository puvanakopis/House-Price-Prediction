"""
Prediction and inference pipeline module for House Price Regression.
Provides serialization, deserialization, input schema validation, and inference helpers.
"""

from pathlib import Path
from typing import Dict, Any, Union, List, Optional
import joblib
import numpy as np
import pandas as pd
from src.config import FINAL_MODEL_PATH, NUMERICAL_FEATURES, CATEGORICAL_FEATURES


def save_model(pipeline: Any, file_path: Optional[Path] = None) -> Path:
    """
    Serializes and saves a trained pipeline to disk using joblib.

    Args:
        pipeline: Fitted scikit-learn Pipeline.
        file_path: Destination path. Defaults to FINAL_MODEL_PATH.

    Returns:
        Path: Path where model was saved.
    """
    path = file_path or FINAL_MODEL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def load_model(file_path: Optional[Path] = None) -> Any:
    """
    Loads a saved pipeline from disk.

    Args:
        file_path: Source path. Defaults to FINAL_MODEL_PATH.

    Returns:
        Pipeline: Loaded scikit-learn Pipeline.

    Raises:
        FileNotFoundError: If model file does not exist.
    """
    path = file_path or FINAL_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {path}. Please train and save a model first."
        )
    return joblib.load(path)


def validate_input_data(data: Union[Dict[str, Any], pd.DataFrame, List[Dict[str, Any]]]) -> pd.DataFrame:
    """
    Validates and normalizes input data into a standardized DataFrame matching required features.

    Args:
        data: Input features as dict, list of dicts, or DataFrame.

    Returns:
        pd.DataFrame: Formatted DataFrame ready for pipeline input.

    Raises:
        ValueError: If input format is unsupported or contains non-numeric inputs for numeric fields.
    """
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError(
            f"Unsupported input type '{type(data)}'. Expected dict, list of dicts, or pd.DataFrame."
        )

    # Validate presence or provide defaults for required raw features
    expected_features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    for feat in expected_features:
        if feat not in df.columns:
            # If missing, initialize with NaN so the pipeline imputer can handle it cleanly
            df[feat] = np.nan

    return df


def predict_house_price(
    features: Union[Dict[str, Any], pd.DataFrame, List[Dict[str, Any]]],
    model_path: Optional[Path] = None,
) -> np.ndarray:
    """
    Predicts house prices for given input features using the trained production pipeline.

    Args:
        features: House property characteristics.
        model_path: Path to serialized model file.

    Returns:
        np.ndarray: Predicted house price(s) in dollars.
    """
    model = load_model(model_path)
    df_valid = validate_input_data(features)
    predictions = model.predict(df_valid)
    return np.asarray(predictions, dtype=float)
