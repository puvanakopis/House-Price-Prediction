"""
Unit tests for preprocessing and feature engineering pipelines.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from src.config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES
from src.feature_engineering import engineer_features, FeatureEngineerTransformer
from src.preprocessing import build_preprocessor, get_all_numerical_features


@pytest.fixture
def sample_raw_dataframe():
    """Provides a synthetic single-batch DataFrame mimicking raw house attributes."""
    return pd.DataFrame(
        {
            "bedrooms": [3, 4, 2],
            "bathrooms": [2.0, 2.5, 1.0],
            "sqft_living": [1800, 2400, 1100],
            "sqft_lot": [6000, 8000, 4500],
            "floors": [1.5, 2.0, 1.0],
            "waterfront": [0, 1, 0],
            "view": [0, 2, 0],
            "condition": [3, 4, 3],
            "grade": [7, 8, 6],
            "sqft_above": [1400, 2000, 1100],
            "sqft_basement": [400, 400, 0],
            "yr_built": [1975, 1995, 1950],
            "yr_renovated": [0, 2010, 0],
            "zipcode": [98178, 98125, 98028],
            "lat": [47.5112, 47.7210, 47.7379],
            "long": [-122.2570, -122.3190, -122.2330],
            "sqft_living15": [1500, 2200, 1200],
            "sqft_lot15": [5800, 7800, 4400],
        }
    )


def test_engineer_features(sample_raw_dataframe):
    """Verifies that all derived features are correctly calculated without NaNs."""
    df_eng = engineer_features(sample_raw_dataframe, reference_year=2015)

    expected_new_cols = [
        "house_age",
        "is_renovated",
        "years_since_renovation",
        "total_rooms",
        "sqft_per_bedroom",
        "sqft_per_bathroom",
        "living_to_lot_ratio",
        "distance_to_center_km",
    ]

    for col in expected_new_cols:
        assert col in df_eng.columns, f"Engineered column {col} is missing."
        assert not df_eng[col].isna().any(), f"Engineered column {col} has unexpected NaN."

    # Check logic
    assert df_eng.loc[0, "house_age"] == 2015 - 1975
    assert df_eng.loc[1, "is_renovated"] == 1
    assert df_eng.loc[0, "is_renovated"] == 0
    assert df_eng.loc[0, "total_rooms"] == 5.0
    assert df_eng.loc[0, "distance_to_center_km"] > 0.0


def test_preprocessor_pipeline_fit_transform(sample_raw_dataframe):
    """Tests that the preprocessor ColumnTransformer handles imputation and scaling correctly."""
    df_eng = engineer_features(sample_raw_dataframe)
    preprocessor = build_preprocessor()

    transformed = preprocessor.fit_transform(df_eng)
    assert isinstance(transformed, np.ndarray)
    assert transformed.shape[0] == sample_raw_dataframe.shape[0]
    assert not np.isnan(transformed).any()


def test_transformer_handles_missing_values():
    """Tests that missing numerical and categorical values are successfully imputed."""
    df_missing = pd.DataFrame(
        {
            "bedrooms": [3, np.nan, 2],
            "bathrooms": [2.0, 2.5, np.nan],
            "sqft_living": [1800, np.nan, 1100],
            "sqft_lot": [6000, 8000, 4500],
            "floors": [1.5, 2.0, 1.0],
            "waterfront": [0, 0, 0],
            "view": [0, 0, 0],
            "condition": [3, 3, 3],
            "grade": [7, 7, 6],
            "sqft_above": [1400, 1400, 1100],
            "sqft_basement": [400, 0, 0],
            "yr_built": [1975, 1990, 1950],
            "yr_renovated": [0, 0, 0],
            "zipcode": [98178, np.nan, 98028],
            "lat": [47.51, 47.72, 47.73],
            "long": [-122.25, -122.31, -122.23],
            "sqft_living15": [1500, 1800, 1200],
            "sqft_lot15": [5800, 7800, 4400],
        }
    )

    df_eng = engineer_features(df_missing)
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(df_eng)
    assert not np.isnan(transformed).any()
