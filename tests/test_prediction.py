from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from src.models import build_full_pipeline
from src.prediction import (
    save_model,
    load_model,
    validate_input_data,
    predict_house_price,
)


@pytest.fixture
def dummy_fitted_pipeline(tmp_path):
    """Creates and saves a quick dummy pipeline to a temporary path."""
    pipe = build_full_pipeline(LinearRegression())
    df = pd.DataFrame(
        {
            "bedrooms": [3, 4, 2, 3],
            "bathrooms": [2.0, 2.5, 1.0, 2.0],
            "sqft_living": [1800, 2400, 1100, 2000],
            "sqft_lot": [6000, 8000, 4500, 5000],
            "floors": [1.5, 2.0, 1.0, 1.0],
            "waterfront": [0, 0, 0, 0],
            "view": [0, 0, 0, 0],
            "condition": [3, 3, 3, 3],
            "grade": [7, 8, 6, 7],
            "sqft_above": [1400, 2000, 1100, 1500],
            "sqft_basement": [400, 400, 0, 500],
            "yr_built": [1975, 1995, 1950, 1980],
            "yr_renovated": [0, 0, 0, 0],
            "zipcode": [98178, 98125, 98028, 98178],
            "lat": [47.5112, 47.7210, 47.7379, 47.6000],
            "long": [-122.2570, -122.3190, -122.2330, -122.3000],
            "sqft_living15": [1500, 2200, 1200, 1700],
            "sqft_lot15": [5800, 7800, 4400, 4800],
        }
    )
    y = pd.Series([400000.0, 650000.0, 280000.0, 450000.0])
    pipe.fit(df, y)
    model_path = tmp_path / "test_pipeline.joblib"
    save_model(pipe, model_path)
    return model_path


def test_input_validation():
    """Verifies that validate_input_data normalizes dicts and dataframes properly."""
    input_dict = {
        "bedrooms": 3,
        "bathrooms": 2.0,
        "sqft_living": 1800,
        "sqft_lot": 5000,
        "floors": 1.0,
        "waterfront": 0,
        "view": 0,
        "condition": 3,
        "grade": 7,
        "sqft_above": 1800,
        "sqft_basement": 0,
        "yr_built": 1990,
        "yr_renovated": 0,
        "zipcode": 98178,
        "lat": 47.51,
        "long": -122.25,
        "sqft_living15": 1600,
        "sqft_lot15": 5000,
    }

    df = validate_input_data(input_dict)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 1
    assert df["bedrooms"].iloc[0] == 3


def test_predict_house_price(dummy_fitted_pipeline):
    """Verifies that predict_house_price returns numeric float prediction."""
    sample_house = {
        "bedrooms": 3,
        "bathrooms": 2.0,
        "sqft_living": 1800,
        "sqft_lot": 5000,
        "floors": 1.0,
        "waterfront": 0,
        "view": 0,
        "condition": 3,
        "grade": 7,
        "sqft_above": 1800,
        "sqft_basement": 0,
        "yr_built": 1990,
        "yr_renovated": 0,
        "zipcode": 98178,
        "lat": 47.51,
        "long": -122.25,
        "sqft_living15": 1600,
        "sqft_lot15": 5000,
    }

    pred = predict_house_price(sample_house, model_path=dummy_fitted_pipeline)
    assert isinstance(pred, np.ndarray)
    assert len(pred) == 1
    assert np.issubdtype(pred.dtype, np.floating)
    assert not np.isnan(pred[0])
    assert pred[0] > 0
