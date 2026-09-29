from typing import Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from src.config import SEATTLE_DOWNTOWN_LAT, SEATTLE_DOWNTOWN_LONG


def haversine_distance(lat1: np.ndarray, lon1: np.ndarray, lat2: float, lon2: float) -> np.ndarray:
    """
    Calculates great-circle distance (in kilometers) between arrays of geographic coordinates.

    Args:
        lat1: Array of latitudes in degrees.
        lon1: Array of longitudes in degrees.
        lat2: Reference latitude in degrees.
        lon2: Reference longitude in degrees.

    Returns:
        np.ndarray: Distance in kilometers.
    """
    r_earth_km = 6371.0  # Mean radius of Earth in km
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return r_earth_km * c


def engineer_features(df: pd.DataFrame, reference_year: int = 2015) -> pd.DataFrame:
    """
    Applies deterministic domain-specific feature engineering.
    Guaranteed to only use features available at inference time (no target leakage).

    Derived Features:
    - house_age: Years since construction relative to reference year.
    - is_renovated: Binary indicator of property renovation.
    - years_since_renovation: Age since renovation (or construction age if never renovated).
    - total_rooms: Sum of bedrooms and bathrooms.
    - sqft_per_bedroom: Living space divided by bedroom count.
    - sqft_per_bathroom: Living space divided by bathroom count.
    - living_to_lot_ratio: Ratio of living area to total lot parcel area.
    - distance_to_center_km: Great-circle distance to Downtown Seattle.

    Args:
        df: Input DataFrame containing raw features.
        reference_year: Reference year for age calculation (default 2015).

    Returns:
        pd.DataFrame: DataFrame augmented with engineered features.
    """
    df_out = df.copy()

    # 1. Age-related features
    if "yr_built" in df_out.columns:
        df_out["house_age"] = (reference_year - df_out["yr_built"]).clip(lower=0)

    if "yr_renovated" in df_out.columns:
        df_out["is_renovated"] = (df_out["yr_renovated"] > 0).astype(int)
        if "yr_built" in df_out.columns:
            renovated_age = reference_year - df_out["yr_renovated"]
            built_age = reference_year - df_out["yr_built"]
            df_out["years_since_renovation"] = np.where(
                df_out["yr_renovated"] > 0,
                renovated_age.clip(lower=0),
                built_age.clip(lower=0)
            )

    # 2. Room composition & proportions
    if "bedrooms" in df_out.columns and "bathrooms" in df_out.columns:
        df_out["total_rooms"] = df_out["bedrooms"] + df_out["bathrooms"]

    if "sqft_living" in df_out.columns and "bedrooms" in df_out.columns:
        df_out["sqft_per_bedroom"] = df_out["sqft_living"] / df_out["bedrooms"].clip(lower=1)

    if "sqft_living" in df_out.columns and "bathrooms" in df_out.columns:
        df_out["sqft_per_bathroom"] = df_out["sqft_living"] / df_out["bathrooms"].clip(lower=0.5)

    if "sqft_living" in df_out.columns and "sqft_lot" in df_out.columns:
        df_out["living_to_lot_ratio"] = df_out["sqft_living"] / df_out["sqft_lot"].clip(lower=1.0)

    # 3. Location distance to economic core (Downtown Seattle)
    if "lat" in df_out.columns and "long" in df_out.columns:
        df_out["distance_to_center_km"] = haversine_distance(
            df_out["lat"].values,
            df_out["long"].values,
            SEATTLE_DOWNTOWN_LAT,
            SEATTLE_DOWNTOWN_LONG
        )

    return df_out


class FeatureEngineerTransformer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible Transformer wrapper for feature engineering.
    Enables seamless integration inside scikit-learn Pipelines.
    """

    def __init__(self, reference_year: int = 2015):
        self.reference_year = reference_year

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return engineer_features(X, reference_year=self.reference_year)
