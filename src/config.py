from pathlib import Path
from typing import List, Dict, Any

# Root Project Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "house_sales.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"

MODELS_DIR = BASE_DIR / "models"
FINAL_MODEL_PATH = MODELS_DIR / "final_model.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"

RESULTS_DIR = BASE_DIR / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
PLOTS_DIR = RESULTS_DIR / "plots"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

# Experiment & Split Configuration
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.20
CV_FOLDS: int = 5

# Target Column
TARGET_COLUMN: str = "price"

# Identifier / Metadata Columns to Drop
COLUMNS_TO_DROP: List[str] = ["date"]

# Raw Numerical Features
NUMERICAL_FEATURES: List[str] = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "grade",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
    "lat",
    "long",
    "sqft_living15",
    "sqft_lot15",
]

# Raw Categorical Features
CATEGORICAL_FEATURES: List[str] = [
    "zipcode"
]

# Downtown Seattle reference coordinates for distance calculations
SEATTLE_DOWNTOWN_LAT: float = 47.6062
SEATTLE_DOWNTOWN_LONG: float = -122.3321

# Hyperparameter Tuning Grids
PARAM_GRIDS: Dict[str, Dict[str, Any]] = {
    "Linear Regression": {
        "regressor__fit_intercept": [True, False]
    },
    "Ridge Regression": {
        "regressor__alpha": [0.1, 1.0, 10.0, 100.0],
        "regressor__solver": ["auto", "cholesky", "lsqr"]
    },
    "Decision Tree": {
        "regressor__max_depth": [8, 12],
        "regressor__min_samples_split": [2, 5],
        "regressor__min_samples_leaf": [1, 2]
    },
    "Random Forest": {
        "regressor__n_estimators": [50, 100],
        "regressor__max_depth": [10, 15, 20],
        "regressor__min_samples_split": [2, 5],
        "regressor__min_samples_leaf": [1, 2],
    },
    "Gradient Boosting": {
        "regressor__n_estimators": [80, 120],
        "regressor__learning_rate": [0.05, 0.1],
        "regressor__max_depth": [3, 5],
        "regressor__subsample": [0.8, 1.0]
    }
}
