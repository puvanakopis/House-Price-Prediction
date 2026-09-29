"""
Model definition and factory module for the 5 regression algorithms:
1. Linear Regression (Baseline)
2. Ridge Regression (L2 Regularization)
3. Decision Tree Regression (Non-linear partitioning)
4. Random Forest Regression (Bagging ensemble)
5. Gradient Boosting Regression (Boosting ensemble)
"""

from typing import Dict, Any, Optional
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from src.config import RANDOM_STATE
from src.feature_engineering import FeatureEngineerTransformer
from src.preprocessing import build_preprocessor


def get_base_models(random_state: int = RANDOM_STATE) -> Dict[str, Any]:
    """
    Instantiates the 5 core regression algorithms with default sensible baseline parameters.

    Args:
        random_state: Seed for reproducibility in stochastic algorithms.

    Returns:
        Dict[str, Any]: Dictionary mapping algorithm names to scikit-learn regressor instances.
    """
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=random_state),
        "Decision Tree": DecisionTreeRegressor(random_state=random_state),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=random_state, n_jobs=1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, random_state=random_state
        ),
    }


def build_full_pipeline(regressor: Any) -> Pipeline:
    """
    Assembles an end-to-end scikit-learn Pipeline incorporating:
    1. Feature engineering
    2. Missing value imputation + scaling + one-hot encoding
    3. Regression estimator

    This guarantees atomic execution during cross-validation, hyperparameter tuning,
    and inference without any data leakage.

    Args:
        regressor: Scikit-learn regression estimator.

    Returns:
        Pipeline: Complete end-to-end pipeline.
    """
    preprocessor = build_preprocessor()
    return Pipeline(
        steps=[
            ("feature_engineer", FeatureEngineerTransformer()),
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )


def get_all_pipelines(random_state: int = RANDOM_STATE) -> Dict[str, Pipeline]:
    """
    Builds full end-to-end pipelines for each of the 5 regression algorithms.

    Args:
        random_state: Random state seed.

    Returns:
        Dict[str, Pipeline]: Dictionary of named regression pipelines.
    """
    base_models = get_base_models(random_state=random_state)
    return {name: build_full_pipeline(model) for name, model in base_models.items()}
