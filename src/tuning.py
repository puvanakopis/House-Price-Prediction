"""
Hyperparameter tuning module for House Price Regression.
Executes systematic GridSearchCV and RandomizedSearchCV strictly on training folds
to find optimal algorithm configurations without test set leakage.
"""

from typing import Dict, Any, Tuple
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
from src.config import PARAM_GRIDS, RANDOM_STATE, CV_FOLDS
from src.models import build_full_pipeline, get_base_models


def tune_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    search_type: str = "grid",
    n_iter: int = 20,
    cv_splits: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> Tuple[Any, Dict[str, Any], float]:
    """
    Tunes hyperparameters for a specified model using cross-validation on training data.

    Args:
        model_name: Name of model ('Linear Regression', 'Ridge Regression',
                    'Decision Tree', 'Random Forest', 'Gradient Boosting').
        X_train: Training feature DataFrame.
        y_train: Training target Series.
        search_type: 'grid' or 'random'.
        n_iter: Number of parameter combinations to sample if search_type == 'random'.
        cv_splits: Number of cross-validation folds.
        random_state: Seed for reproducibility.

    Returns:
        Tuple: (best_pipeline, best_params, best_rmse)
    """
    base_models = get_base_models(random_state=random_state)
    if model_name not in base_models:
        raise ValueError(
            f"Unknown model name '{model_name}'. Available: {list(base_models.keys())}"
        )

    base_estimator = base_models[model_name]
    pipeline = build_full_pipeline(base_estimator)
    param_grid = PARAM_GRIDS.get(model_name, {})

    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    # Use RandomizedSearch if parameter grid is large, otherwise GridSearchCV
    grid_size = 1
    for k, v in param_grid.items():
        grid_size *= len(v)

    if search_type == "random" or grid_size > 30:
        searcher = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            n_iter=min(n_iter, grid_size),
            scoring="neg_root_mean_squared_error",
            cv=kf,
            random_state=random_state,
            n_jobs=None,
            refit=True,
        )
    else:
        searcher = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="neg_root_mean_squared_error",
            cv=kf,
            n_jobs=None,
            refit=True,
        )

    searcher.fit(X_train, y_train)

    best_pipeline = searcher.best_estimator_
    best_params = searcher.best_params_
    best_rmse = float(-searcher.best_score_)

    return best_pipeline, best_params, best_rmse


def tune_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Dict[str, Any]]:
    """
    Iteratively tunes all 5 algorithms and returns results dictionary.

    Args:
        X_train: Training features.
        y_train: Training targets.
        random_state: Seed for reproducibility.

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary mapping model names to tuning results.
    """
    results = {}
    base_models = get_base_models(random_state=random_state)

    for name in base_models.keys():
        print(f"--> Tuning {name}...")
        best_pipe, best_params, best_rmse = tune_model(
            model_name=name,
            X_train=X_train,
            y_train=y_train,
            search_type="grid" if name in ["Linear Regression", "Ridge Regression", "Decision Tree"] else "random",
            n_iter=4,
            random_state=random_state,
        )
        results[name] = {
            "best_pipeline": best_pipe,
            "best_params": best_params,
            "best_cv_rmse": best_rmse,
        }
        print(f"    Done! Best CV RMSE: ${best_rmse:,.2f} with params: {best_params}")

    return results
