from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)
from sklearn.model_selection import KFold, cross_validate
from src.config import PLOTS_DIR, RANDOM_STATE, CV_FOLDS


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculates primary regression performance metrics.

    Args:
        y_true: Ground truth actual prices.
        y_pred: Predicted prices from model.

    Returns:
        Dict[str, float]: Dictionary with MAE, MSE, RMSE, R2, and MAPE.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(mean_absolute_percentage_error(y_true, y_pred)) * 100.0

    return {
        "MAE": round(mae, 2),
        "MSE": round(mse, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4),
        "MAPE(%)": round(mape, 2),
    }


def evaluate_with_cross_validation(
    pipeline: Any,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> Dict[str, float]:
    """
    Executes k-fold cross validation computing mean and standard deviation for MAE, RMSE, and R2.

    Args:
        pipeline: Scikit-learn Pipeline instance.
        X: Training feature matrix.
        y: Training target vector.
        n_splits: Number of CV folds (default 5).
        random_state: Seed for KFold shuffling.

    Returns:
        Dict[str, float]: Aggregated CV metrics (mean and std).
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = {
        "neg_mae": "neg_mean_absolute_error",
        "neg_mse": "neg_mean_squared_error",
        "r2": "r2",
    }

    cv_results = cross_validate(
        pipeline, X, y, cv=kf, scoring=scoring, n_jobs=None, return_train_score=False
    )

    mae_scores = -cv_results["test_neg_mae"]
    mse_scores = -cv_results["test_neg_mse"]
    rmse_scores = np.sqrt(mse_scores)
    r2_scores = cv_results["test_r2"]

    return {
        "CV_MAE_Mean": round(float(np.mean(mae_scores)), 2),
        "CV_MAE_Std": round(float(np.std(mae_scores)), 2),
        "CV_RMSE_Mean": round(float(np.mean(rmse_scores)), 2),
        "CV_RMSE_Std": round(float(np.std(rmse_scores)), 2),
        "CV_R2_Mean": round(float(np.mean(r2_scores)), 4),
        "CV_R2_Std": round(float(np.std(r2_scores)), 4),
    }


def compute_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """
    Calculates individual residual errors and percentage errors.

    Args:
        y_true: Array of actual values.
        y_pred: Array of predicted values.

    Returns:
        pd.DataFrame: Table with Actual, Predicted, Residual, Absolute_Error, and Pct_Error.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    residuals = y_t - y_p
    abs_errors = np.abs(residuals)
    pct_errors = (abs_errors / np.clip(y_t, 1e-5, None)) * 100.0

    return pd.DataFrame(
        {
            "Actual": y_t,
            "Predicted": y_p,
            "Residual": residuals,
            "Abs_Error": abs_errors,
            "Pct_Error": pct_errors,
        }
    )



def set_plot_style():
    """Configures clean, modern plot aesthetic."""
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.titlesize": 16,
            "figure.dpi": 300,
        }
    )


def plot_target_distribution(y: pd.Series, save_path: str = None) -> plt.Figure:
    """Plots original target distribution alongside log-transformed distribution."""
    set_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(y, kde=True, ax=ax1, color="#2b5c8f", bins=50)
    ax1.set_title("House Price Distribution (Skewness: {:.2f})".format(y.skew()))
    ax1.set_xlabel("Price ($)")
    ax1.set_ylabel("Frequency")
    ax1.xaxis.set_major_formatter("${x:,.0f}")

    sns.histplot(np.log1p(y), kde=True, ax=ax2, color="#2e8b57", bins=50)
    ax2.set_title("Log(Price) Distribution (Normalized)")
    ax2.set_xlabel("Log(Price + 1)")
    ax2.set_ylabel("Frequency")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, save_path: str = None) -> plt.Figure:
    """Computes and visualizes correlation matrix for numerical features."""
    set_plot_style()
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8},
        ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap", pad=20)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_actual_vs_predicted(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str, save_path: str = None
) -> plt.Figure:
    """Generates scatter plot of Actual vs Predicted prices with 45-degree parity line."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 7))

    ax.scatter(y_true, y_pred, alpha=0.3, color="#1f77b4", edgecolors="none", s=25)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Fit (y = x)")

    ax.set_title(f"Actual vs Predicted Prices — {model_name}")
    ax.set_xlabel("Actual Price ($)")
    ax.set_ylabel("Predicted Price ($)")
    ax.xaxis.set_major_formatter("${x:,.0f}")
    ax.yaxis.set_major_formatter("${x:,.0f}")
    ax.legend(loc="upper left")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_residual_analysis(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str, save_path: str = None
) -> plt.Figure:
    """Generates dual residual diagnostics: Residual vs Predicted and Residual Distribution."""
    set_plot_style()
    residuals = y_true - y_pred

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Residuals vs Predicted
    ax1.scatter(y_pred, residuals, alpha=0.3, color="#4c72b0", edgecolors="none", s=25)
    ax1.axhline(0, color="red", linestyle="--", lw=2)
    ax1.set_title(f"Residuals vs Predicted — {model_name}")
    ax1.set_xlabel("Predicted Price ($)")
    ax1.set_ylabel("Residual (Actual - Predicted) ($)")
    ax1.xaxis.set_major_formatter("${x:,.0f}")
    ax1.yaxis.set_major_formatter("${x:,.0f}")

    # Residual Histogram + KDE
    sns.histplot(residuals, kde=True, ax=ax2, color="#c44e52", bins=50)
    ax2.axvline(0, color="black", linestyle="--", lw=1.5)
    ax2.set_title(f"Residuals Distribution — {model_name}")
    ax2.set_xlabel("Residual ($)")
    ax2.set_ylabel("Count")
    ax2.xaxis.set_major_formatter("${x:,.0f}")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    title: str = "Feature Importance",
    top_n: int = 20,
    save_path: str = None,
) -> plt.Figure:
    """Visualizes Top N feature importances or standardized coefficient magnitudes."""
    set_plot_style()
    df_imp = pd.DataFrame({"feature": feature_names, "importance": importances})
    df_imp = df_imp.sort_values(by="importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(
        data=df_imp,
        x="importance",
        y="feature",
        palette="viridis",
        hue="feature",
        legend=False,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Importance / Relative Weight")
    ax.set_ylabel("Feature Name")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_model_comparison_bar(
    comparison_df: pd.DataFrame,
    metric: str = "RMSE",
    title: str = "Model Comparison",
    save_path: str = None,
) -> plt.Figure:
    """Generates comparison bar chart across models for a specific metric."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    sorted_df = comparison_df.sort_values(by=metric, ascending=True)
    sns.barplot(
        data=sorted_df,
        x="Model",
        y=metric,
        palette="crest",
        hue="Model",
        legend=False,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel(metric)
    plt.xticks(rotation=25, ha="right")

    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height):
            ax.annotate(
                f"{height:,.2f}" if metric != "R2" else f"{height:.4f}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=9,
                xytext=(0, 3),
                textcoords="offset points",
            )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig
