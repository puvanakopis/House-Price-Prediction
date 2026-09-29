import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    RAW_DATA_PATH,
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    TARGET_COLUMN,
    RANDOM_STATE,
    TEST_SIZE,
    METRICS_DIR,
    PLOTS_DIR,
    PREDICTIONS_DIR,
    FINAL_MODEL_PATH,
)
from src.data_loader import load_raw_data, clean_metadata_columns
from src.models import get_all_pipelines
from src.evaluation import (
    compute_regression_metrics,
    evaluate_with_cross_validation,
    compute_residuals,
    plot_target_distribution,
    plot_correlation_heatmap,
    plot_actual_vs_predicted,
    plot_residual_analysis,
    plot_feature_importance,
    plot_model_comparison_bar,
)
from src.tuning import tune_all_models
from src.prediction import save_model


def run_experiment():
    print("=" * 70)
    print("HOUSE PRICE PREDICTION: COMPARATIVE REGRESSION EXPERIMENT")
    print("=" * 70)

    
    # 1. LOAD DATA & TRAIN/TEST SPLIT
    
    print("\n[PHASE 1 & 3] Loading raw dataset...")
    df_raw = load_raw_data(RAW_DATA_PATH)
    print(f"Loaded raw records: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns.")

    print("\n[PHASE 4] Generating EDA Exploratory Visualizations...")
    plot_target_distribution(
        df_raw[TARGET_COLUMN],
        save_path=str(PLOTS_DIR / "target_distribution.png"),
    )
    plot_correlation_heatmap(
        df_raw,
        save_path=str(PLOTS_DIR / "correlation_heatmap.png"),
    )
    print(f"Saved EDA plots to {PLOTS_DIR}")

    print("\n[PHASE 7] Performing Leakage-Free Train/Test Split (80/20)...")
    df_clean = clean_metadata_columns(df_raw)

    train_df, test_df = train_test_split(
        df_clean,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    test_df.to_csv(TEST_DATA_PATH, index=False)
    print(f"Training set: {train_df.shape[0]} rows | Test set: {test_df.shape[0]} rows")

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    
    # 2. BASELINE EXPERIMENT & 5-FOLD CROSS-VALIDATION
    
    print("\n[PHASE 8 & 9] Training 5 Baseline Models and Computing 5-Fold Cross-Validation...")
    baseline_pipelines = get_all_pipelines(random_state=RANDOM_STATE)

    baseline_metrics_list = []
    cv_metrics_list = []

    for name, pipeline in baseline_pipelines.items():
        print(f"--> Evaluating {name}...")
        # 5-Fold Cross Validation
        cv_res = evaluate_with_cross_validation(
            pipeline=pipeline,
            X=X_train,
            y=y_train,
            n_splits=5,
            random_state=RANDOM_STATE,
        )
        cv_res["Model"] = name
        cv_metrics_list.append(cv_res)

        # Single fit on full training set for baseline metrics
        pipeline.fit(X_train, y_train)
        y_train_pred = pipeline.predict(X_train)
        base_res = compute_regression_metrics(y_train, y_train_pred)
        base_res["Model"] = name
        baseline_metrics_list.append(base_res)

    baseline_df = pd.DataFrame(baseline_metrics_list)[
        ["Model", "MAE", "MSE", "RMSE", "R2", "MAPE(%)"]
    ]
    cv_df = pd.DataFrame(cv_metrics_list)[
        [
            "Model",
            "CV_MAE_Mean",
            "CV_MAE_Std",
            "CV_RMSE_Mean",
            "CV_RMSE_Std",
            "CV_R2_Mean",
            "CV_R2_Std",
        ]
    ]

    baseline_df.to_csv(METRICS_DIR / "baseline_results.csv", index=False)
    cv_df.to_csv(METRICS_DIR / "cross_validation_results.csv", index=False)

    print("\n=== BASELINE TRAINING METRICS ===")
    print(baseline_df.to_string(index=False))

    print("\n=== 5-FOLD CROSS-VALIDATION METRICS ===")
    print(cv_df.to_string(index=False))

    # Model comparison plot for Baseline CV RMSE
    plot_model_comparison_bar(
        cv_df.rename(columns={"CV_RMSE_Mean": "CV RMSE ($)"}),
        metric="CV RMSE ($)",
        title="5-Fold Cross-Validation RMSE Comparison (Baseline)",
        save_path=str(PLOTS_DIR / "model_comparison.png"),
    )

    
    # 3. HYPERPARAMETER TUNING
    
    print("\n[PHASE 10] Running Hyperparameter Tuning via GridSearchCV / RandomizedSearchCV...")
    tuned_results = tune_all_models(X_train, y_train, random_state=RANDOM_STATE)

    # Compile before vs after tuning table
    comparison_records = []
    for name in baseline_pipelines.keys():
        base_rmse = cv_df.loc[cv_df["Model"] == name, "CV_RMSE_Mean"].values[0]
        base_r2 = cv_df.loc[cv_df["Model"] == name, "CV_R2_Mean"].values[0]
        tuned_rmse = tuned_results[name]["best_cv_rmse"]
        
        # Calculate cross-validation R2 for tuned model
        tuned_pipe = tuned_results[name]["best_pipeline"]
        tuned_cv_res = evaluate_with_cross_validation(tuned_pipe, X_train, y_train)
        tuned_r2 = tuned_cv_res["CV_R2_Mean"]

        rmse_improvement = ((base_rmse - tuned_rmse) / base_rmse) * 100.0
        r2_improvement = tuned_r2 - base_r2

        comparison_records.append(
            {
                "Model": name,
                "Baseline_CV_RMSE": base_rmse,
                "Tuned_CV_RMSE": tuned_rmse,
                "RMSE_Improvement_%": round(rmse_improvement, 2),
                "Baseline_CV_R2": base_r2,
                "Tuned_CV_R2": tuned_r2,
                "R2_Improvement": round(r2_improvement, 4),
                "Best_Hyperparameters": str(tuned_results[name]["best_params"]),
            }
        )

    tuned_comparison_df = pd.DataFrame(comparison_records)
    tuned_comparison_df.to_csv(METRICS_DIR / "tuned_results.csv", index=False)

    print("\n=== BEFORE VS AFTER TUNING COMPARISON ===")
    print(
        tuned_comparison_df[
            [
                "Model",
                "Baseline_CV_RMSE",
                "Tuned_CV_RMSE",
                "RMSE_Improvement_%",
                "Baseline_CV_R2",
                "Tuned_CV_R2",
                "R2_Improvement",
            ]
        ].to_string(index=False)
    )

    
    # 4. FINAL MODEL SELECTION & TEST EVALUATION
    
    print("\n[PHASE 14 & 15] Selecting Final Model & Evaluating on Untouched Test Set...")
    # Select candidate with lowest Tuned CV RMSE
    best_candidate_name = tuned_comparison_df.sort_values(
        by="Tuned_CV_RMSE", ascending=True
    ).iloc[0]["Model"]
    print(f"\nWinning Model Selected based on CV RMSE: {best_candidate_name}")

    # Evaluate ALL tuned models on untouched test set for comprehensive benchmark
    test_records = []
    test_preds_dict = {}

    for name, res in tuned_results.items():
        pipe = res["best_pipeline"]
        y_test_pred = pipe.predict(X_test)
        test_preds_dict[name] = y_test_pred
        metrics = compute_regression_metrics(y_test, y_test_pred)
        metrics["Model"] = name
        test_records.append(metrics)

    final_test_df = pd.DataFrame(test_records)[
        ["Model", "MAE", "MSE", "RMSE", "R2", "MAPE(%)"]
    ]
    final_test_df.to_csv(METRICS_DIR / "final_test_results.csv", index=False)

    print("\n=== UNTOUCHED FINAL TEST SET EVALUATION ===")
    print(final_test_df.to_string(index=False))

    # Save test predictions for winning model
    winning_pipeline = tuned_results[best_candidate_name]["best_pipeline"]
    winning_test_pred = test_preds_dict[best_candidate_name]

    pred_df = pd.DataFrame(
        {
            "Actual_Price": y_test.values,
            "Predicted_Price": winning_test_pred,
            "Residual": y_test.values - winning_test_pred,
            "Absolute_Error": np.abs(y_test.values - winning_test_pred),
            "Percentage_Error": (
                np.abs(y_test.values - winning_test_pred) / y_test.values
            )
            * 100.0,
        }
    )
    pred_df.to_csv(PREDICTIONS_DIR / "test_predictions.csv", index=False)

    
    # 5. ERROR ANALYSIS & RESIDUAL PLOTS
    
    print("\n[PHASE 12] Performing In-Depth Error Analysis...")
    plot_actual_vs_predicted(
        y_true=y_test.values,
        y_pred=winning_test_pred,
        model_name=f"{best_candidate_name} (Tuned)",
        save_path=str(PLOTS_DIR / "actual_vs_predicted.png"),
    )

    plot_residual_analysis(
        y_true=y_test.values,
        y_pred=winning_test_pred,
        model_name=f"{best_candidate_name} (Tuned)",
        save_path=str(PLOTS_DIR / "residual_vs_predicted.png"),
    )

    
    # 6. FEATURE IMPORTANCE ANALYSIS
    
    print("\n[PHASE 17] Extracting Feature Importances...")
    # Extract fitted preprocessor to get feature names
    fitted_preprocessor = winning_pipeline.named_steps["preprocessor"]
    fitted_regressor = winning_pipeline.named_steps["regressor"]

    # Transform a sample to inspect column names
    transformed_sample = winning_pipeline.named_steps["feature_engineer"].transform(
        X_train.head(10)
    )
    num_cols = (
        fitted_preprocessor.transformers_[0][2]
    )
    cat_encoder = fitted_preprocessor.named_transformers_["cat"].named_steps["encoder"]
    cat_cols_out = list(cat_encoder.get_feature_names_out(fitted_preprocessor.transformers_[1][2]))
    all_feature_names = list(num_cols) + cat_cols_out

    if hasattr(fitted_regressor, "feature_importances_"):
        importances = fitted_regressor.feature_importances_
        plot_feature_importance(
            feature_names=all_feature_names,
            importances=importances,
            title=f"Top 20 Feature Importances — {best_candidate_name}",
            top_n=20,
            save_path=str(PLOTS_DIR / "feature_importance.png"),
        )
    elif hasattr(fitted_regressor, "coef_"):
        coefs = np.abs(fitted_regressor.coef_)
        plot_feature_importance(
            feature_names=all_feature_names,
            importances=coefs,
            title=f"Top 20 Absolute Standardized Coefficients — {best_candidate_name}",
            top_n=20,
            save_path=str(PLOTS_DIR / "feature_importance.png"),
        )

    
    # 7. SAVE WINNING PRODUCTION PIPELINE
    
    print(f"\n[PHASE 13] Saving final winning pipeline to {FINAL_MODEL_PATH}...")
    save_model(winning_pipeline, FINAL_MODEL_PATH)
    print("Experiment completed successfully!")


if __name__ == "__main__":
    run_experiment()
