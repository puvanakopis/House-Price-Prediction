"""
Streamlit Web Application: House Price Prediction System.
Provides an intuitive, professional UI for entering property characteristics,
predicting estimated market price, and inspecting model performance diagnostics.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# Ensure root directory is accessible
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import FINAL_MODEL_PATH, METRICS_DIR, PLOTS_DIR
from src.prediction import predict_house_price, load_model

# Page Configuration
st.set_page_config(
    page_title="House Price Prediction AI",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for clean aesthetic
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .price-display {
        font-size: 2.5rem;
        font-weight: 800;
        color: #16a34a;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    st.markdown(
        '<div class="main-title">🏡 Comparative House Price Prediction System</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-title">End-to-End Machine Learning Evaluation & Real-Time Property Valuation</div>',
        unsafe_allow_html=True,
    )

    # Sidebar Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Mode:",
        ["🎯 Property Price Predictor", "📊 Model Comparison & Benchmarks", "ℹ️ About the Project"],
    )

    # Tab 1: Predictor
    if page == "🎯 Property Price Predictor":
        st.subheader("Enter House Specifications")
        st.write("Provide the property attributes below to obtain a valuation from the tuned production model.")

        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("##### 📐 Space & Dimensions")
                sqft_living = st.number_input(
                    "Living Area (sqft)", min_value=300, max_value=15000, value=2000, step=50
                )
                sqft_lot = st.number_input(
                    "Lot Size (sqft)", min_value=500, max_value=500000, value=7500, step=100
                )
                sqft_above = st.number_input(
                    "Above Ground Area (sqft)", min_value=300, max_value=12000, value=1600, step=50
                )
                sqft_basement = st.number_input(
                    "Basement Area (sqft)", min_value=0, max_value=5000, value=400, step=50
                )
                floors = st.selectbox("Number of Floors", [1.0, 1.5, 2.0, 2.5, 3.0, 3.5], index=2)

            with col2:
                st.markdown("##### 🛏️ Rooms & Quality")
                bedrooms = st.slider("Bedrooms", min_value=1, max_value=10, value=3)
                bathrooms = st.slider("Bathrooms", min_value=0.5, max_value=8.0, value=2.25, step=0.25)
                grade = st.slider(
                    "Construction Grade (1-13)",
                    min_value=1,
                    max_value=13,
                    value=7,
                    help="King County construction & design quality scale (1-3: poor, 7: average, 11-13: luxury)",
                )
                condition = st.slider(
                    "Property Condition (1-5)",
                    min_value=1,
                    max_value=5,
                    value=3,
                    help="1: poor, 3: average, 5: exceptional maintenance",
                )
                view = st.selectbox("View Quality (0-4)", [0, 1, 2, 3, 4], index=0)
                waterfront = st.radio("Waterfront Property?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", horizontal=True)

            with col3:
                st.markdown("##### 📍 Location & Age")
                yr_built = st.number_input("Year Built", min_value=1900, max_value=2023, value=1985)
                is_renovated_choice = st.checkbox("Has the property been renovated?")
                yr_renovated = (
                    st.number_input("Year Renovated", min_value=1930, max_value=2023, value=2010)
                    if is_renovated_choice
                    else 0
                )
                zipcode = st.number_input(
                    "ZIP Code", min_value=98001, max_value=98199, value=98052, step=1
                )
                lat = st.number_input(
                    "Latitude", min_value=47.1, max_value=47.8, value=47.6062, format="%.4f"
                )
                long = st.number_input(
                    "Longitude", min_value=-122.6, max_value=-121.3, value=-122.3321, format="%.4f"
                )
                sqft_living15 = st.number_input(
                    "Avg Living Area of 15 Nearest Neighbors (sqft)", min_value=300, max_value=10000, value=1900
                )
                sqft_lot15 = st.number_input(
                    "Avg Lot Size of 15 Nearest Neighbors (sqft)", min_value=500, max_value=100000, value=7000
                )

            submit_button = st.form_submit_button("💰 Predict House Price", use_container_width=True)

        if submit_button:
            input_data = {
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "sqft_living": sqft_living,
                "sqft_lot": sqft_lot,
                "floors": floors,
                "waterfront": waterfront,
                "view": view,
                "condition": condition,
                "grade": grade,
                "sqft_above": sqft_above,
                "sqft_basement": sqft_basement,
                "yr_built": yr_built,
                "yr_renovated": yr_renovated,
                "zipcode": int(zipcode),
                "lat": float(lat),
                "long": float(long),
                "sqft_living15": sqft_living15,
                "sqft_lot15": sqft_lot15,
            }

            try:
                predicted_val = predict_house_price(input_data)[0]

                st.markdown("---")
                st.markdown(
                    f"""
                    <div class="metric-box">
                        <h3>Estimated Market Price</h3>
                        <div class="price-display">${predicted_val:,.2f}</div>
                        <p style="color: #64748b; margin: 0;">Predicted using full cross-validated and tuned ensemble pipeline.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Property Highlights summary
                st.markdown("##### 📌 Input Summary")
                res_col1, res_col2, res_col3, res_col4 = st.columns(4)
                res_col1.metric("Living Area", f"{sqft_living:,} sqft")
                res_col2.metric("Bedrooms / Baths", f"{bedrooms} / {bathrooms}")
                res_col3.metric("Property Age", f"{2015 - yr_built} yrs")
                res_col4.metric("Grade / Condition", f"{grade} / {condition}")

            except Exception as e:
                st.error(f"Prediction error: {e}. Ensure model has been trained and saved.")

    # Tab 2: Benchmarks
    elif page == "📊 Model Comparison & Benchmarks":
        st.subheader("Comparative Scientific Evaluation")
        st.write("Rigorous multi-metric comparison of 5 regression algorithms across Baseline, Cross-Validation, and Test sets.")

        # Load metrics tables if available
        base_path = METRICS_DIR / "baseline_results.csv"
        cv_path = METRICS_DIR / "cross_validation_results.csv"
        tuned_path = METRICS_DIR / "tuned_results.csv"
        test_path = METRICS_DIR / "final_test_results.csv"

        if test_path.exists():
            st.markdown("#### 1. Untouched Test Set Benchmark (Final Generalization)")
            df_test = pd.read_csv(test_path)
            st.dataframe(df_test, use_container_width=True)

        if cv_path.exists():
            st.markdown("#### 2. 5-Fold Cross-Validation Performance (Stability Analysis)")
            df_cv = pd.read_csv(cv_path)
            st.dataframe(df_cv, use_container_width=True)

        if tuned_path.exists():
            st.markdown("#### 3. Hyperparameter Tuning Impact (Before vs After)")
            df_tuned = pd.read_csv(tuned_path)
            st.dataframe(df_tuned, use_container_width=True)

        # Visualizations
        st.markdown("#### 4. Diagnostic Visualizations")
        viz_col1, viz_col2 = st.columns(2)

        p_comparison = PLOTS_DIR / "model_comparison.png"
        p_importance = PLOTS_DIR / "feature_importance.png"
        p_act_pred = PLOTS_DIR / "actual_vs_predicted.png"
        p_residuals = PLOTS_DIR / "residual_vs_predicted.png"

        with viz_col1:
            if p_comparison.exists():
                st.image(str(p_comparison), caption="5-Fold Cross-Validation RMSE Comparison")
            if p_act_pred.exists():
                st.image(str(p_act_pred), caption="Actual vs Predicted Prices")

        with viz_col2:
            if p_importance.exists():
                st.image(str(p_importance), caption="Key Feature Importances / Weights")
            if p_residuals.exists():
                st.image(str(p_residuals), caption="Residual Diagnostics")

    # Tab 3: About
    else:
        st.subheader("About This Machine Learning System")
        st.markdown(
            """
            ### Project Overview
            This project provides a comprehensive, scientifically rigorous comparative evaluation of 5 regression algorithms:
            1. **Linear Regression** (Baseline reference)
            2. **Ridge Regression** (L2 Regularized linear model)
            3. **Decision Tree Regression** (Non-linear recursive partitioning)
            4. **Random Forest Regression** (Bagging ensemble)
            5. **Gradient Boosting Regression** (Boosting ensemble)

            ### Experimental Methodology
            - **No Data Leakage**: All imputers, encoders, and scalers are fitted exclusively on training splits via scikit-learn `Pipeline` and `ColumnTransformer`.
            - **5-Fold Cross-Validation**: Measures both average accuracy and model stability across folds.
            - **Hyperparameter Optimization**: Systematically tunes core parameters using cross-validated search.
            - **Multi-Metric Evaluation**: Compares MAE, MSE, RMSE, and R² to prevent misleading single-metric conclusions.
            - **Residual Error Diagnostics**: Analyzes error distribution and heteroscedasticity across price tiers.
            """
        )


if __name__ == "__main__":
    main()
