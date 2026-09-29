# 🏡 King County House Price Prediction

An end-to-end Machine Learning regression project for predicting residential house sales prices in King County, USA (including Seattle). This repository includes exploratory data analysis, robust preprocessing pipelines, feature engineering, baseline vs tuned model benchmarking, an automated training pipeline, unit tests, and an interactive Streamlit web dashboard.

---

## 📌 Project Overview

- **Dataset**: King County House Sales dataset containing structural, geographical, and time-based features.
- **Target Variable**: Continuous house sale price in USD (`price`).
- **Best Model**: Tuned Ensemble Regressor (Random Forest / Gradient Boosting) optimizing $R^2$, RMSE, and MAE metrics.
- **Deployment**: Real-time Streamlit web application for interactive valuation and model diagnostic reporting.

---

## 📂 Repository Structure

```
house-price-regression/
├── app/
│   └── app.py                     # Interactive Streamlit Web Application
├── data/
│   ├── raw/
│   │   └── house_sales.csv        # Raw dataset
│   └── processed/
│       ├── train.csv              # Processed train split
│       └── test.csv               # Processed test split
├── models/
│   └── final_model.joblib         # Serialized production pipeline & model
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_model_comparison.ipynb
│   ├── 05_hyperparameter_tuning.ipynb
│   └── 06_error_analysis.ipynb
├── results/
│   ├── metrics/                   # CSV benchmark results (baseline, CV, tuned, test)
│   ├── plots/                     # Diagnostic plots (actual vs pred, residuals, importance)
│   └── predictions/               # Test set predictions
├── src/
│   ├── __init__.py
│   ├── config.py                  # Global configurations & hyperparameters
│   ├── data_loader.py             # Data ingestion and splitting
│   ├── preprocessing.py          # Data cleansing and missing value handling
│   ├── feature_engineering.py     # Domain-specific feature transformations
│   ├── models.py                  # Model architecture definitions
│   ├── tuning.py                  # GridSearchCV & RandomizedSearchCV tuning
│   ├── evaluation.py              # Performance evaluation & metric reporting
│   └── prediction.py              # Inference interface for single and batch predictions
├── tests/
│   ├── test_preprocessing.py      # Unit tests for preprocessing & transformations
│   └── test_prediction.py         # Unit tests for model inference
├── main_pipeline.py               # Automated end-to-end training & evaluation pipeline
├── pyproject.toml                 # Project metadata and tool configuration
├── requirements.txt               # Environment dependencies
└── README.md                      # Comprehensive project documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/puvanakopis/House-Price-Prediction.git
cd House-Price-Prediction
```

### 2. Set Up Virtual Environment & Dependencies
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Run the End-to-End Training Pipeline
To run the full pipeline (data loading, preprocessing, model training, tuning, evaluation, and saving artifacts):
```bash
python main_pipeline.py
```

### 4. Launch the Interactive Web Application
```bash
streamlit run app/app.py
```

---

## 🧪 Testing

Execute automated unit tests with `pytest`:
```bash
pytest tests/
```

---

## 📊 Key Results & Diagnostics

- **Metrics**: Evaluated using $R^2$, Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).
- **Diagnostics**: Includes target distribution plots, actual vs. predicted scatter comparisons, residual analysis, and feature importance rankings saved under `results/plots/`.

---

## 📜 License
This project is open source and available under the [MIT License](LICENSE).
