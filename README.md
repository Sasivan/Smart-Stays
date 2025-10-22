
# End-to-End Machine Learning Airbnb Project

This repository contains a complete, end-to-end machine learning project. It demonstrates a full workflow from raw data ingestion and preprocessing to feature engineering, model training (Ridge & LightGBM), evaluation, and explainability (SHAP). The final results and model predictions are served through an interactive Streamlit dashboard.

## Features

  * **Modular Pipeline:** All steps are organized into distinct Python scripts within the `src/` directory (preprocessing, feature engineering, training, etc.).
  * **Reproducibility:** A single `run_all.sh` script executes the entire data processing and model training pipeline from start to finish.
  * **Model Comparison:** Trains and evaluates both a simple baseline (Ridge Regression) and a more complex gradient boosting model (LightGBM).
  * **Model Explainability:** Generates feature importance and SHAP summary plots to understand what drives model predictions.
  * **Interactive Dashboard:** A Streamlit application provides a user-friendly interface to view project summaries, reports, and (presumably) make new predictions.

## Setup and Installation

1.  **Clone the repository:**

    ```bash
    https://github.com/Sasivan/Smart-Stays.git
    cd your-repository-name
    ```

2.  **Create a virtual environment** (Recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Add Raw Data:**
    This project requires the raw data file to be added manually.

      * **Place your `listings.csv.gz` file inside the `data/raw/` directory.**

## How to Run

There are two main parts to this project: running the ML pipeline and launching the dashboard.

### 1\. Run the ML Pipeline

This step executes all data processing and model training scripts. It will populate the `data/processed/`, `models/`, and `reports/` directories with new artifacts.

```bash
bash scripts/run_all.sh
```

This master script will likely execute the following in order:

1.  `src/preprocess.py` - Cleans raw data.
2.  `src/features.py` - Creates new features.
3.  `src/train.py` - Trains models and saves them.
4.  `src/explain.py` - Generates SHAP plots for the trained model.

### 2\. Launch the Streamlit Dashboard

After the pipeline has run and the models are saved, you can launch the interactive web application.

```bash
streamlit run dashboards/streamlit_app.py
```

Open your web browser to the local URL provided (e.g., `http://localhost:8501`).

## Project Structure

```
├── dashboards/                  # Contains the Streamlit app and rendered notebooks
│   ├── 01_eda_rendered.html
│   ├── 02_feature_engineering_rendered.html
│   ├── 03_modeling_rendered.html
│   └── streamlit_app.py        # Main Streamlit application script
├── data/
│   ├── processed/              # Cleaned and feature-engineered data
│   │   ├── data_quality_report.txt
│   │   ├── listings.csv
│   │   ├── listings.parquet
│   │   └── listings_featured.parquet # Final data used for training
│   └── raw/                    # Raw input data
│       └── listings.csv.gz     # <<< MUST BE ADDED MANUALLY >>>
├── models/                     # Trained models and related artifacts
│   ├── categorical_feature_info.json # Info about categorical features for prediction
│   ├── lgbm_baseline_model.joblib    # Saved LightGBM model
│   └── ridge_baseline_model.joblib   # Saved Ridge model
├── notebooks/                  # Jupyter notebooks for exploration and development
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
├── reports/                    # Generated reports, summaries, and plots
│   ├── eda.md
│   ├── eda_report_ydata_profiling_minimal.html # Automated EDA report
│   ├── final_summary.md        # Main summary report shown in Streamlit
│   ├── lgbm_feature_importance.png
│   ├── model_scores.json       # Performance metrics for trained models
│   ├── report_analysis.py
│   ├── shap_summary_bar.png
│   └── shap_summary_beeswarm.png
├── requirements.txt            # Python package dependencies
├── scripts/                    # Utility scripts
│   └── run_all.sh              # Master script to execute the ML pipeline
└── src/                        # Source code for the pipeline steps
    ├── explain.py              # Generates SHAP explanation plots
    ├── features.py             # Performs feature engineering
    ├── plotting.py             # Contains plotting functions for the dashboard
    ├── predict.py              # Core logic for making predictions on new data
    ├── preprocess.py           # Initial data loading and cleaning
    └── train.py                # Model training and evaluation
```
