# Smart Stays Project Final Analysis Report 🏠

## 1. Project Workflow Status

| File/Notebook | Description | Status |
| :--- | :--- | :--- |
| `src/preprocess.py` | Clean and preprocess data | ✅ Implemented |
| `src/features.py` | Engineer model-ready features | ✅ Implemented |
| `src/train.py` | Train ML models | ✅ Implemented |
| `src/explain.py` | Interpret trained models (SHAP/Importance) | ✅ Implemented |
| `src/predict.py` | Make predictions | ✅ Implemented |
| `notebooks/01_eda.ipynb` | Explore data trends | ✅ Completed |
| `notebooks/02_feature_engineering.ipynb` | Prototype feature ideas | ✅ Completed |
| `notebooks/03_modeling.ipynb` | Experiment with algorithms | ✅ Completed |
| `dashboards/streamlit_app.py` | Visualize results & predictions | ✅ Implemented |

***

## 2. Data Summary (`/data/processed`)

* **Processed Data File:** `listings_featured.parquet`
* **Existence:** ✅ YES
* **Size:** 2.98 MB
* **Shape:** 15,187 Rows, 100 Columns (Features)

***

## 3. Model Performance Summary (`/reports/model_scores.json`)

Model Scores loaded from file.

### Model: Model
The model was trained on engineered features.
| Metric | Score |
| :--- | :--- |
| **Status/Value** | Ridge Regression |
### Model: MAE
The model was trained on engineered features.
| Metric | Score |
| :--- | :--- |
| **Status/Value** | 1037.9694298491 |
### Model: RMSE
The model was trained on engineered features.
| Metric | Score |
| :--- | :--- |
| **Status/Value** | 35974.3766004453 |
### Model: R2
The model was trained on engineered features.
| Metric | Score |
| :--- | :--- |
| **Status/Value** | -17126.1055164806 |


***

## 4. Model Interpretation & Diagnostics Files

The following files provide detailed visual and quantitative analysis:

* **lgbm_feature_importance.png**: A visual representation of the overall Gini importance of features derived from the trained LightGBM model.
* **shap_summary_bar.png**: A summary bar plot showing the average impact (magnitude) of the top features on the model's output, derived from SHAP values.
* **shap_summary_beeswarm.png**: A detailed SHAP beeswarm plot showing how feature values affect the prediction outcome.
* **model_scores.json**: Contains the raw performance metrics for the trained model.
* **eda.md**
* **eda_report_ydata_profiling_minimal.html**

