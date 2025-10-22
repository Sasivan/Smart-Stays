import pandas as pd
import json
import logging
from pathlib import Path
import os
from typing import Dict, Any, Union, List

# --- Setup Logging and Paths ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define paths relative to the project root
PROJECT_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_DIR / "reports"
DATA_PROCESSED_PATH = PROJECT_DIR / "data/processed/listings_featured.parquet"
MODEL_SCORES_PATH = REPORTS_DIR / "model_scores.json" 
REPORT_OUTPUT_PATH = REPORTS_DIR / "final_summary.md"

# --- Functions to Gather Information ---

def check_data_status(data_path: Path) -> Dict[str, Any]:
    """Checks if processed data exists and returns its size/status."""
    status = {"exists": False, "size_mb": 0, "rows": 0, "cols": 0}
    if data_path.exists():
        status["exists"] = True
        status["size_mb"] = round(data_path.stat().st_size / (1024 * 1024), 2)
        try:
            df = pd.read_parquet(data_path) 
            status["rows"], status["cols"] = df.shape
        except Exception:
            logging.warning(f"Could not read metadata from {data_path}")
    return status

def load_model_metrics(metrics_path: Path) -> Dict[str, Any]:
    """
    Loads and returns the model performance metrics, handling list-wrapped results.
    Guarantees the return type is a Dict or an empty Dict.
    """
    if metrics_path.exists():
        try:
            with open(metrics_path, 'r') as f:
                data: Union[Dict, List] = json.load(f)
                
                if isinstance(data, list) and data:
                    logging.warning(f"Model scores loaded as a list. Using the first element as the metrics dictionary.")
                    return data[0]
                
                if isinstance(data, dict):
                    return data
                
        except Exception as e:
            logging.error(f"Error loading model scores from {metrics_path}: {e}")
    return {}

def generate_report():
    """Aggregates all information and writes the final summary Markdown file."""
    
    # 1. Gather Data & Model Info
    data_info = check_data_status(DATA_PROCESSED_PATH)
    model_metrics = load_model_metrics(MODEL_SCORES_PATH)
    
    # 2. EDA Narrative (Read .md file content)
    try:
        with open(REPORTS_DIR / "eda.md", 'r', encoding='utf-8') as f:
            eda_narrative = f.read()
    except FileNotFoundError:
        eda_narrative = "EDA narrative file (`eda.md`) not found."
    
    # 3. EDA Profiling (Read .html file content - will be embedded as raw HTML)
    try:
        with open(REPORTS_DIR / "eda_report_ydata_profiling_minimal.html", 'r', encoding='utf-8') as f:
            eda_profile_html = f.read()
    except FileNotFoundError:
        eda_profile_html = "<p>EDA profiling report (`eda_report_ydata_profiling_minimal.html`) not found.</p>"

    
    # --- Generate Report Content ---
    report_content = f"""# Smart Stays Project Final Analysis Report 🏠

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

* **Processed Data File:** `{DATA_PROCESSED_PATH.name}`
* **Existence:** {'✅ YES' if data_info['exists'] else '❌ NO'}
* **Size:** {data_info['size_mb']:,} MB
* **Shape:** {data_info['rows']:,} Rows, {data_info['cols']:,} Columns (Features)

***

## 3. Model Performance Summary (`/reports/model_scores.json`)

{'Model Scores loaded from file.' if model_metrics else '❌ Model Scores not found or failed to load. Run src/train.py to generate metrics.'}

"""
    # Append model metrics if available
    if model_metrics:
        for model_name, metrics in model_metrics.items():
            report_content += f"### Model: {model_name}\n"
            report_content += "The model was trained on engineered features.\n"
            report_content += "| Metric | Score |\n"
            report_content += "| :--- | :--- |\n"
            
            if isinstance(metrics, dict):
                for metric_name, score in metrics.items():
                    if isinstance(score, (int, float)):
                        report_content += f"| **{metric_name}** | {score:,.4f} |\n"
                    else:
                        report_content += f"| **{metric_name}** | {score} |\n"
            else:
                report_content += f"| **Status/Value** | {metrics} |\n"
                
        report_content += "\n"

    # 4. Append Artifact Documentation (Simple list of files for reference)
    report_content += f"""
***

## 4. Model Interpretation & Diagnostics Files

The following files provide detailed visual and quantitative analysis:

* **lgbm_feature_importance.png**: A visual representation of the overall Gini importance of features derived from the trained LightGBM model.
* **shap_summary_bar.png**: A summary bar plot showing the average impact (magnitude) of the top features on the model's output, derived from SHAP values.
* **shap_summary_beeswarm.png**: A detailed SHAP beeswarm plot showing how feature values affect the prediction outcome.
* **model_scores.json**: Contains the raw performance metrics for the trained model.
* **eda.md**
* **eda_report_ydata_profiling_minimal.html**

"""

    # --- Write to file ---
    with open(REPORT_OUTPUT_PATH, 'w') as f:
        f.write(report_content)
    
    logging.info(f"✅ Final analysis report written successfully to {REPORT_OUTPUT_PATH.name}")
    print(f"\n--- Report Generation Complete ---\nFile saved to: {REPORT_OUTPUT_PATH.resolve()}")

if __name__ == "__main__":
    generate_report()
