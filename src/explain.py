import pandas as pd
import numpy as np
import joblib
import shap
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split # Needed to re-create test set

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define file paths
PROCESSED_PATH = Path("../data/processed/listings_featured.parquet")
MODEL_PATH = Path("../models/lgbm_baseline_model.joblib")
REPORTS_DIR = Path("../reports")

# --- Main Explainability Function ---

def explain_model():
    logging.info("--- Starting Model Explainability Script ---")

    # --- 1. Load Model and Data ---
    try:
        model = joblib.load(MODEL_PATH)
        logging.info(f"Loaded model from {MODEL_PATH}")
    except FileNotFoundError:
        logging.error(f"FATAL: Model not found at {MODEL_PATH}")
        logging.error("Please run the src/train.py script first.")
        return

    try:
        df = pd.read_parquet(PROCESSED_PATH)
        logging.info(f"Loaded featured data from {PROCESSED_PATH}")
    except FileNotFoundError:
        logging.error(f"FATAL: Featured data not found at {PROCESSED_PATH}")
        return

    # --- 2. Re-create the Test Set ---
    # We must apply the *exact same* data preparation as in train.py
    
    logging.info("Re-creating test data...")
    
    # Drop leakage columns
    LEAKAGE_COLS = ['price_per_person', 'effective_price', 'cleaning_fee'] 
    df = df.drop(columns=LEAKAGE_COLS, errors='ignore')
    
    # Define and drop target
    df['price_log'] = np.log1p(df['price'])
    df_cleaned = df.dropna(subset=['price_log'])
    
    COLS_TO_DROP = ['price', 'price_log', 'id', 'host_id']
    y = df_cleaned['price_log']
    X = df_cleaned.drop(columns=COLS_TO_DROP, errors='ignore')

    # Drop duplicate columns
    X = X.loc[:, ~X.columns.duplicated()]
    logging.info(f"Shape after dropping duplicate columns: {X.shape}")

    # Define feature lists (must match train.py)
    categorical_features_names = [
        'neighbourhood_cleansed', 
        'property_type', 
        'room_type', 
        'spatial_cluster'
    ]
    categorical_features = [col for col in categorical_features_names if col in X.columns]
    
    all_numeric_cols = X.select_dtypes(include=np.number).columns
    numeric_features = [col for col in all_numeric_cols if col not in categorical_features]
    
    logging.info(f"Found {len(categorical_features)} categorical features: {categorical_features}")
    logging.info(f"Found {len(numeric_features)} purely numeric features.")
    
    all_features_to_keep = numeric_features + categorical_features
    X = X[all_features_to_keep] 

    # Clean X features
    logging.info("Cleaning X features...")
    X.loc[:, numeric_features] = X.loc[:, numeric_features].fillna(0)
    X.loc[:, numeric_features] = X.loc[:, numeric_features].replace([np.inf, -np.inf], 0)
    
    # --- FIX: Re-ordered logic ---
    # 1. Convert to category
    # 2. Add "Missing" category
    # 3. Fill NaNs
    for col in categorical_features:
        X[col] = X[col].astype('category')
        
        if "Missing" not in X[col].cat.categories:
            X[col] = X[col].cat.add_categories("Missing")
            
        X[col] = X[col].fillna("Missing")
    # --- END FIX ---
        
    # Split data with the same random_state to get the *exact* same X_test
    _, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    
    logging.info(f"Successfully prepared X_test with shape {X_test.shape}")

    # --- 3. Calculate SHAP Values ---
    logging.info("Calculating SHAP values... (This may take a moment)")
    
    # Use shap.TreeExplainer for tree-based models like LightGBM
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    logging.info("SHAP values calculated.")

    # --- 4. Save SHAP Summary Plots ---
    REPORTS_DIR.mkdir(exist_ok=True)
    
    # Bar plot (global feature importance)
    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title("SHAP Global Feature Importance (Bar)")
    plt.tight_layout()
    bar_plot_path = REPORTS_DIR / "shap_summary_bar.png"
    plt.savefig(bar_plot_path)
    plt.close()
    logging.info(f"Saved SHAP bar plot to {bar_plot_path}")

    # Summary plot (beeswarm)
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Summary Plot (Beeswarm)")
    plt.tight_layout()
    swarm_plot_path = REPORTS_DIR / "shap_summary_beeswarm.png"
    plt.savefig(swarm_plot_path)
    plt.close()
    logging.info(f"Saved SHAP beeswarm plot to {swarm_plot_path}")
    
    logging.info("--- Model Explainability Script Complete ---")


if __name__ == "__main__":
    explain_model()