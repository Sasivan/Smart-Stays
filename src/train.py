import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
import logging
from pathlib import Path

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Models
from sklearn.linear_model import Ridge
import lightgbm as lgb

# --- Configuration & Setup ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
PROCESSED_PATH = Path("data/processed/listings_featured.parquet")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
# Path to save category info
CATEGORY_INFO_PATH = MODELS_DIR / "categorical_feature_info.json"

# --- Helper Function ---

def evaluate_model(model_name, model, X_test, y_test_log):
    """
    Evaluates a model's regression performance.
    Converts log-predictions back to original scale.
    """
    logging.info(f"--- Evaluating {model_name} ---")
    y_pred_log = model.predict(X_test)
    y_test_real = np.expm1(y_test_log)
    y_pred_real = np.expm1(y_pred_log)
    y_pred_real[y_pred_real < 0] = 0  # Set any negative prices to 0 after expm1
    mae = mean_absolute_error(y_test_real, y_pred_real)
    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
    r2 = r2_score(y_test_real, y_pred_real)
    logging.info(f"MAE (Mean Absolute Error): ${mae:,.2f}")
    logging.info(f"RMSE (Root Mean Squared Error): ${rmse:,.2f}")
    logging.info(f"R-squared (R²): {r2:.3f}")
    return {'Model': model_name, 'MAE': mae, 'RMSE': rmse, 'R2': r2}

# --- Main Training Pipeline ---

def main():
    logging.info("--- Starting Model Training Pipeline ---")
    try:
        # Load data
        df = pd.read_parquet(PROCESSED_PATH)
        logging.info(f"Loaded featured data: {df.shape}")
    except FileNotFoundError:
        logging.error(f"FATAL: File not found at {PROCESSED_PATH}")
        logging.error("Please run the src/features.py script first.")
        return

    logging.info("Preparing data...")
    # Log-transform target variable
    df['price_log'] = np.log1p(df['price'])
    TARGET = 'price_log'
    COLS_TO_DROP = ['price', 'price_log', 'id', 'host_id']
    LEAKAGE_COLS = ['price_per_person', 'effective_price', 'cleaning_fee']
    
    # Drop known leakage and identifier columns
    df = df.drop(columns=LEAKAGE_COLS, errors='ignore')
    logging.info(f"Dropped leakage columns: {LEAKAGE_COLS}")
    
    # Clean data
    df_cleaned = df.dropna(subset=[TARGET])
    y = df_cleaned[TARGET]
    X = df_cleaned.drop(columns=COLS_TO_DROP, errors='ignore')
    logging.info(f"Shape after dropping NaN targets: {X.shape}")

    # Define categorical features
    categorical_features_names = ['neighbourhood_cleansed', 'property_type', 'room_type', 'spatial_cluster']
    categorical_features = [col for col in categorical_features_names if col in X.columns]
    logging.info(f"Found {len(categorical_features)} categorical features to convert: {categorical_features}")

    # --- Robust categorical conversion and cleaning ---
    valid_categorical_features = []
    for col in categorical_features:
        if col in X.columns:
            # 1. Convert to string, filling any lingering NaNs with a placeholder
            X[col] = X[col].astype(str).fillna("Unknown")
            
            # 2. Convert to category dtype
            X[col] = X[col].astype('category')

            # 3. Add 'Missing' category for robust handling in OHE/LGBM
            if "Missing" not in X[col].cat.categories:
                X[col] = X[col].cat.add_categories("Missing")
            
            # Replace placeholder 'Unknown' with 'Missing'
            X[col] = X[col].replace("Unknown", "Missing").fillna("Missing") 
            
            valid_categorical_features.append(col)
        else:
             logging.warning(f"Categorical feature '{col}' not found in X, skipping.")
    
    categorical_features = valid_categorical_features # Use the valid list
    # --- END FIX ---

    # Ensure numeric_features list excludes categorical_features
    numeric_features_all = X.select_dtypes(include=np.number).columns.tolist()
    numeric_features = [col for col in numeric_features_all if col not in categorical_features]
    logging.info(f"Found {len(numeric_features)} numeric features.")

    # Filter features in X
    all_features_to_keep = numeric_features + categorical_features
    X = X[all_features_to_keep]
    logging.info(f"Filtered X to keep only {len(all_features_to_keep)} numeric/category columns.")

    # Final cleanup of numeric features (after filtering)
    X[numeric_features] = X[numeric_features].fillna(0)
    X[numeric_features] = X[numeric_features].replace([np.inf, -np.inf], 0)

    # --- Save category info ---
    category_info = {}
    for col in categorical_features:
        # Re-ensure 'Missing' category exists before saving final list
        if "Missing" not in X[col].cat.categories:
             X[col] = X[col].cat.add_categories("Missing")
        category_info[col] = X[col].cat.categories.tolist()

    MODELS_DIR.mkdir(exist_ok=True)
    try:
        with open(CATEGORY_INFO_PATH, 'w') as f:
            json.dump(category_info, f, indent=4)
        logging.info(f"Saved category information for {len(category_info)} features to {CATEGORY_INFO_PATH}")
    except Exception as e:
         logging.error(f"Failed to save category info: {e}")

    logging.info("Cleaned all NaN/Inf values from features (X).")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")

    # --- Train Ridge (Linear Model) ---
    logging.info("Training Ridge model...")
    # Preprocessing for Ridge: Scaling numeric, One-Hot Encoding categorical
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    # handle_unknown='ignore' prevents error if a new category appears in test set
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]) 
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ], 
        remainder='drop' # Drop all other columns not specified
    )
    ridge_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', Ridge(alpha=1.0, random_state=42))
    ])
    ridge_pipeline.fit(X_train, y_train)
    logging.info("Ridge model training complete.")

    # --- Train LightGBM (Gradient Boosting) ---
    # Note: LightGBM natively handles categorical features
    logging.info("Training LightGBM model...")
    lgbm_model = lgb.LGBMRegressor(random_state=42, n_estimators=200, learning_rate=0.1)
    lgbm_model.fit(
        X_train, 
        y_train, 
        eval_set=[(X_test, y_test)], 
        eval_metric='rmse', 
        callbacks=[lgb.early_stopping(50)], 
        categorical_feature=categorical_features
    )
    logging.info("LightGBM model training complete.")

    # --- Evaluation ---
    logging.info("Evaluating models...")
    results = []
    results.append(evaluate_model('Ridge Regression', ridge_pipeline, X_test, y_test))
    results.append(evaluate_model('LightGBM (Baseline)', lgbm_model, X_test, y_test))
    results_df = pd.DataFrame(results)
    logging.info(f"Model Comparison:\n{results_df.set_index('Model').sort_values('MAE')}")

    # --- Save Artifacts ---
    logging.info("Saving model artifacts...")
    REPORTS_DIR.mkdir(exist_ok=True)
    
    # Save trained models
    joblib.dump(ridge_pipeline, MODELS_DIR / "ridge_baseline_model.joblib")
    joblib.dump(lgbm_model, MODELS_DIR / "lgbm_baseline_model.joblib")
    logging.info(f"Saved models to {MODELS_DIR}/")
    
    # Save evaluation scores
    scores_path = REPORTS_DIR / "model_scores.json"
    results_df.to_json(scores_path, orient='records', indent=4)
    logging.info(f"Saved model scores to {scores_path}")
    
    # Save LightGBM feature importance plot
    if hasattr(lgbm_model, 'feature_importances_'):
        logging.info("Saving LightGBM feature importance plot...")
        lgb.plot_importance(lgbm_model, max_num_features=25, figsize=(10, 12))
        plot_path = REPORTS_DIR / "lgbm_feature_importance.png"
        plt.title('LightGBM Feature Importance'); plt.tight_layout(); plt.savefig(plot_path); plt.close()
        logging.info(f"Saved plot to {plot_path}")

    logging.info("--- Model Training Pipeline Complete ---")

if __name__ == "__main__":
    main()