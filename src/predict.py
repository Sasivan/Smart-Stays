import pandas as pd
import numpy as np
import joblib
import json 
import logging
from pathlib import Path

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define file paths (assuming this script is in 'src/')
# These paths are relative to the project root
MODEL_PATH = Path("models/lgbm_baseline_model.joblib")
CATEGORY_PATH = Path("models/categorical_feature_info.json") 
PROCESSED_PATH = Path("data/processed/listings_featured.parquet")

# --- Load Model and Categories ---

def load_model_and_categories():
    """Loads the trained model and associated category mapping from disk."""
    model = None
    category_info = {}
    
    try:
        model = joblib.load(MODEL_PATH)
        logging.info(f"Loaded model from {MODEL_PATH}")
    except FileNotFoundError:
        logging.error(f"FATAL: Model not found at {MODEL_PATH}. Run 'train.py' first.")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        
    try:
        if CATEGORY_PATH.exists():
              with open(CATEGORY_PATH, 'r') as f:
                  category_info = json.load(f)
              logging.info(f"Loaded category info from {CATEGORY_PATH}")
        else:
              logging.warning(f"Category info not found at {CATEGORY_PATH}. Predictions may fail.")
              
    except Exception as e:
        logging.error(f"Error loading category info: {e}")
        
    return model, category_info

# --- Prediction Function ---

def predict_price(featured_data: pd.DataFrame, model, category_info: dict):
    """
    Predicts the price for new listing data, enforcing categorical and numeric alignment.
    This is the core function to prevent 'train and valid dataset' errors.
    """
    if model is None:
        logging.error("Model is not loaded. Cannot make predictions.")
        return None

    logging.info(f"Received {len(featured_data)} rows for prediction.")
    X = featured_data.copy()
    
    try:
        model_features = model.feature_name_
    except AttributeError:
        logging.error("Model object is invalid or does not have 'feature_name_'.")
        return None

    # --- 1. Align Columns ---
    # Ensure all columns the model was trained on are present
    for col in model_features:
        if col not in X.columns:
            if col in category_info:
                X[col] = "Missing" # Default for missing categorical
            else:
                X[col] = 0.0 # Default for missing numeric
    
    # Enforce the exact column order the model was trained on
    X = X[model_features]

    # --- 2. Define Feature Lists ---
    categorical_features = [col for col in model_features if col in category_info]
    numeric_features = [col for col in model_features if col not in categorical_features]
    
    # --- 3. Robust Data Cleaning & Type Enforcement ---
    
    # Helper function to robustly clean numeric values
    def clean_numeric_value(val):
        # If it's a list or tuple, try to get the first element
        if isinstance(val, (list, tuple)):
            val = val[0] if len(val) > 0 else None
        
        # **FIX**: If it's a string that LOOKS like a list, e.g., "['4.0']"
        if isinstance(val, str):
            val = val.strip()
            if val.startswith('[') and val.endswith(']'):
                # Extract the content, remove quotes
                val = val.strip("[]'\" ") 

        # Now, val is hopefully a scalar (str, int, float) or None
        # 'coerce' will turn any invalid text into NaN
        return pd.to_numeric(val, errors='coerce')

    # **FIX PART A: Clean Numeric Features FIRST**
    # This prevents 'incompatible dtype' warnings by forcing columns to be numeric.
    for col in numeric_features:
        # Use .apply() with the more robust helper function
        X[col] = X[col].apply(clean_numeric_value)
        # Fill any NaNs (from coercion or originals) with 0.0
        X[col] = X[col].fillna(0.0)

    # **FIX PART B: Clean and Align Categorical Features SECOND**
    for col in categorical_features:
        known_cats = category_info.get(col, [])
        if "Missing" not in known_cats:
            known_cats.append("Missing")
        
        # 1. Ensure column is string and fill any NaNs with "Missing"
        X[col] = X[col].astype(str).fillna("Missing")
        
        # 2. Convert to pd.Categorical using ONLY the 'known_cats' from training.
        # This is the key: any category not in 'known_cats' becomes NaN.
        X[col] = pd.Categorical(X[col], categories=known_cats, ordered=False)
        
        # 3. Fill the NaNs created in step 2 with the "Missing" category.
        X[col] = X[col].fillna("Missing") 
        
    # --- 4. Make Prediction ---
    try:
        log_predictions = model.predict(X)
        
        # Inverse the log-transform (log1p -> expm1)
        predictions = np.expm1(log_predictions)
        predictions[predictions < 0] = 0 # Safety check for no negative prices
        
        logging.info(f"Successfully made {len(predictions)} predictions.")
        return predictions
        
    except Exception as e:
        # This will catch the 'train and valid dataset categorical_feature do not match' error
        logging.error(f"Error during prediction: {e}", exc_info=True) 
        return None

# --- Test Block ---
if __name__ == "__main__":
    """
    Run this script directly (python src/predict.py) to test
    if the model and categories load correctly and can make a prediction.
    """
    logging.info("--- Running predict.py in test mode ---")
    
    model, category_info = load_model_and_categories()
    
    if model and category_info:
        try:
            # Load a sample of 5 rows from the *original* featured data to test
            sample_df_all = pd.read_parquet(PROCESSED_PATH).sample(5, random_state=42)
            actual_prices = sample_df_all['price'].values
            
            # Pass the sample to the prediction function
            predictions = predict_price(sample_df_all, model, category_info) 
            
            if predictions is not None:
                print("\n--- Test Results ---")
                print("Predictions vs. Actual")
                for pred, actual in zip(predictions, actual_prices):
                    print(f"Predicted: ${pred:,.2f}  |  Actual: ${actual:,.2f}")
                print("--------------------\n")
            else:
                print("Prediction function returned None.")
                
        except FileNotFoundError:
            logging.error(f"Could not load sample data from {PROCESSED_PATH}")
        except Exception as e:
            logging.error(f"An error occurred during testing: {e}", exc_info=True)
    else:
        logging.error("Could not load model or category info. Test failed.")
            
    logging.info("--- Test mode complete ---")

