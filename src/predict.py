import pandas as pd
import numpy as np
import joblib
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
FEATURES_PATH = "data/processed/features.parquet"
TARGET_PATH = "data/processed/target.parquet"
MODELS_DIR = "models/"

def make_predictions():
    """Loads trained models and makes predictions."""
    logging.info("Starting predictions.")

    try:
        # Load data
        X = pd.read_parquet(FEATURES_PATH)
        y = pd.read_parquet(TARGET_PATH) # In a real scenario, you might not have the target

        # Handle potential infinity values from feature engineering
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Simple imputation - you might want to use a more sophisticated approach
        X.fillna(X.median(), inplace=True)


        # Load models and make predictions
        models = ["LinearRegression", "Ridge", "Lasso", "DecisionTree", "RandomForest"]

        for name in models:
            logging.info(f"Making predictions with {name}...")
            model = joblib.load(f"{MODELS_DIR}{name}.joblib")
            predictions = model.predict(X.head()) # Predicting on the first 5 rows as an example

            logging.info(f"Predictions for {name}:\n{predictions}")

        logging.info("Predictions completed successfully.")

    except Exception as e:
        logging.error(f"Prediction failed: {e}")

if __name__ == "__main__":
    make_predictions()
