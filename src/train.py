import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
import joblib
import logging
import time
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
FEATURES_PATH = "data/processed/features.parquet"
TARGET_PATH = "data/processed/target.parquet"
MODELS_DIR = "models/"
EXPERIMENT_LOG_PATH = "reports/experiment_log.csv"

def train_models():
    """Trains baseline models and saves them."""
    logging.info("Starting model training.")

    try:
        # Load data
        X = pd.read_parquet(FEATURES_PATH)
        y = pd.read_parquet(TARGET_PATH)

        # Handle potential infinity values from feature engineering
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Simple imputation - you might want to use a more sophisticated approach
        X.fillna(X.median(), inplace=True)

        # Time-based split (assuming 'month' feature can be used for this)
        # For a more robust approach, use the 'last_review' date before feature engineering
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Define models
        models = {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(),
            "Lasso": Lasso(),
            "DecisionTree": DecisionTreeRegressor(),
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42)
        }

        # Train and evaluate models
        for name, model in models.items():
            logging.info(f"Training {name}...")
            start_time = time.time()
            model.fit(X_train, y_train.values.ravel())
            training_time = time.time() - start_time

            # Make predictions
            y_pred = model.predict(X_test)

            # Evaluate
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            median_ae = median_absolute_error(y_test, y_pred)
            logging.info(f"{name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, MedianAE: {median_ae:.4f}")

            # Save model
            model_path = f"{MODELS_DIR}{name}.joblib"
            joblib.dump(model, model_path)
            model_size = os.path.getsize(model_path) / 1024 # in KB
            logging.info(f"Saved {name} to {model_path}")

            # Log experiment
            with open(EXPERIMENT_LOG_PATH, "a") as f:
                f.write(f"{name},{mae:.4f},{rmse:.4f},{median_ae:.4f},{training_time:.2f},{model_size:.2f},Baseline model\n")

        logging.info("Model training completed successfully.")

    except Exception as e:
        logging.error(f"Model training failed: {e}")

if __name__ == "__main__":
    train_models()
