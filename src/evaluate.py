import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
FEATURES_PATH = "data/processed/features.parquet"
TARGET_PATH = "data/processed/target.parquet"
MODELS_DIR = "models/"
REPORTS_DIR = "reports/"

def evaluate_model(model_name):
    """Evaluates a trained model and generates error analysis plots."""
    logging.info(f"Evaluating {model_name}...")

    try:
        # Load data and model
        X = pd.read_parquet(FEATURES_PATH)
        y = pd.read_parquet(TARGET_PATH)

        # Handle potential infinity values from feature engineering
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Simple imputation - you might want to use a more sophisticated approach
        X.fillna(X.median(), inplace=True)

        model = joblib.load(f"{MODELS_DIR}{model_name}.joblib")

        # Make predictions
        y_pred = model.predict(X)

        # Calculate residuals
        residuals = y.values.ravel() - y_pred

        # Create residual plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_pred, y=residuals)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title(f"Residual Plot for {model_name}")
        plt.savefig(f"{REPORTS_DIR}{model_name}_residual_plot.png")
        plt.close()

        logging.info(f"Generated residual plot for {model_name}")

    except Exception as e:
        logging.error(f"Evaluation failed for {model_name}: {e}")

if __name__ == "__main__":
    # Evaluate a specific model (e.g., RandomForest)
    evaluate_model("RandomForest")
