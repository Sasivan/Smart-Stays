import joblib
import numpy as np
import pandas as pd
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PriceOptimizer:
    def __init__(self, model_path='models/RandomForest.joblib'):
        """Initializes the PriceOptimizer with a trained model."""
        try:
            self.model = joblib.load(model_path)
            logging.info(f"Successfully loaded model from {model_path}")
        except FileNotFoundError:
            logging.error(f"Model not found at {model_path}. Please train the model first.")
            self.model = None

    def recommend_price(self, features, min_price=10, surge_multiplier=1.0):
        """Recommends a price based on model prediction and business rules."""
        if self.model is None:
            return None

        # Predict base price
        base_price = self.model.predict(features)[0]

        # Apply business rules
        price = max(base_price, min_price)
        price *= surge_multiplier
        
        # Round to nearest 5
        price = round(price / 5) * 5

        return price

    def get_price_buckets(self, recommended_price):
        """Returns conservative, recommended, and aggressive price buckets."""
        return {
            'conservative': recommended_price * 0.9,
            'recommended': recommended_price,
            'aggressive': recommended_price * 1.1
        }

    def explain_recommendation(self, features):
        """Provides a basic explanation for the recommendation."""
        # This is a simplified explanation. For real SHAP values, you'd need to integrate the shap library.
        feature_importances = pd.Series(self.model.feature_importances_, index=features.columns)
        top_features = feature_importances.nlargest(3)
        return top_features.to_dict()

def main():
    """Main function to demonstrate price optimization."""
    logging.info("Starting price optimization demonstration.")

    # Load a sample feature vector (first row from our feature set)
    try:
        features_df = pd.read_parquet('data/processed/features.parquet')
        sample_features = features_df.head(1)
         # Handle potential infinity values from feature engineering
        sample_features.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Simple imputation - you might want to use a more sophisticated approach
        sample_features.fillna(features_df.median(), inplace=True)
    except FileNotFoundError:
        logging.error("Features file not found. Please run the feature engineering pipeline.")
        return

    optimizer = PriceOptimizer()
    if optimizer.model:
        recommended_price = optimizer.recommend_price(sample_features)
        if recommended_price:
            price_buckets = optimizer.get_price_buckets(recommended_price)
            explanation = optimizer.explain_recommendation(sample_features)

            logging.info(f"\n--- Price Recommendation ---")
            logging.info(f"Recommended Price: ${recommended_price:.2f}")
            logging.info(f"Price Buckets:")
            logging.info(f"  - Conservative: ${price_buckets['conservative']:.2f}")
            logging.info(f"  - Recommended:  ${price_buckets['recommended']:.2f}")
            logging.info(f"  - Aggressive:   ${price_buckets['aggressive']:.2f}")
            logging.info(f"\n--- Recommendation Explanation (Top 3 Factors) ---")
            for feature, importance in explanation.items():
                logging.info(f"  - {feature}: {importance:.4f}")

if __name__ == "__main__":
    main()
