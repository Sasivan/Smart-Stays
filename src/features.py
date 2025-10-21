
import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
PROCESSED_DATA_PATH = "data/processed/listings.parquet"
FEATURES_PATH = "data/processed/features.parquet"
TARGET_PATH = "data/processed/target.parquet"
AUSTIN_CENTER = (30.2672, -97.7431)

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates the Haversine distance between two points."""
    R = 6371  # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c

def engineer_features():
    """Engineers features for the Airbnb listing data."""
    logging.info("Starting feature engineering.")

    try:
        df = pd.read_parquet(PROCESSED_DATA_PATH)

        # 1. Geographic Features
        df["distance_to_center"] = haversine_distance(df["latitude"], df["longitude"], AUSTIN_CENTER[0], AUSTIN_CENTER[1])
        kmeans = MiniBatchKMeans(n_clusters=10, random_state=42, batch_size=256, n_init=10)
        df["geo_cluster"] = kmeans.fit_predict(df[["latitude", "longitude"]])

        # 2. Temporal Features
        df["month"] = df["last_review"].dt.month
        df["day_of_week"] = df["last_review"].dt.dayofweek
        df["season"] = (df["month"] % 12 + 3) // 3

        # 3. Listing Features
        amenities = ["wifi", "kitchen", "washer", "air conditioning"]
        for amenity in amenities:
            df[f"has_{amenity.replace(' ', '_')}"] = df["amenities"].str.contains(amenity, case=False).astype(int)

        # 4. Host Features
        df["host_is_superhost"] = (df["host_is_superhost"] == "t").astype(int)

        # 5. Price-specific Features
        df["price_per_person"] = df["price"] / df["accommodates"].replace(0, 1)

        # 6. Categorical Encoding (One-Hot for low cardinality)
        df = pd.get_dummies(df, columns=["room_type", "neighbourhood_cleansed", "season"], prefix=["room", "neighbourhood", "season"])

        # Define features and target
        features = [
            "distance_to_center", "geo_cluster", "month", "day_of_week",
            "has_wifi", "has_kitchen", "has_washer", "has_air_conditioning",
            "host_is_superhost", "host_listings_count", "minimum_nights",
            "availability_365", "price_per_person", "review_scores_rating", "number_of_reviews" 
        ] + [col for col in df.columns if col.startswith("room_") or col.startswith("neighbourhood_") or col.startswith("season_")]
        
        target = "price"

        # Save features and target
        df[features].to_parquet(FEATURES_PATH)
        df[target].to_frame().to_parquet(TARGET_PATH)

        logging.info("Successfully engineered and saved features and target.")

    except Exception as e:
        logging.error(f"Feature engineering failed: {e}")

if __name__ == "__main__":
    engineer_features()
