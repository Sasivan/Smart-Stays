import pandas as pd
import numpy as np
import re
import logging
from pathlib import Path
from textblob import TextBlob
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- Configuration & Setup ---

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define file paths
INPUT_PATH = Path("data/processed/listings.parquet")
OUTPUT_PATH = Path("data/processed/listings_featured.parquet")

# Define some popular amenities to check for
COMMON_AMENITIES = [
    'wifi', 'kitchen', 'washer', 'air conditioning', 'dryer',
    'heating', 'tv', 'parking', 'iron', 'pool', 'hot tub'
]

# --- Helper Functions ---

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points
    on the earth (specified in decimal degrees).
    Returns distance in kilometers.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of Earth in kilometers
    return c * r

def clean_amenities_text(text):
    """Helper to clean the raw amenities string for easier parsing."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[\"\'\[\]\{\}]', '', text) # Remove brackets and quotes
    return text

# --- Feature Engineering Functions ---

def create_temporal_features(df):
    """Engineers features from date columns."""
    logging.info("Creating temporal features...")
    
    date_cols = ['host_since', 'first_review', 'last_review']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    if 'last_scraped' in df.columns:
        present_date = pd.to_datetime(df['last_scraped'], errors='coerce').max()
    else:
        present_date = pd.Timestamp.now()
    
    if 'host_since' in df.columns:
        df['host_duration_days'] = (present_date - df['host_since']).dt.days
    
    if 'last_review' in df.columns:
        df['days_since_last_review'] = (present_date - df['last_review']).dt.days
        if df['days_since_last_review'].isnull().any():
            median_days = df['days_since_last_review'].median()
            df['days_since_last_review'] = df['days_since_last_review'].fillna(median_days * 3)
            
    if 'first_review' in df.columns:
        df['days_since_first_review'] = (present_date - df['first_review']).dt.days
    
    df['host_duration_days'] = df['host_duration_days'].fillna(0)
    df['days_since_first_review'] = df['days_since_first_review'].fillna(0)

    return df

def create_geographic_features(df):
    """Engineers features from latitude and longitude."""
    logging.info("Creating geographic features...")
    
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        logging.warning("Latitude/Longitude columns not found. Skipping geo features.")
        return df

    df.dropna(subset=['latitude', 'longitude'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    city_center_lat = df['latitude'].mean()
    city_center_lon = df['longitude'].mean()
    logging.info(f"Calculated city center proxy: ({city_center_lat:.4f}, {city_center_lon:.4f})")
    
    df['distance_to_center'] = df.apply(
        lambda row: haversine(city_center_lat, city_center_lon, row['latitude'], row['longitude']),
        axis=1
    )
    
    POIS = {
        'poi_1_airport': (city_center_lat + 0.1, city_center_lon + 0.1),
        'poi_2_station': (city_center_lat - 0.05, city_center_lon - 0.05)
    }
    
    for name, (lat, lon) in POIS.items():
        df[f'distance_to_{name}'] = df.apply(
            lambda row: haversine(lat, lon, row['latitude'], row['longitude']),
            axis=1
        )
    
    coords = df[['latitude', 'longitude']].copy()
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)
    
    kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
    df['spatial_cluster'] = kmeans.fit_predict(coords_scaled)
    df['spatial_cluster'] = df['spatial_cluster'].astype('category')
    
    return df

def create_listing_features(df):
    """Engineers features from listing properties, like amenities."""
    logging.info("Creating listing features (amenities)...")
    
    if 'amenities' not in df.columns:
        logging.warning("'amenities' column not found. Skipping listing features.")
        return df
        
    df['amenities_cleaned'] = df['amenities'].apply(clean_amenities_text)
    
    for amenity in COMMON_AMENITIES:
        col_name = f'has_{amenity.replace(" ", "_")}'
        df[col_name] = df['amenities_cleaned'].str.contains(f'{amenity}', case=False, na=False).astype(int)
        
    df['amenities_count'] = df['amenities_cleaned'].apply(lambda x: len(x.split(',')) if x else 0)
    df = df.drop(columns=['amenities_cleaned'])
    
    return df

def create_host_features(df):
    """Cleans and engineers features related to the host."""
    logging.info("Creating host features...")
    
    bool_cols = ['host_is_superhost', 'host_identity_verified', 'has_availability']
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: 1 if str(x).lower() == 't' else 0).astype(int)
            
    if 'host_response_rate' in df.columns:
        df['host_response_rate'] = df['host_response_rate'].str.replace('%', '', regex=False).astype(float) / 100.0
        df['host_response_rate'] = df['host_response_rate'].fillna(0)
            
    return df

def create_text_features(df):
    """Engineers features from text columns like 'name' and 'description'."""
    logging.info("Creating text features...")
    
    df['description'] = df['description'].fillna('')
    df['name'] = df['name'].fillna('')
    
    df['description_sentiment'] = df['description'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['description_length'] = df['description'].str.len()
    df['name_length'] = df['name'].str.len()
    
    return df

def create_price_features(df):
    """Engineers price-related features."""
    logging.info("Creating price-specific features...")
    
    if 'accommodates' in df.columns and 'price' in df.columns:
        df['accommodates_clean'] = df['accommodates'].replace(0, 1)
        df['price_per_person'] = df['price'] / df['accommodates_clean']
        df = df.drop(columns=['accommodates_clean'])
    
    if 'cleaning_fee' in df.columns:
        df['cleaning_fee'] = pd.to_numeric(df['cleaning_fee'], errors='coerce').fillna(0)
        df['effective_price'] = df['price'] + df['cleaning_fee']
    
    return df

def create_interaction_features(df):
    """Creates features by interacting existing ones."""
    logging.info("Creating interaction features...")
    
    if 'host_is_superhost' in df.columns and 'review_scores_rating' in df.columns:
        median_rating = df['review_scores_rating'].median()
        df['review_scores_rating_filled'] = df['review_scores_rating'].fillna(median_rating)
        
        df['superhost_x_rating'] = df['host_is_superhost'] * df['review_scores_rating_filled']
        df = df.drop(columns=['review_scores_rating_filled'])
    
    return df

def encode_categorical_features(df):
    """
    Sets the dtype to 'category' so the train.py pipeline can handle it.
    """
    logging.info("Setting categorical dtypes (no OHE).")
    
    cat_cols = ['room_type', 'property_type', 'neighbourhood_cleansed']
    
    for col in cat_cols:
        if col in df.columns:
            # Get top 10 most frequent categories and lump others into 'Other'
            top_10 = df[col].value_counts().nlargest(10).index
            df[col] = df[col].apply(lambda x: x if x in top_10 else 'Other')
            df[col] = df[col].astype('category')
            
    return df

def finalize_feature_set(df):
    """
    Finalizes the feature set, dropping raw columns and filling NaNs.
    """
    logging.info("Finalizing feature set...")
    
    # Drop raw/intermediate/text columns
    cols_to_drop = [
        'amenities', 'description', 'name', 'host_since', 'first_review',
        'last_review', 'latitude', 'longitude', 'host_name', 'host_about',
        'listing_url', 'scrape_id', 'last_scraped', 'picture_url',
        'host_url', 'host_thumbnail_url', 'host_picture_url', 'calendar_last_scraped'
    ]
    
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    # Fill any remaining NaNs in numeric columns with 0
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # Ensure 'price' (our target) is not NaN
    if 'price' in df.columns:
        df = df.dropna(subset=['price'])
        
    logging.info(f"Final feature set shape: {df.shape}")
    
    return df

# --- Main Execution ---

def main():
    """Main pipeline to run all feature engineering steps."""
    logging.info("--- Starting Feature Engineering Pipeline ---")
    
    try:
        df = pd.read_parquet(INPUT_PATH)
        logging.info(f"Loaded data: {df.shape}")
    except FileNotFoundError:
        logging.error(f"FATAL: Input file not found at {INPUT_PATH}")
        logging.error("Please run the src/preprocess.py script first.")
        return

    df_featured = df.copy()
    df_featured = create_temporal_features(df_featured)
    df_featured = create_geographic_features(df_featured)
    df_featured = create_listing_features(df_featured)
    df_featured = create_host_features(df_featured)
    df_featured = create_text_features(df_featured)
    df_featured = create_price_features(df_featured)
    df_featured = create_interaction_features(df_featured)
    df_featured = encode_categorical_features(df_featured)
    
    df_final = finalize_feature_set(df_featured)
    
    try:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df_final.to_parquet(OUTPUT_PATH, index=False)
        logging.info(f"Successfully saved featured data to {OUTPUT_PATH}")
    except Exception as e:
        logging.error(f"Error saving data: {e}")
        
    logging.info("--- Feature Engineering Pipeline Complete ---")

if __name__ == "__main__":
    main()
