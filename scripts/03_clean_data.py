import pandas as pd
import numpy as np
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
RAW_DATA_PATH = "smart-stays/data/raw/listings.csv.gz"
PROCESSED_DATA_PATH = "smart-stays/data/processed/listings.csv"

def clean_data():
    """Cleans and preprocesses the raw Airbnb listing data."""""
    logging.info(f"Starting data cleaning for {RAW_DATA_PATH}")

    try:
        df = pd.read_csv(RAW_DATA_PATH)

        # 1. Standard Cleaning
        df = df.drop_duplicates()
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]

        # 2. Price Cleaning
        df['price'] = df['price'].str.replace('[\$,]', '', regex=True).astype(float)
        # Capping price outliers at the 99th percentile
        price_cap = df['price'].quantile(0.99)
        df.loc[df['price'] > price_cap, 'price'] = price_cap

        # 3. Handling Missing Data
        # Impute missing reviews_per_month with 0
        df['reviews_per_month'].fillna(0, inplace=True)
        df['reviews_per_month_imputed'] = df['reviews_per_month'].isnull().astype(int)

        # 4. Temporal Consistency
        df['last_review'] = pd.to_datetime(df['last_review'])
        df['days_since_last_review'] = (pd.to_datetime('today') - df['last_review']).dt.days
        df['days_since_last_review'].fillna(df['days_since_last_review'].median(), inplace=True)
        df['days_since_last_review_imputed'] = df['days_since_last_review'].isnull().astype(int)

        # 6. Normalize Text Columns (Amenities)
        # For simplicity, we'll just count the number of amenities
        df['num_amenities'] = df['amenities'].apply(lambda x: len(x.split(',')) if isinstance(x, str) else 0)

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
        
        # Save the cleaned data
        df.to_csv(PROCESSED_DATA_PATH, index=False)
        
        # Check if file exists
        if os.path.exists(PROCESSED_DATA_PATH):
            logging.info(f"Successfully cleaned data and saved to {PROCESSED_DATA_PATH}")
        else:
            logging.error(f"File not found after saving: {PROCESSED_DATA_PATH}")

    except Exception as e:
        logging.error(f"Data cleaning failed: {e}")

if __name__ == "__main__":
    clean_data()
