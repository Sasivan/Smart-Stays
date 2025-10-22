import pandas as pd
import numpy as np
import logging
import os
import sys

# --- Configuration ---

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define file paths
RAW_DATA_PATH = "../data/raw/listings.csv.gz"
PROCESSED_DIR = "../data/processed"
PROCESSED_DATA_PATH_CSV = os.path.join(PROCESSED_DIR, "listings.csv")
PROCESSED_DATA_PATH_PARQUET = os.path.join(PROCESSED_DIR, "listings.parquet")
REPORT_PATH = os.path.join(PROCESSED_DIR, "data_quality_report.txt")


# --- Step 1: Validation ---

def validate_data():
    """Performs basic validation on the raw data."""
    logging.info(f"Starting validation for {RAW_DATA_PATH}")

    try:
        df = pd.read_csv(RAW_DATA_PATH)

        # Schema checks
        assert "id" in df.columns, "Missing required column: id"
        assert "latitude" in df.columns, "Missing required column: latitude"
        assert "longitude" in df.columns, "Missing required column: longitude"
        assert "price" in df.columns, "Missing required column: price"

        # Sample counts
        logging.info(f"Number of rows: {len(df)}")
        logging.info(f"Number of columns: {len(df.columns)}")

        logging.info(f"✅ Data validation successful for {RAW_DATA_PATH}")
        return True

    except FileNotFoundError:
        logging.error(f"❌ File not found at {RAW_DATA_PATH}. Please ensure the data is downloaded.")
        return False
    except Exception as e:
        logging.error(f"❌ Data validation failed: {e}")
        return False

# --- Step 2: Cleaning ---

def clean_data():
    """Cleans and preprocesses the raw Airbnb listing data."""
    logging.info(f"Starting data cleaning for {RAW_DATA_PATH}")

    try:
        df = pd.read_csv(RAW_DATA_PATH)

        # 1. Standard Cleaning
        df = df.drop_duplicates()
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]

        # 2. Price Cleaning
        # Ensure price is a string before trying to replace
        df['price'] = df['price'].astype(str).str.replace('[\$,]', '', regex=True).astype(float)
        
        # Capping price outliers at the 99th percentile
        price_cap = df['price'].quantile(0.99)
        df.loc[df['price'] > price_cap, 'price'] = price_cap
        logging.info(f"Cleaned and capped 'price' column (99th percentile: ${price_cap:.2f})")

        # 3. Handling Missing Data
        # Impute missing reviews_per_month with 0
        df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
        df['reviews_per_month_imputed'] = df['reviews_per_month'].isnull().astype(int)

        # 4. Temporal Consistency
        df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')
        # Use a fixed reference date (e.g., max date) instead of 'today' for reproducibility
        reference_date = df['last_review'].max()
        if pd.isna(reference_date):
            reference_date = pd.Timestamp.now() # Fallback if no dates exist
            
        df['days_since_last_review'] = (reference_date - df['last_review']).dt.days
        # Fill NaNs with a high value (e.g., median * 3) to indicate staleness
        median_days = df['days_since_last_review'].median()
        df['days_since_last_review'] = df['days_since_last_review'].fillna(median_days * 3)
        df['days_since_last_review_imputed'] = df['days_since_last_review'].isnull().astype(int)
        logging.info("Engineered temporal features.")

        # 5. Normalize Text Columns (Amenities)
        # For simplicity, we'll just count the number of amenities
        df['num_amenities'] = df['amenities'].apply(lambda x: len(x.split(',')) if isinstance(x, str) else 0)

        # Ensure the output directory exists
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        
        # Save the cleaned data
        df.to_csv(PROCESSED_DATA_PATH_CSV, index=False)
        
        logging.info(f"✅ Successfully cleaned data and saved to {PROCESSED_DATA_PATH_CSV}")
        return True

    except Exception as e:
        logging.error(f"❌ Data cleaning failed: {e}")
        return False

# --- Step 3: Reporting ---

def generate_report():
    """Generates a data quality report for the processed data."""
    logging.info(f"Generating data quality report for {PROCESSED_DATA_PATH_CSV}")

    try:
        df = pd.read_csv(PROCESSED_DATA_PATH_CSV, low_memory=False)

        with open(REPORT_PATH, "w") as f:
            f.write("Data Quality Report\n")
            f.write("===================\n\n")
            f.write(f"Number of rows: {len(df)}\n")
            f.write(f"Number of columns: {len(df.columns)}\n\n")
            f.write("Missing values per column:\n")
            f.write(str(df.isnull().sum()))
            f.write("\n\n")
            f.write("Descriptive statistics for numeric columns:\n")
            f.write(str(df.describe()))

        logging.info(f"✅ Successfully generated data quality report at {REPORT_PATH}")
        return True

    except Exception as e:
        logging.error(f"❌ Failed to generate data quality report: {e}")
        return False

# --- Step 4: Parquet Conversion ---

def convert_to_parquet():
    """Converts the processed CSV file to a parquet file."""
    logging.info(f"Converting {PROCESSED_DATA_PATH_CSV} to parquet format.")

    try:
        df = pd.read_csv(PROCESSED_DATA_PATH_CSV, low_memory=False)
        df.to_parquet(PROCESSED_DATA_PATH_PARQUET, index=False)
        logging.info(f"✅ Successfully converted to {PROCESSED_DATA_PATH_PARQUET}")
        
        # Optional: Remove the intermediate CSV to save space
        # os.remove(PROCESSED_DATA_PATH_CSV)
        # logging.info(f"Removed intermediate file: {PROCESSED_DATA_PATH_CSV}")
        
        return True
        
    except Exception as e:
        logging.error(f"❌ Failed to convert to parquet: {e}")
        return False

# --- Main Execution ---

def main():
    """Runs the full preprocessing pipeline."""
    logging.info("--- Starting Data Preprocessing Pipeline ---")
    
    if not validate_data():
        logging.error("Halting pipeline due to validation failure.")
        sys.exit(1) # Exit with an error code
        
    if not clean_data():
        logging.error("Halting pipeline due to cleaning failure.")
        sys.exit(1)
        
    if not generate_report():
        logging.error("Halting pipeline due to reporting failure.")
        sys.exit(1)
        
    if not convert_to_parquet():
        logging.error("Halting pipeline due to parquet conversion failure.")
        sys.exit(1)

    logging.info("--- ✅ Data Preprocessing Pipeline Completed Successfully ---")

if __name__ == "__main__":
    main()