import pandas as pd
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
PROCESSED_DATA_PATH_CSV = "smart-stays/data/processed/listings.csv"
PROCESSED_DATA_PATH_PARQUET = "smart-stays/data/processed/listings.parquet"

def convert_to_parquet():
    """Converts the processed CSV file to a parquet file."""
    logging.info(f"Converting {PROCESSED_DATA_PATH_CSV} to parquet format.")

    try:
        df = pd.read_csv(PROCESSED_DATA_PATH_CSV, low_memory=False)
        df.to_parquet(PROCESSED_DATA_PATH_PARQUET, index=False)
        logging.info(f"Successfully converted to {PROCESSED_DATA_PATH_PARQUET}")

    except Exception as e:
        logging.error(f"Failed to convert to parquet: {e}")

if __name__ == "__main__":
    convert_to_parquet()
