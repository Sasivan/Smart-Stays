import pandas as pd
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
DATA_PATH = "smart-stays/data/raw/listings.csv.gz"

def validate_data():
    """Performs basic validation on the raw data."""
    logging.info(f"Starting validation for {DATA_PATH}")

    try:
        df = pd.read_csv(DATA_PATH)

        # Schema checks
        assert "id" in df.columns, "Missing required column: id"
        assert "latitude" in df.columns, "Missing required column: latitude"
        assert "longitude" in df.columns, "Missing required column: longitude"
        assert "price" in df.columns, "Missing required column: price"

        # Sample counts
        logging.info(f"Number of rows: {len(df)}")
        logging.info(f"Number of columns: {len(df.columns)}")

        logging.info(f"✅ Data validation successful for {DATA_PATH}")

    except FileNotFoundError:
        logging.error(f"❌ File not found at {DATA_PATH}. Please ensure the data is downloaded and in the correct location.")
    except Exception as e:
        logging.error(f"❌ Data validation failed: {e}")

if __name__ == "__main__":
    validate_data()