import pandas as pd
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
PROCESSED_DATA_PATH = "smart-stays/data/processed/listings.csv"
REPORT_PATH = "smart-stays/data/processed/data_quality_report.txt"

def generate_report():
    """Generates a data quality report for the processed data."""
    logging.info(f"Generating data quality report for {PROCESSED_DATA_PATH}")

    try:
        df = pd.read_csv(PROCESSED_DATA_PATH, low_memory=False)

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

        logging.info(f"Successfully generated data quality report at {REPORT_PATH}")

    except Exception as e:
        logging.error(f"Failed to generate data quality report: {e}")

if __name__ == "__main__":
    generate_report()
