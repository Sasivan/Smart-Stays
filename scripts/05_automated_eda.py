import pandas as pd
from pandas_profiling import ProfileReport
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
PROCESSED_DATA_PATH = "data/processed/listings.parquet"
REPORT_PATH = "reports/pandas_profiling_report.html"

def generate_automated_eda():
    """Generates an automated EDA report using pandas-profiling."""
    logging.info(f"Generating automated EDA report for {PROCESSED_DATA_PATH}")

    try:
        df = pd.read_parquet(PROCESSED_DATA_PATH)
        profile = ProfileReport(df, title="Pandas Profiling Report", explorative=True)
        profile.to_file(REPORT_PATH)

        logging.info(f"Successfully generated automated EDA report at {REPORT_PATH}")

    except Exception as e:
        logging.error(f"Failed to generate automated EDA report: {e}")

if __name__ == "__main__":
    generate_automated_eda()
