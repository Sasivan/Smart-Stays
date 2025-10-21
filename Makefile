.PHONY: data train evaluate price

fetch_data: ## Fetches the raw data
	python scripts/01_fetch_data.py

validate_data: ## Validates the raw data
	python scripts/02_validate_data.py

clean_data: ## Cleans and preprocesses the raw data
	python scripts/03_clean_data.py

data_quality_report: ## Generates a data quality report
	python scripts/04_data_quality_report.py

automated_eda: ## Generates an automated EDA report
	python scripts/05_automated_eda.py

feature_engineering: ## Performs feature engineering
	python src/features.py

data: ## Downloads and preprocesses the data
	make fetch_data
	make validate_data
	make clean_data
	make data_quality_report
	make feature_engineering

eda: ## Runs the exploratory data analysis
	make automated_eda
	# You can add a command here to run your jupyter notebook if you want

train: ## Trains the model
	python src/train.py

evaluate: ## Evaluates the model
	python src/evaluate.py

price: ## Get price recommendations
	python scripts/pricing_rules.py

all: data eda train evaluate price
