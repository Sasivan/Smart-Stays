import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the processed data
df = pd.read_parquet('smart-stays/data/processed/listings.parquet')

# Identify premium/cheap clusters using boxplots
plt.figure(figsize=(12, 8))
sns.boxplot(x='neighbourhood_cleansed', y='price', data=df)
plt.xticks(rotation=90)
plt.title('Price Distribution by Neighbourhood')
plt.show()

# Violin plots for price by room type
plt.figure(figsize=(10, 6))
sns.violinplot(x='room_type', y='price', data=df)
plt.title('Price Distribution by Room Type')
plt.show()

# Correlation matrix
# Select only numeric columns for correlation matrix
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
corr_matrix = df[numeric_cols].corr(method='spearman')

plt.figure(figsize=(15, 12))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Spearman Correlation Matrix of Numeric Features')
plt.show()