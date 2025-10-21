import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import folium

# Load the processed data
df = pd.read_parquet('smart-stays/data/processed/listings.parquet')

# Display the first few rows
df.head()

# Number of listings
print(f'Number of listings: {len(df)}')

# Distribution by neighbourhood
plt.figure(figsize=(12, 6))
sns.countplot(y='neighbourhood_cleansed', data=df, order=df['neighbourhood_cleansed'].value_counts().index)
plt.title('Number of Listings by Neighbourhood')
plt.show()

# Distribution by room type
plt.figure(figsize=(8, 5))
sns.countplot(x='room_type', data=df, order=df['room_type'].value_counts().index)
plt.title('Number of Listings by Room Type')
plt.show()

# Distribution by host response rate
plt.figure(figsize=(8, 5))
sns.histplot(df['host_response_rate'].dropna(), bins=20)
plt.title('Distribution of Host Response Rate')
plt.show()

# Price distribution histogram
plt.figure(figsize=(10, 6))
sns.histplot(df['price'], bins=50, kde=True)
plt.title('Price Distribution')
plt.xlabel('Price')
plt.show()

# Price distribution on a log scale
plt.figure(figsize=(10, 6))
sns.histplot(df['price'], bins=50, log_scale=True, kde=True)
plt.title('Price Distribution (Log Scale)')
plt.xlabel('Price')
plt.show()

# Per-night vs. weekly pricing
# (Assuming 'price' is per-night. We don't have weekly price data in this dataset)