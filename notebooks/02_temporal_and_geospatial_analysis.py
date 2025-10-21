import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import folium

# Load the processed data
df = pd.read_parquet('smart-stays/data/processed/listings.parquet')

# Convert date columns to datetime objects
df['host_since'] = pd.to_datetime(df['host_since'], errors='coerce', format='mixed')
df['first_review'] = pd.to_datetime(df['first_review'], errors='coerce', format='mixed')
df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce', format='mixed')

# Drop rows with NaN values in crucial columns for plotting
df_cleaned = df.dropna(subset=['price', 'latitude', 'longitude', 'last_review'])


# Price seasonality (monthly)
df_cleaned.loc[:, 'month'] = df_cleaned['last_review'].dt.month
plt.figure(figsize=(12, 6))
sns.lineplot(x='month', y='price', data=df_cleaned, errorbar=None)
plt.title('Average Price by Month')
plt.xlabel('Month')
plt.ylabel('Average Price')
plt.xticks(range(1, 13))
plt.grid(True)
plt.show()

# Availability over the year
plt.figure(figsize=(12, 6))
df_cleaned.groupby(df_cleaned['last_review'].dt.to_period('M'))['availability_365'].mean().plot(kind='line')
plt.title('Average Availability Over Time')
plt.xlabel('Date')
plt.ylabel('Average Availability (days)')
plt.grid(True)
plt.show()

# Density map
fig = px.density_mapbox(df_cleaned, lat='latitude', lon='longitude', radius=10,
                        center=dict(lat=df_cleaned.latitude.mean(), lon=df_cleaned.longitude.mean()), zoom=10,
                        mapbox_style="open-street-map")
fig.update_layout(title='Listing Density Map')
fig.show()

# Average price heatmap
# We need geojson data for neighbourhoods to create a choropleth map.
# Without it, we can create a scatter plot on a map.
fig = px.scatter_mapbox(df_cleaned, lat='latitude', lon='longitude', color='price', size='price',
                        color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=10,
                        mapbox_style="open-street-map",
                        hover_name='name', hover_data=['neighbourhood_cleansed', 'price'])
fig.update_layout(title='Average Price Heatmap by Neighbourhood')
fig.show()