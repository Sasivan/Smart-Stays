# src/plotting.py
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import logging # Optional: for logging within functions

def plot_listings_by_neighbourhood(df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    """Generates a Plotly bar chart for listings per neighbourhood."""
    logging.info(f"Generating neighbourhood plot for top {top_n}...")
    if 'neighbourhood_cleansed' not in df.columns:
        logging.warning("Neighbourhood column missing for plot.")
        return go.Figure().update_layout(title_text="Neighbourhood data missing")

    neighbourhood_counts = df['neighbourhood_cleansed'].value_counts().nlargest(top_n).reset_index()
    neighbourhood_counts.columns = ['neighbourhood_cleansed', 'count']
    fig = px.bar(neighbourhood_counts, y='neighbourhood_cleansed', x='count',
                   orientation='h', title=f"Top {top_n} Neighbourhoods by Listing Count",
                   labels={'count': 'Number of Listings', 'neighbourhood_cleansed': 'Neighbourhood'},
                   height=max(400, top_n * 25)) # Dynamic height
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    return fig

def plot_price_distribution(df: pd.DataFrame, percentile: float = 0.98) -> go.Figure:
    """Generates a Plotly histogram for price distribution."""
    logging.info(f"Generating price distribution plot up to {percentile*100:.0f}th percentile...")
    if 'price' not in df.columns or df['price'].isnull().all():
        logging.warning("Price column missing or empty for plot.")
        return go.Figure().update_layout(title_text="Price data missing or invalid")

    price_limit = df['price'].quantile(percentile)
    plot_df_hist = df[(df['price'] > 0) & (df['price'] <= price_limit)]

    if plot_df_hist.empty:
         logging.warning("No valid price data in the range for histogram.")
         return go.Figure().update_layout(title_text=f"No price data up to {price_limit:,.0f}")

    fig = px.histogram(plot_df_hist, x="price", nbins=50,
                        title=f"Distribution of Nightly Prices (Up to {price_limit:,.0f}, > $0)")
    return fig

def plot_listings_by_room_type(df: pd.DataFrame) -> go.Figure:
     """Generates a Plotly bar chart for listings per room type."""
     logging.info("Generating room type plot...")
     if 'room_type' not in df.columns:
         logging.warning("Room type column missing for plot.")
         return go.Figure().update_layout(title_text="Room type data missing")

     room_type_counts = df['room_type'].value_counts().reset_index()
     room_type_counts.columns = ['room_type', 'count']
     fig = px.bar(room_type_counts, x='room_type', y='count',
                  title="Listings by Room Type",
                  labels={'count': 'Number of Listings', 'room_type': 'Room Type'})
     return fig

def plot_price_by_room_type(df: pd.DataFrame, percentile: float = 0.98) -> go.Figure:
    """Generates a Plotly violin plot for price by room type."""
    logging.info("Generating price vs room type plot...")
    if 'room_type' not in df.columns or 'price' not in df.columns:
        logging.warning("Room type or price data missing for violin plot.")
        return go.Figure().update_layout(title_text="Room type or price data missing")

    plot_data_violin = df.dropna(subset=['price', 'room_type']).copy()
    plot_data_violin['room_type'] = plot_data_violin['room_type'].astype(str)
    price_limit_violin = plot_data_violin['price'].quantile(percentile)
    plot_data_violin = plot_data_violin[plot_data_violin['price'] <= price_limit_violin]

    if plot_data_violin.empty:
        logging.warning("No valid data for room type price distribution.")
        return go.Figure().update_layout(title_text=f"No price data up to {price_limit_violin:,.0f}")

    # Order by median price
    order = plot_data_violin.groupby('room_type')['price'].median().sort_values().index.tolist()
    fig = px.violin(plot_data_violin, x='room_type', y='price',
                    box=True, points=False,
                    title=f"Price Distribution by Room Type (up to {price_limit_violin:,.0f})",
                    labels={'price': 'Price ($)', 'room_type': 'Room Type'},
                    category_orders={"room_type": order})
    return fig


def plot_map(df: pd.DataFrame, sample_n: int = 5000) -> go.Figure:
    """Generates a Plotly scatter mapbox plot."""
    logging.info("Generating map plot...")
    cols_needed = ['latitude', 'longitude', 'price', 'id', 'neighbourhood_cleansed', 'room_type']
    if not all(col in df.columns for col in cols_needed):
         logging.warning(f"Missing one or more columns for map: {cols_needed}")
         return go.Figure().update_layout(title_text="Map data missing (Lat/Lon/Price/etc.)")

    map_df = df.sample(min(sample_n, len(df)), random_state=42).copy()
    map_df['price_log1p'] = np.log1p(map_df['price'])
    hover_cols = ['neighbourhood_cleansed', 'room_type', 'price']
    hover_data_dict = {col: True for col in hover_cols if col in map_df.columns}
    if 'price' in hover_data_dict: hover_data_dict['price'] = ':.2f'

    fig = px.scatter_mapbox(map_df,
                            lat="latitude", lon="longitude",
                            color="price_log1p", size="price",
                            color_continuous_scale=px.colors.sequential.Viridis,
                            size_max=15, zoom=10,
                            center=dict(lat=map_df.latitude.median(), lon=map_df.longitude.median()),
                            mapbox_style="carto-positron",
                            hover_name='id', # Assumes 'id' column exists after merge
                            hover_data=hover_data_dict,
                            title="Listing Prices (Colored by Log-Price, Sized by Price)")
    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    return fig