import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob

# Load the processed data
df = pd.read_parquet('smart-stays/data/processed/listings.parquet')

# Drop rows with missing reviews
df_reviews = df.dropna(subset=['review_scores_rating', 'reviews_per_month'])

# Average rating distribution
plt.figure(figsize=(10, 6))
sns.histplot(df_reviews['review_scores_rating'], bins=20, kde=True)
plt.title('Distribution of Review Scores Rating')
plt.show()

# Sentiment score distribution for review text
# Note: We don't have the full review text, so we can't do sentiment analysis or TF-IDF.
# We will proceed with the data we have.

# Correlation between amenities and price/popularity
# (We need to parse the 'amenities' column first)
# This is a complex task and is better suited for the feature engineering stage.