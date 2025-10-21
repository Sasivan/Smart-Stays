# Key Findings from Exploratory Data Analysis

This report summarizes the key insights from the EDA notebooks.

## 1. High-Level Metrics (from 01_eda.ipynb)

* **Number of Listings:** The dataset contains a total of 7,721 listings.
* **Distribution by Neighbourhood:** The listings are not evenly distributed across neighbourhoods. Some neighbourhoods have a significantly higher concentration of listings than others.
* **Distribution by Room Type:** The most common room type is 'Entire home/apt', followed by 'Private room'.
* **Host Response Rate:** The majority of hosts have a high response rate (90-100%).
* **Price Distribution:** The price distribution is right-skewed, with a long tail of expensive listings. A log transformation was applied to normalize the distribution for analysis.

## 2. Temporal Patterns (from 02_temporal_and_geospatial_analysis.ipynb)

* **Price Seasonality:** There is a clear seasonal pattern in pricing, with prices peaking in the summer months.
* **Availability:** Availability also shows a seasonal trend, with higher availability during the off-peak season.

## 3. Geospatial Analysis (from 02_temporal_and_geospatial_analysis.ipynb)

* **Listing Density:** The density map shows that listings are clustered in the city center and other popular areas.
* **Price Heatmap:** The price heatmap confirms that listings in the city center and other desirable locations are more expensive.

## 4. Review and Sentiment Analysis (from 03_review_and_amenity_analysis.ipynb)

* **Rating Distribution:** The distribution of review scores is left-skewed, with most listings having high ratings.

## 5. Outliers and Feature Relationships (from 04_outliers_and_feature_relationships.ipynb)

* **Price Outliers:** There are a number of price outliers, both on the high and low end. These may represent premium or budget listings.
* **Feature Correlations:** The correlation matrix shows that there are a number of features that are correlated with price, such as the number of bedrooms, bathrooms, and accommodates.
