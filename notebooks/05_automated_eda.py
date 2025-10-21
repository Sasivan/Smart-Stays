#!/usr/bin/env python
# coding: utf-8

# # Automated EDA with Sweetviz

# In[ ]:


import pandas as pd
import sweetviz as sv

# Load the processed data
df = pd.read_parquet('smart-stays/data/processed/listings.parquet')

# Generate the report
report = sv.analyze(df)

# Save the report
report.show_html('smart-stays/reports/eda_report.html')

