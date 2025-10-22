import streamlit as st
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import subprocess
import os
import streamlit.components.v1 as components 

# --- Dynamic Path Setup ---
DASHBOARD_DIR = Path(__file__).resolve().parent
PROJECT_DIR = DASHBOARD_DIR.parent
SRC_DIR = PROJECT_DIR / "src"
REPORTS_DIR = PROJECT_DIR / "reports" 

# --- Define Paths for All Notebooks and Output HTMLs ---
NOTEBOOK_01_PATH = PROJECT_DIR / "notebooks/01_eda.ipynb"
HTML_01_OUTPUT_PATH = DASHBOARD_DIR / "01_eda_rendered.html"

NOTEBOOK_02_PATH = PROJECT_DIR / "notebooks/02_feature_engineering.ipynb"
HTML_02_OUTPUT_PATH = DASHBOARD_DIR / "02_feature_engineering_rendered.html"

NOTEBOOK_03_PATH = PROJECT_DIR / "notebooks/03_modeling.ipynb"
HTML_03_OUTPUT_PATH = DASHBOARD_DIR / "03_modeling_rendered.html"

# --- Define Path for the Final Markdown Report ---
REPORT_PATH = REPORTS_DIR / "final_summary.md" 

# --- Define Paths for Images ---
LGBM_IMPORTANCE_PATH = REPORTS_DIR / "lgbm_feature_importance.png"
SHAP_BAR_PATH = REPORTS_DIR / "shap_summary_bar.png"
SHAP_BEESWARM_PATH = REPORTS_DIR / "shap_summary_beeswarm.png"


sys.path.append(str(SRC_DIR))

# --- Import custom functions (Assumes 'predict.py' and 'plotting.py' exist in 'src') ---
try:
    from predict import load_model_and_categories, predict_price
    from plotting import (
        plot_listings_by_neighbourhood,
        plot_price_distribution,
        plot_listings_by_room_type,
        plot_price_by_room_type,
        plot_map
    )
except ImportError as e:
    st.error(f"❌ Could not import functions. Error: {e}. Check paths and file contents.")
    st.stop()

# --- Configuration & File Paths ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
st.set_page_config(layout="wide", page_title="Smart Stays - Price Predictor")

DATA_PATH = PROJECT_DIR / "data/processed/listings_featured.parquet"
ORIGINAL_DATA_PATH = PROJECT_DIR / "data/processed/listings.parquet"


# --- NBConvert Rendering Function (Cached, run when tab is clicked) ---
@st.cache_data(show_spinner=False)
def get_notebook_html(notebook_path, output_path, spinner_text):
    """
    Executes nbconvert and loads the resulting HTML file.
    """
    if not notebook_path.exists():
        return f"❌ Notebook not found at: {notebook_path}"

    with st.spinner(spinner_text):
        command = [
            "jupyter", "nbconvert", 
            str(notebook_path), 
            "--to", "html", 
            f"--output={output_path.name}", 
            f"--output-dir={output_path.parent}",
            "--execute", 
        ]
        
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            
            if output_path.exists():
                with open(output_path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                return f"❌ nbconvert finished, but output file not found at: {output_path}"

        except subprocess.CalledProcessError as e:
            error_message = f"❌ nbconvert failed! Error: {e.stderr}"
            return error_message
        except FileNotFoundError:
            return "❌ Error: 'jupyter' or 'nbconvert' command not found. Ensure Jupyter/nbconvert is installed and in your PATH."
        except Exception as e:
            return f"❌ An unexpected error occurred during rendering: {e}"


# --- Function to load the Final Report Markdown ---
@st.cache_data
def load_report_markdown(report_path: Path) -> str:
    """Loads the final analysis report content."""
    if report_path.exists():
        with open(report_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "❌ Report file not found. Please ensure `reports/report_analysis.py` has been run successfully to generate `reports/final_summary.md`."


# --- Load Resources (Model, Categories) ---
@st.cache_resource
def load_resources_cached():
    """Caches the model, category info, and model features."""
    try:
        model, category_info = load_model_and_categories()
        if model is None or category_info is None:
            return None, None, None
        
        try:
            model_features = model.feature_name_
        except AttributeError:
            return None, None, None
            
        return model, category_info, model_features
    except Exception as e:
        return None, None, None


# --- Load Data for Display/EDA (ROBUST VERSION) ---
@st.cache_data
def load_display_data():
    """
    Loads the featured and original data, merges them, and prepares the display DataFrame.
    """
    
    required_cols = ['price', 'latitude', 'longitude', 'neighbourhood_cleansed', 'room_type']
    
    with st.spinner("Loading and processing market data..."):
        try:
            df_featured = pd.read_parquet(DATA_PATH)
            df_original = pd.read_parquet(
                ORIGINAL_DATA_PATH, 
                columns=['price', 'latitude', 'longitude', 'id', 'neighbourhood_cleansed', 'room_type']
            )
            
            if df_featured.empty or df_original.empty:
                st.error("❌ One or both Parquet files were loaded as empty. Check file contents.")
                return None

            df_display = pd.merge(
                df_featured.reset_index(drop=True), 
                df_original.reset_index(drop=True), 
                left_index=True, 
                right_index=True, 
                how='left', 
                suffixes=('', '_orig')
            )

            for col in required_cols:
                if col not in df_display.columns:
                    fallback_col = col + '_orig'
                    if fallback_col in df_display.columns:
                        df_display[col] = df_display[fallback_col]
                    else:
                        default_val = 0 if col in ['price', 'latitude', 'longitude'] else 'Unknown'
                        df_display[col] = default_val
            
            df_display = df_display.dropna(subset=required_cols)
            
            if df_display.empty:
                st.error("❌ All rows were dropped after cleaning. Check data quality and NaN presence.")
                return None

            return df_display
            
        except FileNotFoundError as fnfe:
            st.error(f"❌ Failed to load display data: File Not Found. Check path: {fnfe}")
            return None
        except Exception as e:
            st.error(f"❌ Failed to load display data: {e}")
            return None

# --- Initialize App ---
st.title("Smart Stays: Airbnb Price Predictor & Insights 🏠")

# Load model and market data on initial run
model, category_info, model_features = load_resources_cached()
df_display = load_display_data()

if df_display is not None and model is not None:
    st.success("✅ App resources and market data loaded successfully.")

if model is None or category_info is None or df_display is None:
    st.error("App initialization failed. Cannot proceed.")
    st.stop()
    
# --- Tabs (PRIORITIZED ORDER: Market, Prediction, Report) ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Market Overview 🗺️",      # 1st
    "Price Prediction 💰",      # 2nd
    "Project Report 📄",         # 3rd 
    "Full EDA 📊",              # 4th 
    "Feature Engineering 🛠️",   # 5th 
    "Modeling 🧠"               # 6th 
])

# =========================
# === Market Overview Tab (Tab 1) ===
# =========================
with tab1:
    st.header("Market Overview")
    st.markdown("Quick visualizations based on current market data.")

    if df_display is not None and not df_display.empty:
        col_eda_1, col_eda_2 = st.columns(2)

        with col_eda_1:
            st.subheader("Listings per Neighbourhood (Top 20)")
            fig_neigh = plot_listings_by_neighbourhood(df_display, top_n=20)
            st.plotly_chart(fig_neigh, use_container_width=True) # FIX: Reverted to use_container_width

            st.subheader("Price Distribution")
            fig_hist = plot_price_distribution(df_display, percentile=0.98)
            st.plotly_chart(fig_hist, use_container_width=True) # FIX: Reverted to use_container_width

        with col_eda_2:
            st.subheader("Listings per Room Type")
            fig_room = plot_listings_by_room_type(df_display)
            st.plotly_chart(fig_room, use_container_width=True) # FIX: Reverted to use_container_width

            st.subheader("Price vs Room Type")
            fig_violin = plot_price_by_room_type(df_display, percentile=0.98)
            st.plotly_chart(fig_violin, use_container_width=True) # FIX: Reverted to use_container_width

        st.subheader("Listing Location and Price Map")
        fig_map = plot_map(df_display, sample_n=7000)
        st.plotly_chart(fig_map, use_container_width=True) # FIX: Reverted to use_container_width
    else:
        st.warning("⚠️ Market data could not be loaded or is empty. Please check the data source paths.")

# =========================
# === Price Prediction Tab (Tab 2) ===
# =========================
with tab2:
    st.header("Get a Price Prediction")
    st.markdown("Enter listing details below.")

    # --- EDIT: Removed 'room_type' and 'property_type' definitions ---
    neighbourhoods = category_info.get('neighbourhood_cleansed', ['Missing'])
    if 'Missing' not in neighbourhoods: neighbourhoods.append('Missing')


    with st.form("prediction_form"):
        st.subheader("Listing Details")
        col1, col2, col3 = st.columns(3)
        with col1:
            neighbourhood = st.selectbox("Neighbourhood", options=neighbourhoods, key='neighbourhood')
            accommodates = st.number_input("Accommodates", min_value=1, max_value=20, value=2, step=1, key='accommodates')
            
            # --- EDIT: Removed 'room_type' and 'property_type' ---
            
        with col2:
            bedrooms = st.number_input("Bedrooms", min_value=0, max_value=15, value=1, step=1, key='bedrooms')
            beds = st.number_input("Beds", min_value=0, max_value=20, value=1, step=1, key='beds')
            
        with col3:
            bathrooms_feature_name = 'bathrooms_text' 
            bathrooms = st.number_input("Bathrooms", min_value=0.0, max_value=10.0, value=1.0, step=0.5, key='bathrooms')
            min_nights = st.number_input("Minimum Nights", min_value=1, max_value=365, value=2, step=1, key='min_nights')
            
        # Moved review_score outside columns to take full width
        review_score = st.slider("Review Score Rating (0-100)", min_value=0, max_value=100, value=95, step=1, key='review_score')


        st.subheader("Select Amenities (Common)")
        amenity_cols = st.columns(4)
        has_wifi = amenity_cols[0].checkbox("WiFi", value=True, key='has_wifi', disabled='has_wifi' not in model_features)
        has_kitchen = amenity_cols[1].checkbox("Kitchen", value=True, key='has_kitchen', disabled='has_kitchen' not in model_features)
        has_ac = amenity_cols[2].checkbox("Air Conditioning", value=True, key='has_ac', disabled='has_air_conditioning' not in model_features)
        has_parking = amenity_cols[3].checkbox("Free Parking", value=False, key='has_parking', disabled='has_parking' not in model_features)

        submitted = st.form_submit_button("Predict Price")

    # --- Prediction Logic ---
    if submitted:
        input_data_dict = {
            'neighbourhood_cleansed': st.session_state.neighbourhood,
            'accommodates': st.session_state.accommodates,
            'bedrooms': st.session_state.bedrooms,
            'beds': st.session_state.beds,
            bathrooms_feature_name: st.session_state.bathrooms,
            'minimum_nights': st.session_state.min_nights,
            'review_scores_rating': st.session_state.review_score,
            'has_wifi': 0 if st.session_state.has_wifi else 1,
            
            # --- EDIT: Removed 'room_type' and 'property_type' ---
            
            # FIX: Inverting kitchen logic for the current model based on user feedback.
            'has_kitchen': 0 if st.session_state.has_kitchen else 1, 
            
            'has_air_conditioning': 1 if st.session_state.has_ac else 0,
            'has_parking': 1 if st.session_state.has_parking else 0,
        }
        
        input_df = pd.DataFrame([input_data_dict])
        
        # Default value imputation for features not explicitly controlled by UI
        # This will now automatically add 'room_type' and 'property_type' with a
        # default value (e.g., "Missing"), which 'predict.py' handles.
        default_values = {}
        for feature in model_features:
            if feature not in input_df.columns:
                if feature in df_display.columns:
                    if pd.api.types.is_numeric_dtype(df_display[feature]): 
                        default_values[feature] = df_display[feature].median()
                    else:
                        try: 
                            default_values[feature] = df_display[feature].mode()[0]
                        except IndexError: 
                            default_values[feature] = "Missing" if feature in category_info else 0
                elif feature in category_info: 
                    default_values[feature] = "Missing"
                else: 
                    default_values[feature] = 0
        
        for feature, default in default_values.items(): 
            input_df[feature] = default

        try: 
            input_df = input_df[model_features]
        except KeyError as e: 
            st.error(f"Aligning columns failed: {e}. Check if model features match input features.")
            st.stop()

        with st.spinner("Predicting..."):
            predictions = predict_price(input_df, model, category_info) 

        if predictions is not None:
            st.success(f"### Predicted Price: ${predictions[0]:,.2f}")
            
            try:
                import shap
                with st.spinner("Generating explanation..."):
                    input_df_for_shap = input_df.copy()
                    
                    shap_cats = [col for col in model_features if col in category_info]
                    shap_nums = [col for col in model_features if col not in category_info]
                    
                    input_df_for_shap.loc[:, shap_nums] = input_df_for_shap.loc[:, shap_nums].fillna(0).replace([np.inf, -np.inf], 0)
                    
                    for col in shap_cats:
                        input_df_for_shap[col] = input_df_for_shap[col].fillna("Missing")
                        known_cats = category_info.get(col, ["Missing"])
                        if "Missing" not in known_cats: known_cats.append("Missing")
                        
                        input_df_for_shap[col] = pd.Categorical(input_df_for_shap[col], categories=known_cats, ordered=False)
                        
                        if input_df_for_shap[col].isnull().any():
                            input_df_for_shap[col] = input_df_for_shap[col].fillna("Missing")
                            input_df_for_shap[col] = pd.Categorical(input_df_for_shap[col], categories=known_cats, ordered=False)

                    explainer = shap.TreeExplainer(model)
                    shap_values_instance = explainer.shap_values(input_df_for_shap)

                st.subheader("Prediction Explanation")
                fig_shap, ax_shap = plt.subplots()
                
                shap_explanation = shap.Explanation(
                    values=shap_values_instance[0], 
                    base_values=explainer.expected_value,
                    data=input_df_for_shap.iloc[0], 
                    feature_names=input_df_for_shap.columns.tolist()
                )
                
                shap.waterfall_plot(shap_explanation, max_display=10, show=False)
                st.pyplot(fig_shap)
                plt.close(fig_shap)

            except ImportError: 
                st.info("SHAP not installed. Install with `pip install shap` for prediction explanation.")
            except Exception as e: 
                st.error(f"SHAP explanation failed: {e}")
        else:
            st.error("Prediction failed.")

# ==========================================================
# === Project Report Tab (Tab 3: NEW PRIORITY) ===
# ==========================================================
with tab3:
    st.header("Final Project Analysis Report")
    st.markdown("This report summarizes the entire workflow, data, and model performance, with **embedded artifacts**.")
    st.markdown("---")
    
    report_markdown_content = load_report_markdown(REPORT_PATH)
    
    # Display the structural Markdown content
    if report_markdown_content.startswith("❌"):
        st.error(report_markdown_content)
    else:
        st.markdown(report_markdown_content, unsafe_allow_html=True)
        
        # --- Explicitly display PNGs using st.image in a column layout ---
        st.subheader("Model Interpretation & Diagnostics")
        
        # Create 3 columns for side-by-side display
        col_lgbm, col_bar, col_beeswarm = st.columns(3)

        # Display Feature Importance (Col 1)
        with col_lgbm:
            st.markdown("**LGBM Feature Importance**")
            st.caption("A visual representation of the overall Gini importance of features derived from the trained LightGBM model.")
            if LGBM_IMPORTANCE_PATH.exists():
                st.image(str(LGBM_IMPORTANCE_PATH), use_container_width=True) # FIX: Reverted to use_container_width
            else:
                st.warning(f"Image not found: {LGBM_IMPORTANCE_PATH.name}")

        # Display SHAP Summary Bar (Col 2)
        with col_bar:
            st.markdown("**SHAP Summary Bar**")
            st.caption("A summary bar plot showing the average impact (magnitude) of the top features on the model's output, derived from SHAP values.")
            if SHAP_BAR_PATH.exists():
                st.image(str(SHAP_BAR_PATH), use_container_width=True) # FIX: Reverted to use_container_width
            else:
                st.warning(f"Image not found: {SHAP_BAR_PATH.name}")

        # Display SHAP Beeswarm (Col 3)
        with col_beeswarm:
            st.markdown("**SHAP Beeswarm Plot**")
            st.caption("A detailed SHAP beeswarm plot showing how feature values affect the prediction outcome.")
            if SHAP_BEESWARM_PATH.exists():
                st.image(str(SHAP_BEESWARM_PATH), use_container_width=True) # FIX: Reverted to use_container_width
            else:
                st.warning(f"Image not found: {SHAP_BEESWARM_PATH.name}")
        
        st.markdown(f"***\n*Model Scores and raw metrics are contained in `model_scores.json` in the `/reports` directory.*")


# =========================================================
# === Full EDA Tab (Tab 4: Background) ===
# =========================================================
with tab4:
    st.header("Full Exploratory Data Analysis Report")
    st.markdown(f"This content is generated by executing and converting `{NOTEBOOK_01_PATH.name}` to HTML using `jupyter nbconvert`.")
    st.markdown("---")
    
    eda_html_content = get_notebook_html(NOTEBOOK_01_PATH, HTML_01_OUTPUT_PATH, "Rendering EDA Notebook...")

    if eda_html_content.startswith("❌"):
        st.error(eda_html_content)
    else:
        components.html(eda_html_content, height=2000, scrolling=True)

# ======================================================================
# === Feature Engineering Tab (Tab 5: Background) ===
# ======================================================================
with tab5:
    st.header("Feature Engineering Report")
    st.markdown(f"Details on how raw features were transformed for modeling, extracted from `{NOTEBOOK_02_PATH.name}`.")
    st.markdown("---")
    
    feature_html_content = get_notebook_html(NOTEBOOK_02_PATH, HTML_02_OUTPUT_PATH, "Rendering Feature Engineering Notebook...")

    if feature_html_content.startswith("❌"):
        st.error(feature_html_content)
    else:
        components.html(feature_html_content, height=2000, scrolling=True)

# ==========================================================
# === Modeling Tab (Tab 6: Background) ===
# ==========================================================
with tab6:
    st.header("Modeling and Training Report")
    st.markdown(f"Details on the machine learning model selection, training, and evaluation, extracted from `{NOTEBOOK_03_PATH.name}`.")
    st.markdown("---")
    
    modeling_html_content = get_notebook_html(NOTEBOOK_03_PATH, HTML_03_OUTPUT_PATH, "Rendering Modeling Notebook...")

    if modeling_html_content.startswith("❌"):
        st.error(modeling_html_content)
    else:
        components.html(modeling_html_content, height=2000, scrolling=True)


# --- Footer ---
st.markdown("---")
st.caption("Smart Stays Price Prediction Model v0.1")

