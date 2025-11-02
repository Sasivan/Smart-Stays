#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- Starting Smart Stays Pipeline ---"

# --- CORRECTED PATHS ---
# Go up one level (from 'scripts' to project root) to find 'venv'
VENV_PYTHON="venv/bin/python"

# Go up one level to find the 'src' directory
SRC_DIR="src"
# --- END CORRECTIONS ---


# --- Check if venv exists ---
if [ ! -f "$VENV_PYTHON" ]; then
    echo "❌ Error: Python executable not found at $VENV_PYTHON."
    echo "Please ensure the virtual environment 'venv' exists in the project root '/workspaces/Smart-Stays/venv'."
    exit 1
fi

echo "🐍 Using Python from: $VENV_PYTHON"

# --- Step 1: Data Preprocessing ---
echo "⏳ [1/4] Running Data Preprocessing (preprocess.py)..."
$VENV_PYTHON $SRC_DIR/preprocess.py
echo "✅ [1/4] Data Preprocessing Complete."

# --- Step 2: Feature Engineering ---
echo "⏳ [2/4] Running Feature Engineering (features.py)..."
$VENV_PYTHON $SRC_DIR/features.py
echo "✅ [2/4] Feature Engineering Complete."

# --- Step 3: Model Training ---
echo "⏳ [3/4] Running Model Training (train.py)..."
$VENV_PYTHON $SRC_DIR/train.py
echo "✅ [3.5/4] Model Training Complete." # Renumbered for clarity

# --- Step 4: Model Explanation ---
echo "⏳ [4/4] Running Model Explanation (explain.py)..."
$VENV_PYTHON $SRC_DIR/explain.py
echo "✅ [4/4] Model Explanation Complete."

echo "--- ✅ Smart Stays Pipeline Completed Successfully ---"

# Optional: Add command to launch Streamlit dashboard at the end
# echo "🚀 Launching Streamlit Dashboard..."
# Note: The Streamlit app path also needs to be corrected
# streamlit run ../dashboard/app.py

exit 0
