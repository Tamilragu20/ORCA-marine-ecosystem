"""
ORCA Master Preprocessing & Feature Engineering Script
Runs data cleaning, validation checks, spatial biodiversity gridding, and creates final training data.
"""

import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.feature_engineering import build_and_save_integrated_dataset

if __name__ == "__main__":
    print("=" * 65)
    print("ORCA PIPELINE: PREPROCESSING & SCIENTIFIC DATA INTEGRATION")
    print("=" * 65)
    df_integrated, grid_bio = build_and_save_integrated_dataset()
    print("Preprocessing & Feature Engineering Completed Successfully!")
    print(f"Final Integrated Dataset Dimensions: {df_integrated.shape}")
    print(f"Columns: {list(df_integrated.columns)}")
