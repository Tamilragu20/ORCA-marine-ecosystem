"""
ORCA Master Training Script
Trains all candidate ML algorithms, performs cross-validation, and serializes models.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.train import train_and_compare_models

if __name__ == "__main__":
    train_and_compare_models()
