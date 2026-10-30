"""
ORCA Master Evaluation Script
Computes test metrics, confusion matrix, ROC curves, and generates reports.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.evaluate import evaluate_orca_models

if __name__ == "__main__":
    evaluate_orca_models()
