"""
Pytest configuration for ORCA test suite.
"""

import sys
import os

# Insert root directory into sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
