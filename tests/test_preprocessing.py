"""
ORCA Test Suite: Data Preprocessing & Validation
"""

import pytest
import numpy as np
from ml.preprocessing import MarineDataValidator


def test_marine_data_validator_valid_input():
    valid_record = {
        "depth": 20.0,
        "temperature": 14.5,
        "salinity": 33.4,
        "dissolved_oxygen": 5.8,
        "oxygen_saturation": 95.0,
        "chlorophyll": 1.2,
        "latitude": 34.2,
        "longitude": -120.0
    }
    is_valid, issues = MarineDataValidator.validate_record(valid_record)
    assert is_valid is True
    assert len(issues) == 0


def test_marine_data_validator_out_of_bounds():
    invalid_record = {
        "depth": -5.0,           # impossible negative depth
        "temperature": 45.0,     # impossible ocean temp
        "salinity": 15.0,        # impossible open ocean salinity
        "dissolved_oxygen": 25.0 # impossible supersaturation
    }
    is_valid, issues = MarineDataValidator.validate_record(invalid_record)
    assert is_valid is False
    assert len(issues) >= 3
