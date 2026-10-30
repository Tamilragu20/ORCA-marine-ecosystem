"""
ORCA Test Suite: Machine Learning & Explainability
"""

import pytest
from ml.predict import get_predictor


def test_predictor_healthy_sample():
    predictor = get_predictor()
    assert predictor.model is not None
    assert predictor.scaler is not None

    sample = {
        "depth": 15.0,
        "temperature": 15.2,
        "salinity": 33.5,
        "dissolved_oxygen": 6.1,
        "chlorophyll": 1.8,
        "phosphate": 0.6,
        "nitrate": 5.5,
        "silicate": 6.8,
        "month": 6
    }
    res = predictor.predict(sample)
    assert res["prediction"] in ["Healthy", "Moderate Stress", "High Stress"]
    assert res["prediction"] == "Healthy"
    assert 0.0 <= res["confidence"] <= 1.0
    assert "Healthy" in res["probabilities"]
    assert res["hypoxia_alert"] is False
    assert "explanation" in res
    assert "top_contributing_factors" in res["explanation"]


def test_predictor_hypoxic_sample():
    predictor = get_predictor()
    hypoxic_sample = {
        "depth": 140.0,
        "temperature": 8.5,
        "salinity": 34.2,
        "dissolved_oxygen": 1.1, # Severe hypoxia (< 2.0 ml/L)
        "chlorophyll": 0.02,
        "phosphate": 2.4,
        "nitrate": 30.0,
        "silicate": 40.0,
        "month": 7
    }
    res = predictor.predict(hypoxic_sample)
    assert res["prediction"] == "High Stress"
    assert res["hypoxia_alert"] is True
    assert res["probabilities"]["High Stress"] > 0.8
