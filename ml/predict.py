"""
ORCA Machine Learning Inference Engine
Provides high-performance inference, probability calibration, anomaly scoring,
and integrates with the explainability subsystem.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Tuple, List

from ml.explain import MarineExplainer

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


class MarinePredictor:
    """Manages trained ML models and performs end-to-end inference and explanation."""

    CLASS_MAP = {0: "Healthy", 1: "Moderate Stress", 2: "High Stress"}

    def __init__(self):
        self.model_path = os.path.join(MODELS_DIR, "orca_classifier.joblib")
        self.scaler_path = os.path.join(MODELS_DIR, "orca_scaler.joblib")
        self.iso_path = os.path.join(MODELS_DIR, "anomaly_detector.joblib")
        self.meta_path = os.path.join(MODELS_DIR, "model_metadata.json")

        self.model = None
        self.scaler = None
        self.iso_forest = None
        self.metadata = {}
        self.explainer = None
        self._load_artifacts()

    def _load_artifacts(self):
        # Ensure scikit-learn Cython loss extensions are properly mapped in sys.modules
        # across all environments and scikit-learn versions to prevent ModuleNotFoundError: No module named '_loss'
        try:
            import sklearn._loss._loss as _c_loss
            if "_loss" not in sys.modules:
                sys.modules["_loss"] = _c_loss
        except (ImportError, AttributeError):
            pass

        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            if os.path.exists(self.iso_path):
                self.iso_forest = joblib.load(self.iso_path)
            if os.path.exists(self.meta_path):
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            self.explainer = MarineExplainer()
        else:
            print("Warning: Model artifacts not found. Please train models first using ml/train.py.")

    def prepare_features(self, input_data: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, float]]:
        """Extracts required 17 features, imputing oceanographic auxiliary features if omitted."""
        depth = float(input_data.get("depth", 10.0))
        temp = float(input_data.get("temperature", 13.5))
        sal = float(input_data.get("salinity", 33.4))
        do = float(input_data.get("dissolved_oxygen", 5.2))
        chla = float(input_data.get("chlorophyll", 1.2))
        po4 = float(input_data.get("phosphate", 0.75))
        no3 = float(input_data.get("nitrate", 8.5))
        sio3 = float(input_data.get("silicate", 9.0))
        month = int(input_data.get("month", 6))

        # Computed features
        temp_depth_gradient = temp / (depth + 10.0)
        sigma_theta = (sal * 0.8) - (temp * 0.18) + (depth * 0.004)
        o2_sat_ratio = float(input_data.get("oxygen_saturation", (do / 5.5) * 100.0))
        np_ratio = (no3 + 0.01) / (po4 + 0.01)
        prod_proxy = chla * np.exp(-depth / 45.0)
        seasonal_sin = np.sin(2 * np.pi * month / 12.0)
        seasonal_cos = np.cos(2 * np.pi * month / 12.0)
        richness = float(input_data.get("species_richness", 22.0))
        shannon = float(input_data.get("shannon_diversity", 2.35))
        lat = float(input_data.get("latitude", 34.0))
        lon = float(input_data.get("longitude", -120.0))

        feature_dict = {
            "depth": depth,
            "temperature": temp,
            "salinity": sal,
            "dissolved_oxygen": do,
            "chlorophyll": chla,
            "phosphate": po4,
            "nitrate": no3,
            "silicate": sio3,
            "temp_depth_gradient": temp_depth_gradient,
            "sigma_theta_proxy": sigma_theta,
            "o2_saturation_ratio": o2_sat_ratio,
            "np_ratio": np_ratio,
            "primary_productivity_proxy": prod_proxy,
            "seasonal_sin": seasonal_sin,
            "seasonal_cos": seasonal_cos,
            "species_richness": richness,
            "shannon_diversity": shannon,
            "latitude": lat,
            "longitude": lon
        }

        feature_order = self.metadata.get("features", list(feature_dict.keys()))
        vec = np.array([feature_dict[col] for col in feature_order]).reshape(1, -1)
        return vec, feature_dict

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Runs full prediction pipeline with probabilities, anomalies, and explanation."""
        if self.model is None or self.scaler is None:
            self._load_artifacts()
            if self.model is None:
                raise RuntimeError("Models not loaded. Run ml/train.py first.")

        vec, feat_dict = self.prepare_features(input_data)
        vec_scaled = self.scaler.transform(vec)

        # Classifier prediction
        pred_class_idx = int(self.model.predict(vec_scaled)[0])
        pred_class_label = self.CLASS_MAP.get(pred_class_idx, "Unknown")

        # Class probabilities
        probs = self.model.predict_proba(vec_scaled)[0]
        prob_dict = {
            self.CLASS_MAP[i]: round(float(probs[i]), 4)
            for i in range(len(probs))
        }

        # Anomaly scoring
        is_anomaly = False
        anomaly_score = 0.5
        if self.iso_forest is not None:
            # -1 for anomaly, 1 for normal
            raw_iso = self.iso_forest.predict(vec_scaled)[0]
            is_anomaly = bool(raw_iso == -1)
            # Isolation score: lower = more abnormal
            raw_score = self.iso_forest.decision_function(vec_scaled)[0]
            anomaly_score = round(float(raw_score), 4)

        # Explainability
        explanation = self.explainer.explain_prediction(feat_dict, pred_class_label, prob_dict)

        # Critical alert flags
        hypoxia_flag = feat_dict["dissolved_oxygen"] < 2.0
        thermal_flag = feat_dict["temperature"] > 21.0

        return {
            "prediction": pred_class_label,
            "prediction_code": pred_class_idx,
            "confidence": round(float(np.max(probs)), 4),
            "probabilities": prob_dict,
            "is_anomaly": is_anomaly,
            "anomaly_score": anomaly_score,
            "hypoxia_alert": hypoxia_flag,
            "thermal_stress_alert": thermal_flag,
            "features_used": feat_dict,
            "explanation": explanation
        }


# Singleton predictor instance for fast module imports
predictor_instance = None

def get_predictor() -> MarinePredictor:
    global predictor_instance
    if predictor_instance is None:
        predictor_instance = MarinePredictor()
    return predictor_instance
