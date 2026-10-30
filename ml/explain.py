"""
ORCA Explainability Module
Provides feature attribution, baseline deviation analysis, and scientific factor breakdown
to explain individual predictions transparently.
"""

import os
import json
import joblib
import numpy as np
from typing import Dict, Any, List

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


class MarineExplainer:
    """Explains model decisions using global feature importances and local baseline deviation."""
    
    # Established oceanographic normative baseline ranges for healthy temperate marine ecosystems
    NORMATIVE_RANGES = {
        "dissolved_oxygen": {"min": 4.5, "max": 8.0, "unit": "ml/L", "desc": "Dissolved Oxygen", "critical_low": 2.0},
        "temperature": {"min": 9.0, "max": 18.0, "unit": "°C", "desc": "Water Temperature", "critical_high": 22.0},
        "salinity": {"min": 32.8, "max": 34.5, "unit": "PSU", "desc": "Practical Salinity"},
        "chlorophyll": {"min": 0.2, "max": 4.0, "unit": "μg/L", "desc": "Chlorophyll-a", "critical_high": 12.0},
        "nitrate": {"min": 2.0, "max": 25.0, "unit": "μmol/L", "desc": "Nitrate Concentration"},
        "phosphate": {"min": 0.3, "max": 2.2, "unit": "μmol/L", "desc": "Phosphate Concentration"},
        "np_ratio": {"min": 10.0, "max": 20.0, "unit": "ratio", "desc": "N:P Redfield Ratio (16:1 baseline)"},
        "species_richness": {"min": 12, "max": 50, "unit": "species", "desc": "Local Species Richness"}
    }

    def __init__(self):
        meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
        self.metadata = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        self.global_importances = self.metadata.get("feature_importances", {})

    def explain_prediction(self, feature_dict: Dict[str, float], predicted_class: str, probabilities: Dict[str, float]) -> Dict[str, Any]:
        """
        Generates structured, human-readable explanations of why the model reached this assessment.
        """
        contributing_factors: List[Dict[str, Any]] = []

        # 1. Analyze deviations from normative scientific baselines
        for feat, norm in self.NORMATIVE_RANGES.items():
            if feat in feature_dict:
                val = feature_dict[feat]
                if val is None or np.isnan(val):
                    continue

                global_weight = self.global_importances.get(feat, 0.05)

                if "critical_low" in norm and val < norm["critical_low"]:
                    contributing_factors.append({
                        "feature": feat,
                        "display_name": norm["desc"],
                        "observed_value": round(val, 2),
                        "unit": norm["unit"],
                        "normative_range": f"{norm['min']} - {norm['max']} {norm['unit']}",
                        "status": "CRITICAL LOW (HYPOXIC)",
                        "impact": "Primary Stress Driver",
                        "weight": round(global_weight, 3),
                        "scientific_rationale": f"Observed {val:.2f} {norm['unit']} is below the critical hypoxia boundary (< 2.0 ml/L), severely impeding benthic and pelagic respiration."
                    })
                elif "critical_high" in norm and val > norm["critical_high"]:
                    contributing_factors.append({
                        "feature": feat,
                        "display_name": norm["desc"],
                        "observed_value": round(val, 2),
                        "unit": norm["unit"],
                        "normative_range": f"{norm['min']} - {norm['max']} {norm['unit']}",
                        "status": "ANOMALOUS HIGH",
                        "impact": "Stress Contributor",
                        "weight": round(global_weight, 3),
                        "scientific_rationale": f"Observed {val:.2f} {norm['unit']} exceeds normative threshold, suggesting thermal stress or elevated algal concentration."
                    })
                elif val < norm["min"]:
                    contributing_factors.append({
                        "feature": feat,
                        "display_name": norm["desc"],
                        "observed_value": round(val, 2),
                        "unit": norm["unit"],
                        "normative_range": f"{norm['min']} - {norm['max']} {norm['unit']}",
                        "status": "SUB-OPTIMAL LOW",
                        "impact": "Moderate Influence",
                        "weight": round(global_weight, 3),
                        "scientific_rationale": f"Measurement is below optimal baseline ({norm['min']} {norm['unit']})."
                    })
                elif val > norm["max"]:
                    contributing_factors.append({
                        "feature": feat,
                        "display_name": norm["desc"],
                        "observed_value": round(val, 2),
                        "unit": norm["unit"],
                        "normative_range": f"{norm['min']} - {norm['max']} {norm['unit']}",
                        "status": "ELEVATED",
                        "impact": "Moderate Influence",
                        "weight": round(global_weight, 3),
                        "scientific_rationale": f"Measurement is above normal envelope ({norm['max']} {norm['unit']})."
                    })

        # Sort factors by importance weight
        contributing_factors.sort(key=lambda x: x.get("weight", 0), reverse=True)

        # Synthesize plain-language scientific reasoning
        if predicted_class == "High Stress":
            summary = (
                "The model classified this ecosystem state as HIGH STRESS primarily due to acute "
                f"hydrographic stressors, most notably dissolved oxygen depletion ({feature_dict.get('dissolved_oxygen', 'N/A')} ml/L) "
                "or thermal-nutrient divergence from California Current baselines."
            )
        elif predicted_class == "Moderate Stress":
            summary = (
                "The ecosystem exhibits MODERATE STRESS, characterized by sub-surface oxygen depression or "
                "moderate thermal/nutrient shifts, while avoiding acute anoxia or collapse."
            )
        else:
            summary = (
                "The observations indicate a HEALTHY / OPTIMAL ecosystem state with dissolved oxygen "
                f"above normative baselines ({feature_dict.get('dissolved_oxygen', 'N/A')} ml/L) and balanced nutrient ratios."
            )

        return {
            "predicted_class": predicted_class,
            "prediction_confidence": max(probabilities.values()),
            "reasoning_summary": summary,
            "top_contributing_factors": contributing_factors[:5],
            "global_model_importances": dict(list(self.global_importances.items())[:6]),
            "uncertainty_note": (
                "Model-derived analytical output based on trained statistical patterns. "
                "Does not constitute an exhaustive field survey."
            )
        }
