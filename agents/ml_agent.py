"""
ORCA Machine Learning Analysis Agent
Interfaces directly with the serialized Gradient Boosting, Isolation Forest,
and explainability engines, formatting predictions into structured agent findings.
"""

from typing import Dict, Any, List
from agents.base_agent import BaseAgent, AgentResult, AgentFinding
from ml.predict import get_predictor


class MLAgent(BaseAgent):
    """Executes trained statistical models to provide data-driven ecosystem state inference."""

    def __init__(self):
        super().__init__(agent_id="ml_analysis", display_name="Machine Learning Analysis Agent")
        self.predictor = get_predictor()

    def run(self, context: Dict[str, Any]) -> AgentResult:
        env = context.get("environmental", {})
        
        # Run inference
        pred_res = self.predictor.predict(env)

        predicted_class = pred_res["prediction"]
        confidence = pred_res["confidence"]
        probabilities = pred_res["probabilities"]
        is_anomaly = pred_res["is_anomaly"]
        anomaly_score = pred_res["anomaly_score"]
        explanation = pred_res.get("explanation", {})

        findings: List[AgentFinding] = []
        severity = "low"
        status = "completed"

        if predicted_class == "High Stress":
            severity = "critical"
            status = "alert"
        elif predicted_class == "Moderate Stress":
            severity = "moderate"
            status = "warning"

        # 1. Primary Classification Finding
        findings.append(AgentFinding(
            indicator="ml_ecosystem_condition",
            observed_value=f"{predicted_class} (Confidence: {confidence*100:.1f}%)",
            baseline_reference="Healthy Ecosystem",
            status="critical" if severity == "critical" else ("warning" if severity == "moderate" else "optimal"),
            description=f"Gradient Boosting classifier determined ecosystem condition to be '{predicted_class}' based on trained 17-feature hydrographic patterns.",
            confidence=confidence
        ))

        # 2. Probability Distribution Finding
        findings.append(AgentFinding(
            indicator="class_probabilities",
            observed_value=f"Healthy: {probabilities.get('Healthy', 0)*100:.1f}%, Mod: {probabilities.get('Moderate Stress', 0)*100:.1f}%, High: {probabilities.get('High Stress', 0)*100:.1f}%",
            status="informative",
            description="Normalized posterior class probabilities across multi-class decision boundaries.",
            confidence=confidence
        ))

        # 3. Anomaly Detection (Isolation Forest)
        if is_anomaly:
            if severity == "low":
                severity = "moderate"
            findings.append(AgentFinding(
                indicator="isolation_forest_anomaly",
                observed_value=f"Anomaly Flagged (Score: {anomaly_score})",
                baseline_reference="Normative ocean state (> 0.0)",
                status="warning",
                description="Isolation Forest identified multivariate hydrographic outlier conditions deviating from typical historical baseline clusters.",
                confidence=0.88
            ))

        # 4. Top Feature Attributions
        for f in explanation.get("top_contributing_factors", [])[:3]:
            findings.append(AgentFinding(
                indicator=f["feature"],
                observed_value=f"{f['observed_value']} {f['unit']}",
                baseline_reference=f["normative_range"],
                status="warning" if "CRITICAL" in f["status"] or "HIGH" in f["status"] else "informative",
                description=f"{f['display_name']} ({f['impact']}): {f['scientific_rationale']}",
                confidence=0.90
            ))

        summary = (
            f"Machine Learning model predicts: '{predicted_class}' with {confidence*100:.1f}% confidence. "
            f"Isolation Forest classifies sample as {'anomalous' if is_anomaly else 'normative'} (score: {anomaly_score})."
        )

        recommendations = [
            "Validate ML output against direct in-situ sensor telemetry before triggering automated marine interventions.",
            "Cross-reference probability distributions with seasonal upwelling index to filter transient physical dynamics."
        ]

        return AgentResult(
            agent_id=self.agent_id,
            display_name=self.display_name,
            status=status,
            severity=severity,
            confidence=confidence,
            summary=summary,
            findings=findings,
            metrics={
                "prediction": predicted_class,
                "confidence": confidence,
                "probabilities": probabilities,
                "is_anomaly": is_anomaly,
                "anomaly_score": anomaly_score,
                "top_factors": explanation.get("top_contributing_factors", [])
            },
            recommendations=recommendations
        )
