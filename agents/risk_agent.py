"""
ORCA Risk Analysis Agent
Synthesizes environmental, biological, and machine learning signals to compute
a multi-dimensional ecological risk matrix and isolate actionable threats.
"""

from typing import Dict, Any, List
from agents.base_agent import BaseAgent, AgentResult, AgentFinding


class RiskAgent(BaseAgent):
    """Evaluates multi-stressor compounding vulnerabilities and assigns risk levels."""

    def __init__(self):
        super().__init__(agent_id="risk_analysis", display_name="Ecological Risk Analysis Agent")

    def run(self, context: Dict[str, Any]) -> AgentResult:
        prior_agents = context.get("agent_outputs", {})
        env = context.get("environmental", {})
        
        do = float(env.get("dissolved_oxygen", 5.0))
        temp = float(env.get("temperature", 14.0))
        chla = float(env.get("chlorophyll", 1.5))
        depth = float(env.get("depth", 10.0))
        
        ml_out = prior_agents.get("ml_analysis", {})
        ml_pred = ml_out.get("metrics", {}).get("prediction", "Healthy")
        is_anomaly = ml_out.get("metrics", {}).get("is_anomaly", False)

        findings: List[AgentFinding] = []
        risk_vectors: Dict[str, str] = {}
        risk_scores: Dict[str, float] = {}

        # 1. Hypoxia / Anoxia Risk Vector
        if do < 2.0:
            risk_vectors["hypoxia_risk"] = "CRITICAL / SEVERE"
            risk_scores["hypoxia"] = 0.95
            findings.append(AgentFinding(
                indicator="hypoxia_threat",
                observed_value=f"{do:.2f} ml/L",
                baseline_reference=">= 4.5 ml/L",
                status="critical",
                description="Severe dead-zone condition. High probability of benthic invertebrate asphyxiation and fish displacement.",
                confidence=0.98
            ))
        elif do < 4.0:
            risk_vectors["hypoxia_risk"] = "MODERATE"
            risk_scores["hypoxia"] = 0.50
            findings.append(AgentFinding(
                indicator="hypoxia_threat",
                observed_value=f"{do:.2f} ml/L",
                baseline_reference=">= 4.5 ml/L",
                status="warning",
                description="Sub-surface oxygen depression. Increased physiological vulnerability during nocturnal respiration.",
                confidence=0.90
            ))
        else:
            risk_vectors["hypoxia_risk"] = "MINIMAL"
            risk_scores["hypoxia"] = 0.10

        # 2. Thermal Stress / Marine Heatwave Risk
        if temp > 21.0:
            risk_vectors["thermal_risk"] = "ELEVATED"
            risk_scores["thermal"] = 0.75
            findings.append(AgentFinding(
                indicator="thermal_stress_threat",
                observed_value=f"{temp:.1f} °C",
                baseline_reference="< 18.0 °C",
                status="warning",
                description="Elevated thermal profile indicates heatwave stress, suppressing vertical mixing and kelp canopy health.",
                confidence=0.88
            ))
        elif temp > 18.5:
            risk_vectors["thermal_risk"] = "MILD"
            risk_scores["thermal"] = 0.40
        else:
            risk_vectors["thermal_risk"] = "MINIMAL"
            risk_scores["thermal"] = 0.10

        # 3. Eutrophication & Bloom Senescence Risk
        if chla > 8.0:
            risk_vectors["eutrophication_risk"] = "ELEVATED"
            risk_scores["eutrophication"] = 0.70
            findings.append(AgentFinding(
                indicator="eutrophication_threat",
                observed_value=f"{chla:.2f} μg/L",
                baseline_reference="< 4.0 μg/L",
                status="warning",
                description="Algal biomass accumulation risks subsequent bacterial oxygen draw-down upon bloom collapse.",
                confidence=0.85
            ))
        else:
            risk_vectors["eutrophication_risk"] = "MINIMAL"
            risk_scores["eutrophication"] = 0.15

        # 4. Compounding Multivariate Risk Calculation
        composite_risk_score = (
            0.45 * risk_scores["hypoxia"] +
            0.25 * risk_scores["thermal"] +
            0.20 * risk_scores["eutrophication"] +
            (0.10 if is_anomaly else 0.0)
        )
        composite_risk_score = round(min(1.0, composite_risk_score), 3)

        if composite_risk_score >= 0.65 or do < 2.0:
            overall_risk = "HIGH / SEVERE"
            severity = "critical"
            status = "alert"
        elif composite_risk_score >= 0.35 or ml_pred == "Moderate Stress":
            overall_risk = "MODERATE"
            severity = "moderate"
            status = "warning"
        else:
            overall_risk = "LOW / NOMINAL"
            severity = "low"
            status = "completed"

        findings.append(AgentFinding(
            indicator="composite_ecological_risk",
            observed_value=f"{overall_risk} (Score: {composite_risk_score:.2f})",
            baseline_reference="Low Risk (< 0.35)",
            status="critical" if severity == "critical" else ("warning" if severity == "moderate" else "optimal"),
            description=f"Compounded risk synthesis across oxygen, temperature, chlorophyll, and statistical anomaly indicators.",
            confidence=0.92
        ))

        summary = (
            f"Overall Ecosystem Risk Level: {overall_risk} (Composite Score: {composite_risk_score:.2f}). "
            f"Primary Threat: {max(risk_scores, key=risk_scores.get).capitalize()}."
        )

        recommendations = [
            "Establish automated sentinel alerts if dissolved oxygen drops below 2.5 ml/L.",
            "Integrate benthic lander data to verify bottom-water hypoxia extent."
        ]
        if "HIGH" in overall_risk:
            recommendations.append("Alert regional coastal fisheries managers to prospective dead-zone avoidance shifts.")

        return AgentResult(
            agent_id=self.agent_id,
            display_name=self.display_name,
            status=status,
            severity=severity,
            confidence=0.92,
            summary=summary,
            findings=findings,
            metrics={
                "overall_risk_level": overall_risk,
                "composite_risk_score": composite_risk_score,
                "risk_vectors": risk_vectors,
                "individual_risk_scores": risk_scores
            },
            recommendations=recommendations
        )
