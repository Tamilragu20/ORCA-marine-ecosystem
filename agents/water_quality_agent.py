"""
ORCA Water Quality Agent
Specializes in physical and chemical hydrography: temperature, salinity,
dissolved oxygen saturation, and nutrient stoichiometry.
"""

from typing import Dict, Any, List
import numpy as np
from agents.base_agent import BaseAgent, AgentResult, AgentFinding


class WaterQualityAgent(BaseAgent):
    """Analyzes marine hydrographic and physicochemical properties."""

    def __init__(self):
        super().__init__(agent_id="water_quality", display_name="Water Quality Agent")

    def run(self, context: Dict[str, Any]) -> AgentResult:
        env = context.get("environmental", {})
        temp = float(env.get("temperature", 14.0))
        sal = float(env.get("salinity", 33.4))
        do = float(env.get("dissolved_oxygen", 5.0))
        depth = float(env.get("depth", 10.0))
        no3 = float(env.get("nitrate", 8.0))
        po4 = float(env.get("phosphate", 0.7))
        sio3 = float(env.get("silicate", 8.0))

        findings: List[AgentFinding] = []
        severity = "low"
        status = "completed"

        # 1. Dissolved Oxygen & Hypoxia Evaluation
        # Diaz & Rosenberg 2008 / Vaquer-Sunyer & Duarte 2008 benchmarks
        if do < 2.0:
            severity = "critical"
            status = "alert"
            findings.append(AgentFinding(
                indicator="dissolved_oxygen",
                observed_value=f"{do:.2f} ml/L",
                baseline_reference=">= 4.5 ml/L (Normoxic threshold)",
                status="critical",
                description="Severe coastal hypoxia detected (< 2.0 ml/L). Extreme threat to finfish and benthic invertebrates.",
                confidence=0.98
            ))
        elif do < 4.5:
            if severity != "critical":
                severity = "moderate"
            findings.append(AgentFinding(
                indicator="dissolved_oxygen",
                observed_value=f"{do:.2f} ml/L",
                baseline_reference="4.5 - 7.5 ml/L",
                status="warning",
                description="Moderate oxygen depression observed (2.0 - 4.5 ml/L). Sub-surface metabolic stress likely in active species.",
                confidence=0.92
            ))
        else:
            findings.append(AgentFinding(
                indicator="dissolved_oxygen",
                observed_value=f"{do:.2f} ml/L",
                baseline_reference=">= 4.5 ml/L",
                status="optimal",
                description="Well-oxygenated water column supporting optimal aerobic marine respiration.",
                confidence=0.95
            ))

        # 2. Thermal Profile & Anomaly
        if temp > 21.0 and depth < 50.0:
            if severity == "low":
                severity = "moderate"
            findings.append(AgentFinding(
                indicator="temperature",
                observed_value=f"{temp:.1f} °C",
                baseline_reference="11.0 - 17.5 °C (Temperate California Current surface)",
                status="warning",
                description="Elevated sea surface temperature indicative of potential thermal stratification or marine heatwave conditions.",
                confidence=0.88
            ))
        elif temp < 7.0 and depth < 30.0:
            findings.append(AgentFinding(
                indicator="temperature",
                observed_value=f"{temp:.1f} °C",
                baseline_reference="11.0 - 17.5 °C",
                status="informative",
                description="Strong coastal upwelling cold-water signature detected in upper column.",
                confidence=0.90
            ))
        else:
            findings.append(AgentFinding(
                indicator="temperature",
                observed_value=f"{temp:.1f} °C",
                baseline_reference="9.0 - 18.0 °C",
                status="optimal",
                description="Temperature falls within expected seasonal range for this depth horizon.",
                confidence=0.95
            ))

        # 3. Salinity Consistency
        if sal < 31.5:
            findings.append(AgentFinding(
                indicator="salinity",
                observed_value=f"{sal:.2f} PSU",
                baseline_reference="32.8 - 34.2 PSU",
                status="warning",
                description="Sub-normative salinity observed, suggesting freshwater river plume intrusion or heavy localized runoff.",
                confidence=0.85
            ))
        elif sal > 34.6:
            findings.append(AgentFinding(
                indicator="salinity",
                observed_value=f"{sal:.2f} PSU",
                baseline_reference="32.8 - 34.2 PSU",
                status="informative",
                description="Elevated salinity indicating deep ocean water mass upwelling or high evaporation.",
                confidence=0.90
            ))
        else:
            findings.append(AgentFinding(
                indicator="salinity",
                observed_value=f"{sal:.2f} PSU",
                baseline_reference="32.8 - 34.2 PSU",
                status="optimal",
                description="Salinity conforms to standard Pacific coastal seawater baselines.",
                confidence=0.96
            ))

        # 4. Stoichiometry & Nutrients (Redfield Ratio N:P 16:1)
        np_ratio = (no3 + 0.01) / (po4 + 0.01)
        if np_ratio < 8.0 and depth < 100:
            findings.append(AgentFinding(
                indicator="np_ratio",
                observed_value=f"{np_ratio:.1f}:1",
                baseline_reference="16:1 (Redfield Stoichiometry)",
                status="warning",
                description="Nitrate limitation relative to phosphate (N:P < 8:1); potential constraint on diatom growth.",
                confidence=0.86
            ))
        elif np_ratio > 30.0:
            findings.append(AgentFinding(
                indicator="np_ratio",
                observed_value=f"{np_ratio:.1f}:1",
                baseline_reference="16:1",
                status="warning",
                description="Excess nitrate ratio observed (N:P > 30:1); typical of agricultural runoff or intense deep nitrification.",
                confidence=0.84
            ))
        else:
            findings.append(AgentFinding(
                indicator="np_ratio",
                observed_value=f"{np_ratio:.1f}:1",
                baseline_reference="16:1",
                status="optimal",
                description="Nutrient stoichiometry conforms to standard oceanographic Redfield proportions.",
                confidence=0.92
            ))

        # Summary statement
        if severity == "critical":
            summary = "Critical water quality alert: Acute dissolved oxygen depletion (< 2.0 ml/L) constitutes severe hypoxia."
        elif severity == "moderate":
            summary = "Moderate water quality deviation: Sub-surface oxygen depression or thermal/nutrient divergence observed."
        else:
            summary = "Water quality parameters are optimal: Well-oxygenated, normative thermal envelope, and balanced stoichiometry."

        recommendations = []
        if do < 4.0:
            recommendations.append("Deploy high-frequency dissolved oxygen sonde profiling to track oxycline migration.")
        if temp > 20.0:
            recommendations.append("Monitor sea surface temperature anomalies via satellite SST to evaluate marine heatwave persistence.")
        if np_ratio > 25.0:
            recommendations.append("Sample nearshore runoff transects to evaluate terrestrial nitrogen discharge.")

        return AgentResult(
            agent_id=self.agent_id,
            display_name=self.display_name,
            status=status,
            severity=severity,
            confidence=0.94,
            summary=summary,
            findings=findings,
            metrics={
                "temperature_degC": temp,
                "salinity_psu": sal,
                "dissolved_oxygen_ml_L": do,
                "depth_m": depth,
                "np_ratio": round(np_ratio, 2),
                "hypoxia_risk": bool(do < 2.0)
            },
            recommendations=recommendations
        )
