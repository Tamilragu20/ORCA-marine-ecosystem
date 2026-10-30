"""
ORCA Ocean Productivity Agent
Specializes in phytoplankton dynamics, chlorophyll-a concentration,
trophic state classification, and euphotic primary production proxies.
"""

from typing import Dict, Any, List
import numpy as np
from agents.base_agent import BaseAgent, AgentResult, AgentFinding


class OceanProductivityAgent(BaseAgent):
    """Analyzes ocean primary productivity, phytoplankton chlorophyll-a, and bloom dynamics."""

    def __init__(self):
        super().__init__(agent_id="ocean_productivity", display_name="Ocean Productivity Agent")

    def run(self, context: Dict[str, Any]) -> AgentResult:
        env = context.get("environmental", {})
        chla = float(env.get("chlorophyll", 1.5))
        phaeop = float(env.get("phaeopigments", chla * 0.3))
        depth = float(env.get("depth", 10.0))
        no3 = float(env.get("nitrate", 8.0))
        do = float(env.get("dissolved_oxygen", 5.0))

        findings: List[AgentFinding] = []
        severity = "low"
        status = "completed"

        # 1. Trophic State & Chlorophyll Classification
        # Standard oceanographic classification (Carlson / coastal criteria)
        if chla > 10.0:
            severity = "moderate"
            trophic_state = "Hyper-Eutrophic / Algal Bloom"
            findings.append(AgentFinding(
                indicator="chlorophyll_concentration",
                observed_value=f"{chla:.2f} μg/L",
                baseline_reference="0.5 - 4.0 μg/L (Mesotrophic temperate baseline)",
                status="warning",
                description="Hyper-elevated chlorophyll-a indicating dense phytoplankton bloom or localized algal proliferation.",
                confidence=0.92
            ))
        elif chla >= 3.0:
            trophic_state = "Eutrophic / Highly Productive"
            findings.append(AgentFinding(
                indicator="chlorophyll_concentration",
                observed_value=f"{chla:.2f} μg/L",
                baseline_reference="0.5 - 4.0 μg/L",
                status="optimal",
                description="Elevated primary productivity characteristic of active coastal upwelling supporting abundant forage fish.",
                confidence=0.94
            ))
        elif chla >= 0.3:
            trophic_state = "Mesotrophic (Healthy / Balanced)"
            findings.append(AgentFinding(
                indicator="chlorophyll_concentration",
                observed_value=f"{chla:.2f} μg/L",
                baseline_reference="0.5 - 4.0 μg/L",
                status="optimal",
                description="Normative chlorophyll concentration maintaining healthy pelagic food web support.",
                confidence=0.96
            ))
        else:
            trophic_state = "Oligotrophic / Low Biomass"
            findings.append(AgentFinding(
                indicator="chlorophyll_concentration",
                observed_value=f"{chla:.2f} μg/L",
                baseline_reference="0.5 - 4.0 μg/L",
                status="informative",
                description="Low surface chlorophyll indicating nutrient-depleted, stratified oligotrophic waters.",
                confidence=0.88
            ))

        # 2. Phaeopigment / Senescence Ratio
        # High Phaeop relative to ChlorA indicates decaying bloom or intense zooplankton grazing
        phaeo_ratio = phaeop / (chla + 0.05)
        if phaeo_ratio > 1.2 and chla > 2.0:
            if severity == "low":
                severity = "moderate"
            findings.append(AgentFinding(
                indicator="phaeopigment_ratio",
                observed_value=f"{phaeo_ratio:.2f}",
                baseline_reference="< 0.8 (Active photosynthetic cells)",
                status="warning",
                description="High phaeopigment ratio reflects phytoplankton senescence, cell lysis, or intense microzooplankton grazing.",
                confidence=0.87
            ))

        # 3. Euphotic Zone Depth Context
        # Photosynthesis occurs predominantly in euphotic zone (depth < 100m)
        prod_proxy = chla * np.exp(-depth / 45.0)

        # 4. Bloom Decay & Hypoxia Linkage Risk
        if chla > 6.0 and do < 3.5:
            severity = "high"
            status = "alert"
            findings.append(AgentFinding(
                indicator="bloom_respiration_coupling",
                observed_value="High Chl-a + Low DO",
                baseline_reference="Decoupled or Normoxic",
                status="critical",
                description="Co-occurrence of elevated phytoplankton biomass with depressed oxygen suggests organic matter decomposition and bacterial respiration stress.",
                confidence=0.93
            ))

        summary = (
            f"Trophic classification: {trophic_state} (Chlorophyll-a = {chla:.2f} μg/L). "
            f"Euphotic productivity proxy: {prod_proxy:.3f}. Primary producer biomass is {'active and robust' if chla >= 0.5 else 'quiescent'}."
        )

        recommendations = [
            "Monitor satellite Ocean Color radiometry (MODIS-Aqua / Sentinel-3 OLCI) for regional bloom trajectory.",
            "Screen for toxic diatoms (e.g. Pseudo-nitzschia / domoic acid) or dinoflagellates via microscopic imaging."
        ]
        if chla > 8.0:
            recommendations.append("Track subsurface oxygen draw-down as post-bloom senescence initiates.")

        return AgentResult(
            agent_id=self.agent_id,
            display_name=self.display_name,
            status=status,
            severity=severity,
            confidence=0.91,
            summary=summary,
            findings=findings,
            metrics={
                "chlorophyll_ug_L": chla,
                "phaeopigments_ug_L": phaeop,
                "trophic_classification": trophic_state,
                "productivity_proxy": round(prod_proxy, 3),
                "depth_m": depth
            },
            recommendations=recommendations
        )
