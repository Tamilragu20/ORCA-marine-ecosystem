"""
ORCA Test Suite: Specialized Collaborative Agents
"""

import pytest
from agents.water_quality_agent import WaterQualityAgent
from agents.biodiversity_agent import BiodiversityAgent
from agents.productivity_agent import OceanProductivityAgent
from agents.ml_agent import MLAgent
from agents.risk_agent import RiskAgent
from agents.reasoning_agent import ReasoningAgent
from agents.report_agent import ReportAgent


def test_water_quality_agent():
    agent = WaterQualityAgent()
    context = {
        "environmental": {
            "temperature": 14.5,
            "salinity": 33.4,
            "dissolved_oxygen": 5.8,
            "depth": 10.0,
            "nitrate": 7.5,
            "phosphate": 0.6
        }
    }
    result = agent.execute(context)
    assert result.agent_id == "water_quality"
    assert result.status == "completed"
    assert len(result.findings) >= 3


def test_water_quality_agent_hypoxia():
    agent = WaterQualityAgent()
    context = {
        "environmental": {
            "temperature": 9.0,
            "salinity": 34.1,
            "dissolved_oxygen": 1.2, # Hypoxia
            "depth": 120.0
        }
    }
    result = agent.execute(context)
    assert result.severity == "critical"
    assert result.metrics["hypoxia_risk"] is True


def test_productivity_agent():
    agent = OceanProductivityAgent()
    context = {
        "environmental": {
            "chlorophyll": 2.2,
            "phaeopigments": 0.5,
            "depth": 15.0
        }
    }
    result = agent.execute(context)
    assert result.agent_id == "ocean_productivity"
    assert "Mesotrophic" in result.metrics["trophic_classification"]


def test_biodiversity_agent():
    agent = BiodiversityAgent()
    context = {
        "environmental": {
            "latitude": 34.0,
            "longitude": -120.0,
            "depth": 20.0,
            "dissolved_oxygen": 5.5
        }
    }
    result = agent.execute(context)
    assert result.agent_id == "marine_biodiversity"
    assert result.metrics["species_richness"] > 0
