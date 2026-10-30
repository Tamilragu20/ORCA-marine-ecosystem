"""
ORCA Test Suite: Workflow & Multi-Agent Collaboration Orchestration
"""

import pytest
from orchestration.workflow import get_orchestrator


def test_orchestration_pipeline_healthy():
    orchestrator = get_orchestrator()
    sample_input = {
        "depth": 15.0,
        "temperature": 15.0,
        "salinity": 33.5,
        "dissolved_oxygen": 6.0,
        "chlorophyll": 1.6,
        "phosphate": 0.6,
        "nitrate": 6.0,
        "silicate": 7.0,
        "latitude": 34.2,
        "longitude": -119.8,
        "month": 6
    }
    result = orchestrator.run_pipeline(sample_input)
    assert result.overall_status == "SUCCESS"
    assert len(result.pipeline_steps) == 7
    assert "Healthy" in result.final_assessment
    assert "water_quality" in result.agent_results
    assert "coordinator_reasoning" in result.agent_results
    assert "report_generator" in result.agent_results
    assert len(result.report_markdown) > 500


def test_orchestration_pipeline_hypoxic():
    orchestrator = get_orchestrator()
    hypoxic_input = {
        "depth": 150.0,
        "temperature": 8.5,
        "salinity": 34.2,
        "dissolved_oxygen": 1.2, # Dead zone
        "chlorophyll": 0.05,
        "phosphate": 2.4,
        "nitrate": 28.0,
        "silicate": 35.0,
        "latitude": 33.5,
        "longitude": -119.0,
        "month": 7
    }
    result = orchestrator.run_pipeline(hypoxic_input)
    assert result.overall_status == "SUCCESS"
    assert "High Stress" in result.final_assessment
    assert result.agent_results["water_quality"]["severity"] == "critical"
