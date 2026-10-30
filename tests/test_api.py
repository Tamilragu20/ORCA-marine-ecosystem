"""
ORCA Test Suite: FastAPI Endpoints & Integration
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_api_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["orchestrator_ready"] is True


def test_api_agents_list():
    res = client.get("/api/agents")
    assert res.status_code == 200
    agents = res.json()
    assert len(agents) == 7
    agent_ids = [a["agent_id"] for a in agents]
    assert "water_quality" in agent_ids
    assert "coordinator_reasoning" in agent_ids


def test_api_predict():
    payload = {
        "depth": 20.0,
        "temperature": 14.5,
        "salinity": 33.4,
        "dissolved_oxygen": 5.8,
        "chlorophyll": 1.5
    }
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "prediction" in data
    assert "probabilities" in data
    assert "confidence" in data
    assert "explanation" in data


def test_api_analyze():
    payload = {
        "depth": 15.0,
        "temperature": 15.0,
        "salinity": 33.5,
        "dissolved_oxygen": 6.0,
        "chlorophyll": 1.6,
        "latitude": 34.2,
        "longitude": -119.8
    }
    res = client.post("/api/analyze", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "final_assessment" in data
    assert "pipeline_steps" in data
    assert "agent_results" in data
    assert "report_markdown" in data


def test_api_data_summary():
    res = client.get("/api/data-summary")
    assert res.status_code == 200
    data = res.json()
    assert data["total_environmental_records"] > 0
    assert data["distinct_marine_species"] > 0


def test_api_stations():
    res = client.get("/api/stations?limit=50")
    assert res.status_code == 200
    stations = res.json()
    assert len(stations) > 0
    assert "latitude" in stations[0]
    assert "dissolved_oxygen" in stations[0]


def test_api_species():
    res = client.get("/api/species?limit=50")
    assert res.status_code == 200
    species = res.json()
    assert len(species) > 0
    assert "scientificName" in species[0]


def test_api_upload_csv():
    csv_content = "depth,temperature,salinity,dissolved_oxygen,chlorophyll\n10.0,14.5,33.4,5.5,1.2\n50.0,11.2,33.8,3.8,0.5\n"
    files = {"file": ("test_ocean.csv", csv_content, "text/csv")}
    res = client.post("/api/upload", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["total_records"] == 2
