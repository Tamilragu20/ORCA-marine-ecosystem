"""
ORCA FastAPI Endpoints Definition
Exposes clean RESTful APIs for multi-agent reasoning, ML prediction,
data exploration, mapping, and report retrieval.
"""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
import uuid
import json
import io
import pandas as pd
from typing import Dict, Any, List
from sqlalchemy.orm import Session

from backend.schemas.schemas import (
    EnvironmentalInput, PredictionResponse, AnalysisResponse,
    AgentInfo, DataSummaryResponse, StationRecordDTO, SpeciesRecordDTO
)
from backend.services.marine_service import get_marine_service
from orchestration.workflow import get_orchestrator
from ml.predict import get_predictor
from database.db import get_db, AnalysisSession, UploadedDataset

router = APIRouter(prefix="/api", tags=["ORCA Marine Intelligence API"])


@router.get("/health")
def health_check():
    """System liveness and component readiness check."""
    pred = get_predictor()
    orch = get_orchestrator()
    service = get_marine_service()
    
    return {
        "status": "healthy",
        "service": "ORCA Marine Ecosystem Platform",
        "model_loaded": pred.model is not None,
        "champion_model": pred.metadata.get("best_model_name", "Gradient Boosting"),
        "orchestrator_ready": orch is not None,
        "database_connected": True,
        "datasets_loaded": len(service._df) > 0
    }


@router.get("/agents", response_model=List[AgentInfo])
def list_agents():
    """Returns documentation and status of all specialized collaborating agents."""
    return [
        AgentInfo(
            agent_id="water_quality",
            display_name="Water Quality Agent",
            role="Physical & Chemical Oceanography",
            responsibilities=[
                "Analyze in-situ temperature, salinity, and dissolved oxygen",
                "Evaluate oceanic nutrient stoichiometry (N:P Redfield ratio)",
                "Identify sub-surface hypoxia boundaries and marine heatwaves"
            ],
            status="active"
        ),
        AgentInfo(
            agent_id="marine_biodiversity",
            display_name="Marine Life & Biodiversity Agent",
            role="Community Ecology & Taxonomic Diversity",
            responsibilities=[
                "Catalog local marine species from UNESCO OBIS occurrences",
                "Compute species richness and Shannon-Wiener diversity index",
                "Evaluate faunal physiological vulnerability under oxygen stress"
            ],
            status="active"
        ),
        AgentInfo(
            agent_id="ocean_productivity",
            display_name="Ocean Productivity Agent",
            role="Phytoplankton & Trophic State Dynamics",
            responsibilities=[
                "Analyze chlorophyll-a concentration and phaeopigment ratio",
                "Classify ocean trophic state (Oligotrophic, Mesotrophic, Eutrophic)",
                "Model light attenuation and primary productivity depth proxies"
            ],
            status="active"
        ),
        AgentInfo(
            agent_id="ml_analysis",
            display_name="Machine Learning Analysis Agent",
            role="Predictive Inference & Anomaly Detection",
            responsibilities=[
                "Execute trained Gradient Boosting classifier across 17 features",
                "Calculate multi-class posterior probability distributions",
                "Run Isolation Forest for multivariate oceanographic anomaly detection"
            ],
            status="active"
        ),
        AgentInfo(
            agent_id="risk_analysis",
            display_name="Ecological Risk Analysis Agent",
            role="Multi-Stressor Threat Synthesis",
            responsibilities=[
                "Synthesize hypoxia, thermal stress, and bloom collapse risks",
                "Compute composite ecological risk score (0.0 to 1.0)",
                "Distinguish empirical observations from predictive interpretations"
            ],
            status="active"
        ),
        AgentInfo(
            agent_id="coordinator_reasoning",
            display_name="Coordinator & Reasoning Agent",
            role="Central Cognitive Arbiter & Epistemic Demarcation",
            responsibilities=[
                "Compare multi-agent evidence and detect cross-agent agreements",
                "Resolve conflicting signals transparently using scientific precaution",
                "Enforce epistemic demarcation across data, models, heuristics, and caveats"
            ],
            status="active"
        ),
        AgentInfo(
            agent_id="report_generator",
            display_name="Executive Report Agent",
            role="Formal Scientific Documentation",
            responsibilities=[
                "Assemble comprehensive 14-section ecosystem assessment report",
                "Format outputs into structured Markdown, HTML, and JSON representations",
                "Compile actionable recommendations for environmental monitoring"
            ],
            status="active"
        )
    ]


@router.post("/predict", response_model=PredictionResponse)
def predict_ecosystem(input_data: EnvironmentalInput):
    """Direct high-speed machine learning inference endpoint."""
    predictor = get_predictor()
    try:
        res = predictor.predict(input_data.model_dump())
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@router.post("/analyze", response_model=AnalysisResponse)
def analyze_ecosystem(input_data: EnvironmentalInput, db: Session = Depends(get_db)):
    """Executes the full collaborative multi-agent reasoning pipeline and logs session."""
    orchestrator = get_orchestrator()
    session_id = f"orca-{uuid.uuid4().hex[:10]}"
    
    try:
        orch_res = orchestrator.run_pipeline(input_data.model_dump(), session_id=session_id)
        
        # Persist session in SQLite
        db_session = AnalysisSession(
            session_id=session_id,
            latitude=input_data.latitude,
            longitude=input_data.longitude,
            depth=input_data.depth,
            temperature=input_data.temperature,
            salinity=input_data.salinity,
            dissolved_oxygen=input_data.dissolved_oxygen,
            chlorophyll=input_data.chlorophyll,
            final_assessment=orch_res.final_assessment,
            final_confidence=orch_res.final_confidence,
            overall_risk_level=orch_res.agent_results.get("risk_analysis", {}).get("metrics", {}).get("overall_risk_level", "LOW"),
            is_anomaly=str(orch_res.agent_results.get("ml_analysis", {}).get("metrics", {}).get("is_anomaly", False)),
            execution_time_ms=orch_res.total_execution_time_ms,
            report_markdown=orch_res.report_markdown,
            agent_outputs_json=json.dumps(orch_res.agent_results)
        )
        db.add(db_session)
        db.commit()

        return orch_res
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Multi-agent reasoning error: {str(e)}")


@router.get("/data-summary", response_model=DataSummaryResponse)
def data_summary():
    """Returns official dataset statistics, variables, and provenance."""
    service = get_marine_service()
    return service.get_summary()


@router.get("/trends")
def ocean_trends():
    """Returns multi-decadal time-series trends from CalCOFI hydrography."""
    service = get_marine_service()
    return service.get_trends()


@router.get("/stations", response_model=List[StationRecordDTO])
def map_stations(limit: int = 500):
    """Returns real CalCOFI hydrographic station points for interactive mapping."""
    service = get_marine_service()
    return service.get_stations(limit=limit)


@router.get("/species")
def biodiversity_species(limit: int = 300):
    """Returns real OBIS marine species occurrences for biodiversity explorer."""
    service = get_marine_service()
    return service.get_species(limit=limit)


@router.get("/biodiversity-stats")
def biodiversity_stats():
    """Returns taxonomic distributions across phyla, classes, and top species."""
    service = get_marine_service()
    return service.get_biodiversity_stats()


@router.get("/reports")
def list_reports(limit: int = 20, db: Session = Depends(get_db)):
    """Lists recently executed assessment sessions."""
    sessions = db.query(AnalysisSession).order_by(AnalysisSession.timestamp.desc()).limit(limit).all()
    return [
        {
            "session_id": s.session_id,
            "timestamp": s.timestamp.isoformat() if s.timestamp else "",
            "coordinates": f"{s.latitude:.2f}°N, {s.longitude:.2f}°W" if s.latitude else "N/A",
            "depth": f"{s.depth:.1f} m" if s.depth else "N/A",
            "temperature": f"{s.temperature:.1f} °C" if s.temperature else "N/A",
            "dissolved_oxygen": f"{s.dissolved_oxygen:.2f} ml/L" if s.dissolved_oxygen else "N/A",
            "final_assessment": s.final_assessment,
            "confidence": f"{s.final_confidence * 100:.1f}%",
            "risk_level": s.overall_risk_level,
            "execution_time_ms": s.execution_time_ms
        }
        for s in sessions
    ]


@router.get("/report/{session_id}")
def get_report(session_id: str, db: Session = Depends(get_db)):
    """Fetches the complete markdown report and full agent payloads for a session."""
    session = db.query(AnalysisSession).filter(AnalysisSession.session_id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    agent_outputs = json.loads(session.agent_outputs_json) if session.agent_outputs_json else {}
    return {
        "session_id": session.session_id,
        "timestamp": session.timestamp.isoformat() if session.timestamp else "",
        "final_assessment": session.final_assessment,
        "final_confidence": session.final_confidence,
        "overall_risk_level": session.overall_risk_level,
        "report_markdown": session.report_markdown,
        "agent_outputs": agent_outputs
    }


@router.post("/upload")
async def upload_csv(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Allows user to upload a custom marine environmental CSV dataset."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    
    contents = await file.read()
    try:
        df = pd.read_csv(io.StringIO(contents.decode("utf-8", errors="ignore")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")

    record_count = len(df)
    columns = list(df.columns)
    
    # Run batch predictions on up to 50 rows
    predictor = get_predictor()
    batch_results = []
    for idx, row in df.head(50).iterrows():
        try:
            row_dict = row.to_dict()
            pred = predictor.predict(row_dict)
            batch_results.append({
                "row_index": idx + 1,
                "prediction": pred["prediction"],
                "confidence": pred["confidence"],
                "hypoxia_alert": pred["hypoxia_alert"]
            })
        except Exception:
            continue

    # Record in database
    uploaded = UploadedDataset(
        filename=file.filename,
        record_count=record_count,
        columns_json=json.dumps(columns),
        summary_json=json.dumps({"processed_sample_size": len(batch_results)})
    )
    db.add(uploaded)
    db.commit()

    return {
        "status": "success",
        "filename": file.filename,
        "total_records": record_count,
        "columns_found": columns,
        "sample_evaluations": batch_results
    }
