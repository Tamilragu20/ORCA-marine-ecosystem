"""
ORCA Pydantic Schemas for API Requests and Responses
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional


class EnvironmentalInput(BaseModel):
    """Input payload for single oceanographic profile observation."""
    depth: float = Field(default=15.0, ge=0.0, le=1000.0, description="Depth in meters")
    temperature: float = Field(default=14.5, ge=-2.0, le=35.0, description="Water temperature in deg C")
    salinity: float = Field(default=33.5, ge=25.0, le=40.0, description="Practical salinity in PSU")
    dissolved_oxygen: float = Field(default=5.4, ge=0.0, le=15.0, description="Dissolved oxygen in ml/L")
    chlorophyll: float = Field(default=1.5, ge=0.0, le=100.0, description="Chlorophyll-a in ug/L")
    phosphate: Optional[float] = Field(default=0.7, ge=0.0, le=10.0, description="Phosphate in umol/L")
    nitrate: Optional[float] = Field(default=7.5, ge=0.0, le=60.0, description="Nitrate in umol/L")
    silicate: Optional[float] = Field(default=8.0, ge=0.0, le=100.0, description="Silicate in umol/L")
    latitude: Optional[float] = Field(default=34.2, ge=30.0, le=40.0, description="Latitude in decimal degrees")
    longitude: Optional[float] = Field(default=-119.8, ge=-127.0, le=-116.0, description="Longitude in decimal degrees")
    month: Optional[int] = Field(default=6, ge=1, le=12, description="Month of observation (1-12)")


class PredictionResponse(BaseModel):
    prediction: str
    prediction_code: int
    confidence: float
    probabilities: Dict[str, float]
    is_anomaly: bool
    anomaly_score: float
    hypoxia_alert: bool
    thermal_stress_alert: bool
    features_used: Dict[str, float]
    explanation: Dict[str, Any]


class PipelineStepDTO(BaseModel):
    step_number: int
    agent_id: str
    display_name: str
    status: str
    severity: str
    execution_time_ms: float
    summary: str


class AnalysisResponse(BaseModel):
    session_id: str
    timestamp: str
    overall_status: str
    final_assessment: str
    final_confidence: float
    pipeline_steps: List[PipelineStepDTO]
    agent_results: Dict[str, Any]
    input_environmental: Dict[str, Any]
    report_markdown: str
    total_execution_time_ms: float


class AgentInfo(BaseModel):
    agent_id: str
    display_name: str
    role: str
    responsibilities: List[str]
    status: str


class DataSummaryResponse(BaseModel):
    total_environmental_records: int
    total_biodiversity_occurrences: int
    distinct_marine_species: int
    distinct_phyla: int
    study_region: str
    temporal_coverage: str
    key_variables: List[str]
    data_sources: List[Dict[str, str]]


class SpeciesRecordDTO(BaseModel):
    scientificName: str
    vernacularName: Optional[str] = None
    phylum: str
    taxa_class: Optional[str] = None
    decimalLatitude: float
    decimalLongitude: float
    depth: Optional[float] = None
    individualCount: Optional[Any] = None


class StationRecordDTO(BaseModel):
    station_id: int
    latitude: float
    longitude: float
    depth: float
    temperature: float
    salinity: float
    dissolved_oxygen: float
    chlorophyll: float
    ecosystem_condition: str
    stress_index: float
