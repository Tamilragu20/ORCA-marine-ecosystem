"""
ORCA Database Schema and SQLite Models
Stores analysis sessions, multi-agent outputs, and generated reports.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
import datetime
import os

DATABASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(DATABASE_DIR, "orca.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

Base = declarative_base()


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc)


class AnalysisSession(Base):
    __tablename__ = "analysis_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), unique=True, index=True, nullable=False)
    timestamp = Column(DateTime, default=utc_now)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    depth = Column(Float, nullable=True)
    temperature = Column(Float, nullable=True)
    salinity = Column(Float, nullable=True)
    dissolved_oxygen = Column(Float, nullable=True)
    chlorophyll = Column(Float, nullable=True)
    
    final_assessment = Column(String(128), nullable=False)
    final_confidence = Column(Float, nullable=False)
    overall_risk_level = Column(String(64), nullable=True)
    is_anomaly = Column(String(16), default="false")
    execution_time_ms = Column(Float, nullable=True)
    
    report_markdown = Column(Text, nullable=True)
    agent_outputs_json = Column(Text, nullable=True)


class UploadedDataset(Base):
    __tablename__ = "uploaded_datasets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(256), nullable=False)
    upload_time = Column(DateTime, default=utc_now)
    record_count = Column(Integer, nullable=False)
    columns_json = Column(Text, nullable=False)
    summary_json = Column(Text, nullable=True)


engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)
    print("ORCA SQLite database initialized successfully.")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
