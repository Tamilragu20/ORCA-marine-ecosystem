"""
ORCA Marine Data & Analysis Service
Handles data access, trends aggregation, species queries, and file processing.
"""

import os
import json
import uuid
import pandas as pd
import numpy as np
from typing import Dict, Any, List

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
INTEGRATED_PATH = os.path.join(DATA_DIR, "orca_integrated_ecosystem.csv")
OBIS_PATH = os.path.join(DATA_DIR, "obis_cleaned_biodiversity.csv")


class MarineDataService:
    def __init__(self):
        self._df = None
        self._obis_df = None
        self._load_data()

    def _load_data(self):
        if os.path.exists(INTEGRATED_PATH):
            self._df = pd.read_csv(INTEGRATED_PATH)
        else:
            self._df = pd.DataFrame()

        if os.path.exists(OBIS_PATH):
            self._obis_df = pd.read_csv(OBIS_PATH)
        else:
            self._obis_df = pd.DataFrame()

    def get_summary(self) -> Dict[str, Any]:
        """Returns high-level statistics for the ORCA platform."""
        total_env = len(self._df) if self._df is not None else 0
        total_bio = len(self._obis_df) if self._obis_df is not None else 0
        unique_species = int(self._obis_df["scientificName"].nunique()) if not self._obis_df.empty else 0
        unique_phyla = int(self._obis_df["phylum"].nunique()) if not self._obis_df.empty else 0

        return {
            "total_environmental_records": total_env,
            "total_biodiversity_occurrences": total_bio,
            "distinct_marine_species": unique_species,
            "distinct_phyla": unique_phyla,
            "study_region": "California Current Large Marine Ecosystem (31.0°N - 38.5°N, -125.0°W - -117.0°W)",
            "temporal_coverage": "2000 - 2021 (Modern High-Resolution Oceanography)",
            "key_variables": [
                "Depth (m)", "Temperature (°C)", "Practical Salinity (PSU)", 
                "Dissolved Oxygen (ml/L)", "Chlorophyll-a (μg/L)", 
                "Nitrate (μmol/L)", "Phosphate (μmol/L)", "Silicate (μmol/L)", 
                "Species Richness", "Shannon Biodiversity Index"
            ],
            "data_sources": [
                {
                    "name": "NOAA / Scripps CalCOFI Hydrographic Database",
                    "url": "https://calcofi.org/downloads/database/",
                    "type": "Oceanographic In-Situ Hydrography"
                },
                {
                    "name": "UNESCO / IOC Ocean Biodiversity Information System (OBIS)",
                    "url": "https://portal.obis.org/",
                    "type": "Marine Species Occurrences"
                }
            ]
        }

    def get_trends(self) -> Dict[str, Any]:
        """Returns annual trends from 2000 to 2021 for dashboard charts."""
        if self._df.empty or "year" not in self._df.columns:
            return {}

        yearly = self._df.groupby("year").agg({
            "temperature": "mean",
            "dissolved_oxygen": "mean",
            "chlorophyll": "mean",
            "salinity": "mean",
            "ecological_stress_index": "mean"
        }).reset_index()

        return {
            "years": yearly["year"].tolist(),
            "temperature": [round(float(v), 2) for v in yearly["temperature"]],
            "dissolved_oxygen": [round(float(v), 2) for v in yearly["dissolved_oxygen"]],
            "chlorophyll": [round(float(v), 2) for v in yearly["chlorophyll"]],
            "salinity": [round(float(v), 2) for v in yearly["salinity"]],
            "stress_index": [round(float(v), 3) for v in yearly["ecological_stress_index"]]
        }

    def get_stations(self, limit: int = 500) -> List[Dict[str, Any]]:
        """Returns representative oceanographic stations for interactive map visualization."""
        if self._df.empty:
            return []

        # Sample across conditions to show spatial diversity
        sample_df = self._df.sample(min(limit, len(self._df)), random_state=42)
        records = []
        for _, row in sample_df.iterrows():
            records.append({
                "station_id": int(row.get("Cst_Cnt", 0)),
                "latitude": round(float(row.get("latitude", 34.0)), 4),
                "longitude": round(float(row.get("longitude", -120.0)), 4),
                "depth": round(float(row.get("depth", 10.0)), 1),
                "temperature": round(float(row.get("temperature", 14.0)), 1),
                "salinity": round(float(row.get("salinity", 33.4)), 2),
                "dissolved_oxygen": round(float(row.get("dissolved_oxygen", 5.0)), 2),
                "chlorophyll": round(float(row.get("chlorophyll", 1.5)), 2),
                "ecosystem_condition": str(row.get("ecosystem_condition_label", "Healthy")),
                "stress_index": round(float(row.get("ecological_stress_index", 0.2)), 3)
            })
        return records

    def get_species(self, limit: int = 300) -> List[Dict[str, Any]]:
        """Returns OBIS species occurrences for biodiversity explorer and map overlay."""
        if self._obis_df.empty:
            return []

        sample_df = self._obis_df.sample(min(limit, len(self._obis_df)), random_state=42)
        records = []
        for _, row in sample_df.iterrows():
            records.append({
                "scientificName": str(row.get("scientificName", "Unknown")),
                "vernacularName": str(row.get("vernacularName", "")) if pd.notna(row.get("vernacularName")) else "",
                "phylum": str(row.get("phylum", "Marine Life")),
                "class": str(row.get("class", "")) if pd.notna(row.get("class")) else "",
                "order": str(row.get("order", "")) if pd.notna(row.get("order")) else "",
                "family": str(row.get("family", "")) if pd.notna(row.get("family")) else "",
                "decimalLatitude": round(float(row.get("decimalLatitude", 34.0)), 4),
                "decimalLongitude": round(float(row.get("decimalLongitude", -120.0)), 4),
                "depth": round(float(row.get("depth", 15.0)), 1) if pd.notna(row.get("depth")) else 10.0,
                "individualCount": str(row.get("individualCount", "1")) if pd.notna(row.get("individualCount")) else "1"
            })
        return records

    def get_biodiversity_stats(self) -> Dict[str, Any]:
        """Computes taxonomic distribution for charts."""
        if self._obis_df.empty:
            return {}

        phyla = self._obis_df["phylum"].value_counts().head(8).to_dict()
        classes = self._obis_df["class"].value_counts().head(8).to_dict()
        top_species = self._obis_df["scientificName"].value_counts().head(10).to_dict()

        return {
            "phyla": phyla,
            "classes": classes,
            "top_species": top_species,
            "total_occurrences": len(self._obis_df),
            "total_species": int(self._obis_df["scientificName"].nunique())
        }


# Singleton service
_marine_service = None

def get_marine_service() -> MarineDataService:
    global _marine_service
    if _marine_service is None:
        _marine_service = MarineDataService()
    return _marine_service
