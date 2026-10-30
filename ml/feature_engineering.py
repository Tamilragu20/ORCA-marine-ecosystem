"""
ORCA Feature Engineering & Dataset Integration Module
Performs oceanographic feature derivation, spatial-temporal gridding, 
OBIS biodiversity aggregation, and scientifically grounded ecological target derivation.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Tuple

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def compute_grid_biodiversity_metrics(obis_df: pd.DataFrame, grid_res: float = 0.25) -> pd.DataFrame:
    """
    Aggregates OBIS occurrences onto a 0.25-deg spatial grid, calculating species richness,
    Shannon-Wiener diversity index, and occurrence density.
    """
    obis = obis_df.copy()
    obis["grid_lat"] = np.round(obis["decimalLatitude"] / grid_res) * grid_res
    obis["grid_lon"] = np.round(obis["decimalLongitude"] / grid_res) * grid_res

    records = []
    for (glat, glon), group in obis.groupby(["grid_lat", "grid_lon"]):
        total_obs = len(group)
        species_counts = group["scientificName"].value_counts()
        richness = len(species_counts)
        
        # Shannon-Wiener Index: H' = - sum(p * ln(p))
        proportions = species_counts / total_obs
        shannon = -float(np.sum(proportions * np.log(proportions + 1e-9)))
        
        # Simpson Diversity: 1 - sum(p^2)
        simpson = 1.0 - float(np.sum(proportions ** 2))
        
        dominant_phylum = group["phylum"].mode().iloc[0] if not group["phylum"].dropna().empty else "Unknown"
        dominant_class = group["class"].mode().iloc[0] if not group["class"].dropna().empty else "Unknown"
        
        records.append({
            "grid_lat": glat,
            "grid_lon": glon,
            "species_richness": richness,
            "shannon_diversity": round(shannon, 3),
            "simpson_diversity": round(simpson, 3),
            "bio_occurrence_density": total_obs,
            "dominant_phylum": dominant_phylum,
            "dominant_class": dominant_class
        })

    grid_bio_df = pd.DataFrame(records)
    print(f"Generated biodiversity metrics across {len(grid_bio_df)} spatial grid cells.")
    return grid_bio_df


def engineer_marine_features(
    env_df: pd.DataFrame, 
    grid_bio_df: pd.DataFrame,
    grid_res: float = 0.25
) -> pd.DataFrame:
    """
    Synthesizes oceanographic features, matches with spatial biodiversity metrics,
    and calculates scientific ecosystem condition labels.
    """
    df = env_df.copy()

    # 1. Spatial Grid Coordinates
    df["grid_lat"] = np.round(df["latitude"] / grid_res) * grid_res
    df["grid_lon"] = np.round(df["longitude"] / grid_res) * grid_res

    # 2. Oceanographic & Physical Dynamic Features
    # Temperature-Depth gradient
    df["temp_depth_gradient"] = df["temperature"] / (df["depth"] + 10.0)
    
    # Potential density proxy (sigma-theta approx)
    df["sigma_theta_proxy"] = (df["salinity"] * 0.8) - (df["temperature"] * 0.18) + (df["depth"] * 0.004)
    
    # Apparent Oxygen Saturation ratio
    df["o2_saturation_ratio"] = df["oxygen_saturation"].fillna(
        (df["dissolved_oxygen"] / 5.5) * 100.0
    ).clip(0.0, 150.0)
    
    # Redfield N:P Ratio proxy (standard oceanic Redfield baseline is 16:1)
    df["np_ratio"] = (df["nitrate"] + 0.01) / (df["phosphate"] + 0.01)
    
    # Euphotic Primary Productivity proxy (Light attenuation with depth * Chlorophyll-a)
    df["primary_productivity_proxy"] = df["chlorophyll"] * np.exp(-df["depth"] / 45.0)

    # Seasonal cyclicity
    df["seasonal_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["seasonal_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

    # 3. Spatial Merge with OBIS Biodiversity Metrics
    # Exact grid match first
    df = pd.merge(df, grid_bio_df, on=["grid_lat", "grid_lon"], how="left")
    
    # For stations with no direct grid overlap, assign to geographically nearest OBIS census cell
    missing_mask = df["species_richness"].isna()
    if missing_mask.any() and not grid_bio_df.empty:
        grid_coords = grid_bio_df[["grid_lat", "grid_lon"]].values
        st_coords = df.loc[missing_mask, ["latitude", "longitude"]].values
        
        # Euclidean spatial distance in degree space
        dists = ((st_coords[:, None, :] - grid_coords[None, :, :]) ** 2).sum(axis=2)
        nearest_indices = np.argmin(dists, axis=1)
        
        for col in ["species_richness", "shannon_diversity", "simpson_diversity", "bio_occurrence_density", "dominant_phylum", "dominant_class"]:
            df.loc[missing_mask, col] = grid_bio_df[col].iloc[nearest_indices].values

    # 4. Scientifically Grounded Target Derivation
    # Scientific Hypoxia Criteria (Diaz & Rosenberg 2008; Vaquer-Sunyer & Duarte 2008):
    # - DO >= 4.5 ml/L: Normoxic / Healthy
    # - 2.0 <= DO < 4.5 ml/L: Moderate Oxygen Depression / Mesotrophic Stress
    # - DO < 2.0 ml/L: Severe Hypoxia / High Stress (lethal/sublethal to marine fauna)
    #
    # Thermal Stress: Surface/subsurface temperature anomaly relative to depth-seasonal mean
    temp_seasonal_mean = df.groupby(["month", pd.cut(df["depth"], bins=[0, 30, 100, 300, 600])])["temperature"].transform("mean")
    df["temp_anomaly"] = df["temperature"] - temp_seasonal_mean.fillna(df["temperature"].mean())

    # Multi-factor Ecological Stress Index (0.0 to 1.0)
    # Factor 1: Oxygen deficit (0 = well oxygenated > 5.5, 1 = hypoxic < 1.5)
    s_o2 = np.clip((5.5 - df["dissolved_oxygen"]) / 4.0, 0.0, 1.0)
    
    # Factor 2: Thermal deviation (|anomaly| > 2.5 C indicates heatwave or anomalous cold intrusion)
    s_temp = np.clip(np.abs(df["temp_anomaly"]) / 3.5, 0.0, 1.0)
    
    # Factor 3: Eutrophic/productivity imbalance (chlorophyll extremes)
    s_chl = np.clip((df["chlorophyll"] - 3.0) / 10.0, 0.0, 1.0)
    
    # Factor 4: Nutrient stress (deviation from Redfield 16:1)
    s_nut = np.clip(np.abs(df["np_ratio"] - 16.0) / 25.0, 0.0, 1.0)

    # Weighted composite index: O2 is the primary ecological determinant for marine life survival
    df["ecological_stress_index"] = (0.50 * s_o2) + (0.25 * s_temp) + (0.15 * s_chl) + (0.10 * s_nut)
    df["ecological_stress_index"] = df["ecological_stress_index"].round(4)

    # Categorical Ecosystem Condition Class:
    # 0: Healthy (DO >= 4.5 ml/L and stress index < 0.35)
    # 1: Moderate Stress (2.0 <= DO < 4.5 ml/L or 0.35 <= stress index < 0.65)
    # 2: High Stress (DO < 2.0 ml/L [hypoxia] or stress index >= 0.65)
    conditions = [
        (df["dissolved_oxygen"] < 2.0) | (df["ecological_stress_index"] >= 0.65),
        (df["dissolved_oxygen"] < 4.5) | (df["ecological_stress_index"] >= 0.35)
    ]
    choices = [2, 1] # 2 = High Stress, 1 = Moderate Stress, default = 0 (Healthy)
    df["ecosystem_condition"] = np.select(conditions, choices, default=0)

    class_names = {0: "Healthy", 1: "Moderate Stress", 2: "High Stress"}
    df["ecosystem_condition_label"] = df["ecosystem_condition"].map(class_names)

    # Specific sub-flags for explainability
    df["is_hypoxic"] = (df["dissolved_oxygen"] < 2.0).astype(int)
    df["is_thermal_stress"] = (df["temp_anomaly"].abs() > 2.0).astype(int)

    print("Target class distribution:")
    print(df["ecosystem_condition_label"].value_counts(normalize=True))

    return df


def build_and_save_integrated_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Orchestrates loading, cleaning, feature engineering, and saving final dataset."""
    from ml.preprocessing import load_and_clean_raw_datasets
    
    df_env, df_obis, _ = load_and_clean_raw_datasets(sample_size=25000)
    grid_bio = compute_grid_biodiversity_metrics(df_obis)
    df_integrated = engineer_marine_features(df_env, grid_bio)

    out_integrated = os.path.join(PROCESSED_DIR, "orca_integrated_ecosystem.csv")
    out_bio_grid = os.path.join(PROCESSED_DIR, "orca_spatial_biodiversity_grid.csv")

    df_integrated.to_csv(out_integrated, index=False)
    grid_bio.to_csv(out_bio_grid, index=False)
    print(f"Integrated dataset saved to: {out_integrated} ({len(df_integrated)} rows)")
    print(f"Spatial biodiversity grid saved to: {out_bio_grid}")

    return df_integrated, grid_bio


if __name__ == "__main__":
    build_and_save_integrated_dataset()
