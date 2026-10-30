"""
ORCA Exploratory Data Analysis (EDA) Pipeline
Generates scientific figures and statistical summaries for marine oceanographic
and biodiversity observations, saving charts to reports/figures/.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Plot aesthetics for oceanographic theme
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["figure.titlesize"] = 14

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "orca_integrated_ecosystem.csv")
OBIS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "obis_cleaned_biodiversity.csv")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def run_eda():
    print(f"Loading integrated dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    obis_df = pd.read_csv(OBIS_PATH)
    print(f"Dataset loaded: {df.shape[0]} records, {df.shape[1]} features.")

    # -------------------------------------------------------------
    # 1. Oceanographic Variable Distributions
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Temperature
    sns.histplot(df["temperature"], kde=True, ax=axes[0, 0], color="#0284c7", bins=35)
    axes[0, 0].set_title("In-Situ Water Temperature (°C)")
    axes[0, 0].set_xlabel("Temperature (°C)")
    axes[0, 0].axvline(df["temperature"].median(), color="#0f172a", linestyle="--", label=f"Median: {df['temperature'].median():.1f}°C")
    axes[0, 0].legend()

    # Salinity
    sns.histplot(df["salinity"], kde=True, ax=axes[0, 1], color="#0d9488", bins=35)
    axes[0, 1].set_title("Practical Salinity (PSU)")
    axes[0, 1].set_xlabel("Salinity (PSU)")
    axes[0, 1].axvline(df["salinity"].median(), color="#0f172a", linestyle="--", label=f"Median: {df['salinity'].median():.2f} PSU")
    axes[0, 1].legend()

    # Dissolved Oxygen
    sns.histplot(df["dissolved_oxygen"], kde=True, ax=axes[1, 0], color="#2563eb", bins=35)
    axes[1, 0].set_title("Dissolved Oxygen (ml/L)")
    axes[1, 0].set_xlabel("Dissolved Oxygen (ml/L)")
    axes[1, 0].axvline(2.0, color="#ef4444", linestyle=":", linewidth=2, label="Hypoxia Threshold (2.0 ml/L)")
    axes[1, 0].axvline(df["dissolved_oxygen"].median(), color="#0f172a", linestyle="--", label=f"Median: {df['dissolved_oxygen'].median():.2f} ml/L")
    axes[1, 0].legend()

    # Chlorophyll-a
    sns.histplot(df["chlorophyll"].clip(0, 10), kde=True, ax=axes[1, 1], color="#10b981", bins=35)
    axes[1, 1].set_title("Chlorophyll-a (μg/L)")
    axes[1, 1].set_xlabel("Chlorophyll-a (μg/L)")
    axes[1, 1].axvline(df["chlorophyll"].median(), color="#0f172a", linestyle="--", label=f"Median: {df['chlorophyll'].median():.2f} μg/L")
    axes[1, 1].legend()

    plt.suptitle("ORCA: Key Oceanographic Indicator Distributions (CalCOFI)", y=0.98)
    plt.tight_layout()
    p1 = os.path.join(FIG_DIR, "01_water_quality_distributions.png")
    plt.savefig(p1, dpi=200)
    plt.close()
    print(f"Saved: {p1}")

    # -------------------------------------------------------------
    # 2. Correlation Matrix
    # -------------------------------------------------------------
    corr_cols = [
        "temperature", "salinity", "dissolved_oxygen", "chlorophyll", 
        "phosphate", "nitrate", "depth", "ecological_stress_index", 
        "species_richness", "shannon_diversity"
    ]
    corr = df[corr_cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title("Oceanographic & Biodiversity Indicator Correlation Matrix", pad=15)
    plt.tight_layout()
    p2 = os.path.join(FIG_DIR, "02_correlation_matrix.png")
    plt.savefig(p2, dpi=200)
    plt.close()
    print(f"Saved: {p2}")

    # -------------------------------------------------------------
    # 3. Geographic Distribution Map (CalCOFI Stations + OBIS)
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 8))
    # Subsample for clear scatter rendering
    sample_df = df.sample(min(3000, len(df)), random_state=42)
    scatter = plt.scatter(
        sample_df["longitude"], sample_df["latitude"],
        c=sample_df["ecological_stress_index"], cmap="plasma",
        alpha=0.6, s=18, edgecolors="none"
    )
    cbar = plt.colorbar(scatter)
    cbar.set_label("Ecological Stress Index (0 = Low, 1 = High)")
    
    # Overlay OBIS samples
    obis_sample = obis_df.sample(min(800, len(obis_df)), random_state=42)
    plt.scatter(
        obis_sample["decimalLongitude"], obis_sample["decimalLatitude"],
        color="#10b981", s=10, alpha=0.5, marker="^", label="OBIS Biodiversity Observations"
    )
    plt.title("Geographic Distribution: California Current Hydrographic Stations & Biodiversity", pad=12)
    plt.xlabel("Longitude (°W)")
    plt.ylabel("Latitude (°N)")
    plt.legend(loc="lower left")
    plt.tight_layout()
    p3 = os.path.join(FIG_DIR, "03_geographic_stations_map.png")
    plt.savefig(p3, dpi=200)
    plt.close()
    print(f"Saved: {p3}")

    # -------------------------------------------------------------
    # 4. Temporal Trends (2000 - 2021)
    # -------------------------------------------------------------
    yearly = df.groupby("year").agg({
        "temperature": "mean",
        "dissolved_oxygen": "mean",
        "chlorophyll": "mean",
        "ecological_stress_index": "mean"
    }).reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0, 0].plot(yearly["year"], yearly["temperature"], marker="o", color="#e11d48", linewidth=2)
    axes[0, 0].set_title("Mean Annual Water Temperature (°C)")
    axes[0, 0].set_xlabel("Year")
    
    axes[0, 1].plot(yearly["year"], yearly["dissolved_oxygen"], marker="s", color="#0284c7", linewidth=2)
    axes[0, 1].set_title("Mean Annual Dissolved Oxygen (ml/L)")
    axes[0, 1].set_xlabel("Year")

    axes[1, 0].plot(yearly["year"], yearly["chlorophyll"], marker="^", color="#059669", linewidth=2)
    axes[1, 0].set_title("Mean Annual Chlorophyll-a (μg/L)")
    axes[1, 0].set_xlabel("Year")

    axes[1, 1].plot(yearly["year"], yearly["ecological_stress_index"], marker="d", color="#d97706", linewidth=2)
    axes[1, 1].set_title("Mean Annual Ecological Stress Index")
    axes[1, 1].set_xlabel("Year")

    plt.suptitle("ORCA: California Current Multi-Decadal Temporal Trends (2000–2021)", y=0.98)
    plt.tight_layout()
    p4 = os.path.join(FIG_DIR, "04_temporal_trends.png")
    plt.savefig(p4, dpi=200)
    plt.close()
    print(f"Saved: {p4}")

    # -------------------------------------------------------------
    # 5. OBIS Marine Taxonomic Breakdown
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    top_phyla = obis_df["phylum"].value_counts().head(8)
    sns.barplot(x=top_phyla.values, y=top_phyla.index, palette="mako")
    plt.title("OBIS Marine Biodiversity: Observations by Taxonomic Phylum", pad=12)
    plt.xlabel("Number of Validated Occurrences")
    plt.ylabel("Phylum")
    plt.tight_layout()
    p5 = os.path.join(FIG_DIR, "05_biodiversity_taxa_breakdown.png")
    plt.savefig(p5, dpi=200)
    plt.close()
    print(f"Saved: {p5}")

    # -------------------------------------------------------------
    # 6. Ecosystem Condition Class Distribution
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    cond_counts = df["ecosystem_condition_label"].value_counts()
    colors = ["#10b981", "#f59e0b", "#ef4444"]
    plt.pie(cond_counts, labels=cond_counts.index, autopct="%1.1f%%", startangle=140, 
            colors=colors, explode=(0.04, 0.04, 0.04), textprops={"fontsize": 11, "weight": "bold"})
    plt.title("ORCA Grounded Target: Marine Ecosystem Condition Classes", pad=15)
    plt.tight_layout()
    p6 = os.path.join(FIG_DIR, "06_ecosystem_condition_breakdown.png")
    plt.savefig(p6, dpi=200)
    plt.close()
    print(f"Saved: {p6}")

    # -------------------------------------------------------------
    # 7. Oceanographic Depth Profiles
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 7), sharey=True)
    depth_sample = df.sample(min(2500, len(df)), random_state=42)

    # Temp vs Depth
    axes[0].scatter(depth_sample["temperature"], depth_sample["depth"], alpha=0.25, color="#0284c7", s=10)
    axes[0].set_title("Thermocline Profile (Temp vs Depth)")
    axes[0].set_xlabel("Temperature (°C)")
    axes[0].set_ylabel("Depth (meters)")
    axes[0].invert_yaxis()

    # O2 vs Depth (Oxycline)
    axes[1].scatter(depth_sample["dissolved_oxygen"], depth_sample["depth"], alpha=0.25, color="#0d9488", s=10)
    axes[1].axvline(2.0, color="#ef4444", linestyle="--", label="Hypoxic Boundary (<2.0)")
    axes[1].set_title("Oxycline Profile (Dissolved O2 vs Depth)")
    axes[1].set_xlabel("Dissolved Oxygen (ml/L)")
    axes[1].legend()

    # Salinity vs Depth (Halocline)
    axes[2].scatter(depth_sample["salinity"], depth_sample["depth"], alpha=0.25, color="#6366f1", s=10)
    axes[2].set_title("Halocline Profile (Salinity vs Depth)")
    axes[2].set_xlabel("Practical Salinity (PSU)")

    plt.suptitle("ORCA: Water Column Vertical Structure (Epipelagic to Mesopelagic)", y=0.98)
    plt.tight_layout()
    p7 = os.path.join(FIG_DIR, "07_depth_profiles.png")
    plt.savefig(p7, dpi=200)
    plt.close()
    print(f"Saved: {p7}")

    # -------------------------------------------------------------
    # 8. Sensor Completeness
    # -------------------------------------------------------------
    key_vars = ["depth", "temperature", "salinity", "dissolved_oxygen", "chlorophyll", "phosphate", "nitrate", "silicate"]
    completeness = (1.0 - df[key_vars].isna().mean()) * 100.0

    plt.figure(figsize=(10, 5))
    sns.barplot(x=completeness.values, y=completeness.index, palette="crest")
    plt.xlim(0, 105)
    plt.title("Sensor Completeness After Quality Assurance Pipeline (%)", pad=12)
    plt.xlabel("Data Completeness (%)")
    plt.ylabel("Oceanographic Indicator")
    for i, v in enumerate(completeness.values):
        plt.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=10, weight="bold")
    plt.tight_layout()
    p8 = os.path.join(FIG_DIR, "08_data_completeness.png")
    plt.savefig(p8, dpi=200)
    plt.close()
    print(f"Saved: {p8}")

    print("\nEDA figure generation completed successfully! All 8 figures saved in reports/figures/.")


if __name__ == "__main__":
    run_eda()
