# ORCA — Marine Ecosystem Reasoning with Collaborative Agents

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.8.0-orange.svg)](https://scikit-learn.org/)
[![License: Open Academic](https://img.shields.io/badge/license-CC--BY--4.0-green.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Tests Passing](https://img.shields.io/badge/pytest-18%20passed-brightgreen.svg)]()

> **ORCA** (*Marine Ecosystem Reasoning with Collaborative Agents*) is an advanced, production-grade AIML platform that synthesizes real-world oceanographic hydrography, machine learning predictive models, and seven specialized collaborating AI agents to evaluate coastal marine ecosystem conditions, detect hypoxia dead-zones, and provide transparent scientific reasoning.

---

## 1. Project Vision & Executive Summary

Marine ecosystems are facing unprecedented anthropogenic and climatic pressures, including sub-surface deoxygenation (hypoxia), marine heatwaves (MHW), ocean acidification, and altered trophic productivity. Traditional environmental monitoring platforms typically provide isolated sensor readings or single black-box ML predictions without contextual domain reasoning.

**ORCA** addresses this challenge through a multi-agent collaborative architecture. It couples **100% verified, real-world observational data** from the **NOAA / Scripps CalCOFI** database (2000–2021) and the **UNESCO/IOC Ocean Biodiversity Information System (OBIS)** with an interpretable machine learning pipeline and a strict **Four-Pillar Epistemic Demarcation** system that separates empirical sensor data, statistical predictions, domain deductions, and scientific uncertainty.

---

## 2. Multi-Agent Collaborative Architecture

Rather than functioning as independent chatbots, ORCA’s seven agents execute a linear, structured Directed Acyclic Graph (DAG) workflow exchanging standardized, typed JSON payloads (`AgentResult`):

```
                        In-Situ Oceanographic Observation
                                        │
                                        ▼
                             [Data Quality Validator]
                                        │
             ┌──────────────────────────┴──────────────────────────┐
             ▼                                                     ▼
    [Water Quality Agent]                               [Biodiversity Agent]
    • Temperature & Salinity                             • OBIS Taxonomic Census
    • Dissolved Oxygen Saturation                        • Species Richness
    • Stoichiometry (N:P Redfield)                       • Shannon Diversity (H')
             │                                                     │
             └──────────────────────────┬──────────────────────────┘
                                        ▼
                           [Ocean Productivity Agent]
                           • Chlorophyll-a Trophic State
                           • Phaeopigment Senescence Ratio
                           • Euphotic Depth Light Attenuation
                                        │
                                        ▼
                           [ML Analysis Agent]
                           • Gradient Boosting Inference (17 features)
                           • Multi-class Posterior Probabilities
                           • Isolation Forest Anomaly Detection
                                        │
                                        ▼
                          [Ecological Risk Agent]
                          • Hypoxia & Dead-Zone Risk
                          • Thermal Stratification & Heatwaves
                          • Compound Risk Matrix (0.0 to 1.0)
                                        │
                                        ▼
                       [Coordinator & Reasoning Agent]
                       • Cross-Agent Consensus Analysis
                       • Discrepancy & Conflict Resolution
                       • Four-Pillar Epistemic Demarcation
                                        │
                                        ▼
                           [Executive Report Agent]
                           • 14-Section Formal Assessment Report
                           • Markdown, HTML & JSON Compilation
```

### The Seven Specialized Agents

| # | Agent | Primary Focus | Key Oceanographic Output |
|---|---|---|---|
| 1 | **Water Quality Agent** | Physical & Chemical Hydrography | DO deficit, Potential density $\sigma_\theta$, N:P ratio |
| 2 | **Marine Life & Biodiversity Agent** | Community Ecology & Taxonomy | Species richness ($S$), Shannon index ($H'$), faunal vulnerability |
| 3 | **Ocean Productivity Agent** | Phytoplankton & Trophic Dynamics | Trophic state (Oligotrophic to Hyper-eutrophic), Phaeop ratio |
| 4 | **ML Analysis Agent** | Statistical Pattern Inference | Predicted condition, class probabilities, anomaly score |
| 5 | **Ecological Risk Agent** | Multi-Stressor Threat Analysis | Dead-zone risk, thermal stress index, composite risk |
| 6 | **Coordinator & Reasoning Agent** | Central Arbiter & Epistemic Demarcation | Agreement evaluation, conflict resolution, unified assessment |
| 7 | **Executive Report Agent** | Scientific Documentation | Formal 14-section ecosystem assessment report |

---

## 3. Real-World Datasets & Data Provenance

ORCA uses **only official scientific data** with complete provenance documented in [`data/SOURCES.md`](file:///c:/Users/hp/Desktop/Marine%20project/data/SOURCES.md):

1. **NOAA / Scripps CalCOFI Database (1949–2021)**:
   - **Source**: California Cooperative Oceanic Fisheries Investigations (NOAA Fisheries + Scripps Institution of Oceanography + California DFW).
   - **File**: `CalCOFI_Database_194903-202105_csv_16October2023.zip`.
   - **Modern Subset**: 2000–2021 high-resolution epipelagic and upper mesopelagic observations (25,000 verified profiles).
   - **Variables**: Depth, Temperature (°C), Salinity (PSU), Dissolved Oxygen (ml/L), Chlorophyll-a (μg/L), Phosphate (μmol/L), Nitrate (μmol/L), Silicate (μmol/L).
2. **UNESCO / IOC Ocean Biodiversity Information System (OBIS)**:
   - **Source**: Global marine biodiversity database coordinated by IOC/UNESCO.
   - **Study Polygon**: California Current Large Marine Ecosystem ($31.0^\circ\text{N} - 38.5^\circ\text{N}, -125.0^\circ\text{W} - -117.0^\circ\text{W}$).
   - **Occurrences**: 6,000 validated in-situ occurrence records spanning 360 distinct marine species across 24 biological phyla (Chordata, Ochrophyta, Echinodermata, Mollusca, Cnidaria, Arthropoda, etc.).

### Spatial-Temporal Integration Methodology
Data is unified via **0.25° × 0.25° Mesoscale Spatial-Temporal Gridding** (~25 km oceanographic bins):
- In-situ hydrographic bottle profiles are assigned to regional spatial grid cells.
- Regional biological metrics (species richness, Shannon-Wiener diversity $H'$, occurrence density) are calculated from OBIS occurrences within that grid cell.
- Raw biological records remain preserved for independent querying by the Biodiversity Agent.

---

## 4. Machine Learning Pipeline & Benchmarking

### Target Formulation & Leakage-Free Design
Ecosystem condition is categorized into three scientifically grounded, peer-reviewed ecological states:
- **0: Optimal / Healthy** ($\text{DO} \ge 4.5\text{ ml/L}$, normative temperature and balanced stoichiometry).
- **1: Moderate Stress** ($2.0 \le \text{DO} < 4.5\text{ ml/L}$, moderate thermal anomaly, or sub-surface oxygen depression).
- **2: High Stress / Hypoxic Threat** ($\text{DO} < 2.0\text{ ml/L}$ — Diaz & Rosenberg 2008 global marine dead-zone threshold).

> **Methodological Rigor & Target Leakage Prevention**: To eliminate target leakage and circular prediction, direct dissolved oxygen measurements and $O_2$ saturation ratio are strictly **excluded** from the predictive feature space. The model predicts the holistic ecosystem condition using 17 independent physical, chemical, spatial, temporal, and biological proxies (depth, temperature, salinity, chlorophyll-a, nutrients $NO_3, PO_4, SiO_3$, density proxy $\sigma_\theta$, vertical temperature gradient, coordinates, seasonality, and local OBIS biodiversity indices).

### Spatial-Temporal Validation Strategy
Random train/test splits introduce significant data leakage in oceanographic time-series due to spatial autocorrelation and multi-depth sampling from identical research cruises. ORCA implements a **Strict Temporal Forward-Holdout Split**:
- **Training Set (Years 2000–2016)**: 17,049 historical oceanographic profiles.
- **Validation Set (Years 2015–2016)**: 2,273 observations for hyperparameter tuning.
- **Temporal Holdout Test Set (Years 2017–2021)**: 5,678 observations from future cruises evaluated strictly out-of-time.
- Standard scaling is fit exclusively on the Training set and applied downstream to prevent distributional leakage.

### Real Evaluated Model Comparison Results (Temporal Holdout Test Set: 5,678 observations)

| Algorithm | 5-Fold CV Macro F1 | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC |
|---|---|---|---|---|---|---|
| **Gradient Boosting (Champion)** | **0.9690** | **95.44%** | **0.9594** | **0.9501** | **0.9542** | **0.9965** |
| Random Forest (150 estimators) | 0.9668 | 95.26% | 0.9576 | 0.9481 | 0.9522 | 0.9966 |
| Logistic Regression (scaled) | 0.9598 | 95.67% | 0.9594 | 0.9533 | 0.9561 | 0.9951 |
| Decision Tree (depth = 8) | 0.9580 | 94.49% | 0.9511 | 0.9414 | 0.9456 | 0.9849 |

### Top Predictive Hydrographic Drivers
Feature importance analysis from the champion Gradient Boosting model highlights the primary environmental predictors of marine ecosystem stress:
1. **Phosphate ($PO_4$)** (47.15%): Indicator of deep upwelling and nutrient enrichment.
2. **Potential Density Anomaly ($\sigma_\theta$ proxy)** (34.68%): Physical pycnocline stratification barrier.
3. **Nitrate ($NO_3$)** (5.90%): Macronutrient availability governing primary productivity.
4. **Salinity** (4.74%): Water mass identification and halocline positioning.
5. **Vertical Temperature Gradient** (2.64%): Thermocline stability impeding atmospheric gas exchange.

### Unsupervised Modeling
- **Isolation Forest**: Contamination rate = 0.05. Evaluates multivariate oceanographic anomalies and outlier hydrographic states.
- **K-Means Clustering ($k=3$)**: Identifies natural water column strata (Surface Epipelagic, Thermocline Transition, Deep Sub-surface Mesopelagic).

---

## 5. Epistemic Demarcation & Reasoning Safety

To prevent hallucination and unscientific causal claims, ORCA enforces the **Four Scientific Pillars**:
1. **Observed Empirical Data**: Direct, verified measurements from calibrated NOAA/Scripps CTD sensors.
2. **Machine Learning Predictions**: Statistical boundary mapping trained across 20 years of real oceanographic profiles.
3. **Agent Heuristics & Deductions**: Biochemical and stoichiometric reasoning (Redfield ratios, hypoxia tolerances).
4. **Scientific Uncertainty & Limitations**: Explicit disclosure of spatial resolution limits, sensor variance, and the distinction between statistical correlation and biological causation.

---

## 6. Web Application & Interactive Dashboard

The platform features an ocean-themed, dark-navy web interface (`http://localhost:8000`) comprising 10 dedicated functional views:

1. **Dashboard**: High-level KPI status cards, interactive visual agent pipeline, and real-time condition monitors.
2. **Ecosystem Analysis**: Interactive parameter controls (depth, temperature, salinity, DO, nutrients), preset real-world scenarios (Kelp Forest, Mesopelagic Depression, Dead Zone Basin, Marine Heatwave), and one-click multi-agent pipeline execution.
3. **Interactive Marine Map**: High-performance Leaflet.js map with CartoDB Dark Matter tiles, rendering 400+ CalCOFI stations color-coded by condition with click-to-load capabilities.
4. **Agent Insights**: Detailed activity cards for all 7 agents showing operational status, responsibilities, and sample findings.
5. **ML Prediction**: Interactive posterior probability bar charts, Isolation Forest anomaly meters, and model benchmarking tables.
6. **Biodiversity Explorer**: Doughnut charts of taxonomic phyla and searchable table of real California Current marine species.
7. **Ocean Trends**: 20-year multi-decadal time-series charts (2000–2021) for Temperature, Dissolved Oxygen, Chlorophyll-a, and Ecological Stress.
8. **Risk Analysis**: Multi-stressor threat matrix evaluating hypoxia, thermal heatwaves, and bloom collapse.
9. **Assessment Reports**: Full 14-section formal scientific assessment viewer with Markdown export, PDF print styling, and SQLite session history.
10. **Data Sources**: Provenance documentation, official URLs, and variable metadata.

---

## 7. Project Structure

```
Marine project/
├── data/
│   ├── raw/                              # Original CalCOFI zip & OBIS CSV records
│   ├── processed/                        # Cleaned, validated, and integrated datasets
│   └── SOURCES.md                        # Formal data provenance, citations, and licenses
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb# Documented Jupyter EDA notebook
├── ml/
│   ├── preprocessing.py                  # Ingestion, physical bounds screening, quality log
│   ├── feature_engineering.py            # Spatial gridding, biodiversity metrics, target derivation
│   ├── train.py                          # Multi-model training, 5-fold CV, model serialization
│   ├── evaluate.py                       # Evaluation metrics, confusion matrix, ROC curves
│   ├── predict.py                        # High-speed inference & anomaly detection engine
│   └── explain.py                        # Normative baseline deviation & feature attribution
├── models/
│   ├── orca_classifier.joblib            # Champion Gradient Boosting model
│   ├── orca_scaler.joblib                # Standard scaler fit strictly on Train
│   ├── anomaly_detector.joblib           # Trained Isolation Forest model
│   ├── clustering_model.joblib           # Trained K-Means model
│   └── model_metadata.json               # Model parameters, features, and importance weights
├── agents/
│   ├── base_agent.py                     # Standard AgentResult schema & BaseAgent class
│   ├── water_quality_agent.py            # Physical & chemical hydrography agent
│   ├── biodiversity_agent.py             # Marine taxonomy & community ecology agent
│   ├── productivity_agent.py             # Phytoplankton & trophic dynamics agent
│   ├── ml_agent.py                       # Predictive inference & explainability agent
│   ├── risk_agent.py                     # Multi-stressor threat matrix agent
│   ├── reasoning_agent.py                # Coordinator & epistemic demarcation agent
│   └── report_agent.py                   # 14-section formal report generator
├── orchestration/
│   └── workflow.py                       # Sequential multi-agent pipeline orchestrator
├── backend/
│   ├── main.py                           # FastAPI application entrypoint & static mounting
│   ├── api/
│   │   └── endpoints.py                  # Clean REST API routes (/analyze, /predict, etc.)
│   ├── schemas/
│   │   └── schemas.py                    # Pydantic request/response data contracts
│   └── services/
│       └── marine_service.py             # Data service layer for trends, stations, species
├── frontend/
│   ├── index.html                        # 11-section ocean-themed web dashboard
│   ├── css/
│   │   └── style.css                     # Glassmorphism, animations, responsive ocean styles
│   └── js/
│       └── app.js                        # Master frontend controller (Leaflet, Chart.js)
├── database/
│   ├── models.py                         # SQLAlchemy database models (AnalysisSession, Uploads)
│   └── orca.db                           # Local SQLite database
├── reports/
│   ├── data_quality_report.md            # Data cleaning & validation report
│   ├── model_evaluation_report.md        # Comprehensive ML benchmarking report
│   └── figures/                          # 12 Publication-quality figures (PNG)
│       ├── 01_water_quality_distributions.png
│       ├── 02_correlation_matrix.png
│       ├── 03_geographic_stations_map.png
│       ├── 04_temporal_trends.png
│       ├── 05_biodiversity_taxa_breakdown.png
│       ├── 06_ecosystem_condition_breakdown.png
│       ├── 07_depth_profiles.png
│       ├── 08_data_completeness.png
│       ├── 09_confusion_matrix.png
│       ├── 10_model_comparison_metrics.png
│       ├── 11_feature_importance.png
│       └── 12_roc_curves.png
├── tests/
│   ├── conftest.py                       # Pytest root configuration
│   ├── test_preprocessing.py             # Validation bounds tests
│   ├── test_ml.py                        # Model inference & explanation tests
│   ├── test_agents.py                    # Individual agent unit tests
│   ├── test_workflow.py                  # Multi-agent collaboration tests
│   └── test_api.py                       # FastAPI integration tests
├── scripts/
│   ├── download_data.py                  # Automated dataset acquisition script
│   ├── preprocess_data.py                # Preprocessing & integration script
│   ├── generate_eda.py                   # Automated figure generation script
│   ├── train_model.py                    # Model training script
│   └── evaluate_model.py                 # Evaluation & benchmarking script
├── requirements.txt                      # Project Python dependencies
├── .env.example                          # Environment configuration template
├── .gitignore                            # Git exclusion rules
└── README.md                             # Comprehensive project documentation
```

---

## 8. Step-by-Step Reproduction Guide

To run or completely rebuild the project from scratch on any standard laptop:

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: (Optional) Re-acquire Raw Data
```bash
python scripts/download_data.py
```

### Step 3: Run Preprocessing & Feature Engineering
```bash
python scripts/preprocess_data.py
```

### Step 4: Generate EDA Visualizations
```bash
python scripts/generate_eda.py
```

### Step 5: Train Machine Learning Models
```bash
python scripts/train_model.py
```

### Step 6: Evaluate Models
```bash
python scripts/evaluate_model.py
```

### Step 7: Run Unit & Integration Tests
```bash
pytest tests/ -v
```

### Step 8: Launch Web Application
```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
Open your browser and navigate to:
- **Web Dashboard**: `http://localhost:8000`
- **Interactive Swagger API Docs**: `http://localhost:8000/docs`

---

## 9. Viva Defense & Oral Examination Guide

### Q1: Why is this project not just a standard machine learning classifier?
> **Answer**: Standard ML models output single label probabilities without domain context. ORCA wraps predictive inference inside a multi-agent system where independent domain agents (Water Quality, Ocean Productivity, Biodiversity, Risk) evaluate empirical physical laws and oceanographic thresholds. The Coordinator Agent detects consensus or contradictions (e.g., high chlorophyll vs. low biodiversity) and enforces epistemic demarcation.

### Q2: What is the scientific basis for the "High Stress" label?
> **Answer**: The High Stress label is grounded in the globally accepted criterion for marine hypoxia defined by Diaz & Rosenberg (2008, *Science*): dissolved oxygen $< 2.0\text{ ml/L}$ ($\approx 2.8\text{ mg/L}$). Below this threshold, aerobic cellular respiration collapses for over 80% of teleost fish and benthic crustaceans.

### Q3: How do the agents avoid hallucinating causal claims?
> **Answer**: Under Pillar 4 of ORCA's Epistemic Demarcation protocol, the Coordinator Agent explicitly separates correlation from causation. The system states that low oxygen is *statistically associated* with high ecological stress in the model, rather than asserting unverified external causes such as agricultural pollution unless direct chemical runoff indicators are measured.

### Q4: How is data leakage prevented between CalCOFI and OBIS?
> **Answer**: Preprocessing standard scalers are fitted exclusively on the 80% training partition. Spatial-temporal gridding computes biodiversity indexes at the mesoscale grid level independently of the target condition labels, preventing holdout test set contamination.

### Q5: Why use Gradient Boosting instead of Deep Learning?
> **Answer**: Oceanographic hydrographic data is tabular and heterogeneous, with measurements spanning different physical units (°C, PSU, ml/L, μg/L). Gradient tree boosting provides superior performance on tabular data, requires no arbitrary hyperparameter scaling, trains in seconds on a standard 16 GB laptop, and produces exact tree feature importances for scientific explainability.

---

## 10. Limitations & Ethical Boundaries

1. **Regional Specificity**: Calibrated for the California Current Large Marine Ecosystem (temperate eastern boundary upwelling regime). Applying the model to tropical coral reefs or polar waters requires regional recalibration.
2. **Depth Distribution**: Sampling density is highest in the euphotic and epipelagic zones ($0 - 200\text{ m}$) where marine fisheries concentrate; deep mesopelagic observations ($> 500\text{ m}$) are naturally sparser.
3. **Correlation vs. Causation**: Feature importance scores represent statistical associations within historical observational profiles, not laboratory-isolated biochemical mechanisms.
4. **Operational Scope**: ORCA is developed as an academic, scientific, and research platform. It does not replace in-situ regulatory compliance testing.

---

## 11. Academic Citations

1. **CalCOFI**: California Cooperative Oceanic Fisheries Investigations (2023). *Hydrographic Database 1949–2021*. Scripps Institution of Oceanography & NOAA Southwest Fisheries Science Center. [https://calcofi.org](https://calcofi.org)
2. **OBIS**: Ocean Biodiversity Information System (2024). *Intergovernmental Oceanographic Commission of UNESCO*. [https://obis.org](https://obis.org)
3. **Hypoxia Benchmarks**: Diaz, R. J., & Rosenberg, R. (2008). Spreading dead zones and consequences for marine ecosystems. *Science*, 321(5891), 926–929.
4. **Marine Heatwaves**: Hobday, A. J., et al. (2016). A hierarchical approach to defining marine heatwaves. *Progress in Oceanography*, 141, 227–238.
5. **Redfield Ratio**: Redfield, A. C. (1958). The biological control of chemical factors in the environment. *American Scientist*, 46(3), 205–221.
