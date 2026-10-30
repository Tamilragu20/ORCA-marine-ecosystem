# ORCA Marine Ecosystem Data Provenance and Sources

This document provides full scientific provenance, citations, and data access specifications for all datasets incorporated into the **ORCA** (Marine Ecosystem Reasoning with Collaborative Agents) platform.

---

## 1. NOAA / Scripps CalCOFI Hydrographic & Oceanographic Database

- **Dataset Name**: California Cooperative Oceanic Fisheries Investigations (CalCOFI) Hydrographic Database (1949–2021)
- **Official Source URL**: [https://calcofi.org/downloads/database/](https://calcofi.org/downloads/database/)
- **Coordinating Organizations**: 
  - National Oceanic and Atmospheric Administration (NOAA) Fisheries (Southwest Fisheries Science Center)
  - Scripps Institution of Oceanography (University of California, San Diego)
  - California Department of Fish and Wildlife (CDFW)
- **Archive File**: `CalCOFI_Database_194903-202105_csv_16October2023.zip`
  - Cast Metadata: `194903-202105_Cast.csv`
  - Bottle Observations: `194903-202105_Bottle.csv`
- **Geographic Coverage**: California Current Marine Ecosystem (Lat: 31.0°N to 38.5°N, Lon: -125.0°W to -117.0°W)
- **Temporal Coverage**: 1949 to 2021 (Modern analytical focus: 2000–2021 for continuous multi-sensor oceanography)
- **Key Variables Used**:
  - `Lat_Dec`, `Lon_Dec`: Geographic coordinates (decimal degrees)
  - `Date`, `Year`, `Month`: Temporal markers
  - `Depthm`: Depth of observation (meters)
  - `T_degC`: In-situ water temperature (°C)
  - `Salnty`: Practical Salinity (PSU / PSS-78)
  - `O2ml_L`: Dissolved Oxygen (ml/L)
  - `O2Sat`: Dissolved Oxygen saturation percentage (%)
  - `ChlorA`: Chlorophyll-a concentration (μg/L)
  - `Phaeop`: Phaeopigments (μg/L)
  - `PO4uM`: Phosphate concentration (μmol/L)
  - `NO3uM`: Nitrate concentration (μmol/L)
  - `SiO3uM`: Silicate concentration (μmol/L)
- **Scientific License**: Open Access / US Government Public Domain & University of California Open Research.
- **Citation**: 
  > CalCOFI (2023). Hydrographic Database: 1949–2021. California Cooperative Oceanic Fisheries Investigations. Scripps Institution of Oceanography & NOAA Southwest Fisheries Science Center. https://calcofi.org

---

## 2. UNESCO / IOC Ocean Biodiversity Information System (OBIS)

- **Dataset Name**: Ocean Biodiversity Information System (OBIS) Marine Occurrences
- **Official Source URL**: [https://portal.obis.org/](https://portal.obis.org/) / API: [https://api.obis.org/v3/](https://api.obis.org/v3/)
- **Coordinating Organization**: Intergovernmental Oceanographic Commission (IOC) of UNESCO / International Oceanographic Data and Information Exchange (IODE)
- **Geographic Coverage**: California Current Marine Ecosystem (Lat: 31.0°N to 38.5°N, Lon: -125.0°W to -117.0°W)
- **Temporal Coverage**: 1990–2024
- **Key Variables Used**:
  - `scientificName`: Valid scientific taxonomic name (WoRMS standard)
  - `vernacularName`: Common name
  - `phylum`, `class`, `order`, `family`, `genus`, `species`: Complete taxonomic lineage
  - `decimalLatitude`, `decimalLongitude`: Geographic coordinates
  - `eventDate`, `date_year`: Observation timestamp
  - `depth`: Depth of observation (meters)
  - `individualCount`: Count of organisms observed
  - `basisOfRecord`: Observation method (Human observation, preserved specimen, machine observation)
- **Scientific License**: Creative Commons Attribution (CC-BY 4.0) / Public Domain (CC0).
- **Citation**: 
  > OBIS (2024). Ocean Biodiversity Information System. Intergovernmental Oceanographic Commission of UNESCO. https://obis.org

---

## 3. Data Integration Methodology

The environmental (CalCOFI) and biological (OBIS) datasets are integrated using a scientifically sound **Mesoscale Spatial-Temporal Gridding** framework:
1. **Spatial Grid**: Both datasets are mapped to 0.25° × 0.25° latitude/longitude bins (~25 km oceanographic mesoscale grid cells).
2. **Temporal Alignment**: Grouped across seasonal oceanographic cycles (Q1: Winter upwelling precursor, Q2: Spring peak upwelling, Q3: Summer thermal stratification, Q4: Autumn relaxation).
3. **Biological Metric Aggregation**: For each spatial grid cell, species richness ($S$), Shannon-Wiener diversity index ($H' = -\sum p_i \ln p_i$), and taxonomic representations are computed.
4. **Hydrographic Fusion**: Environmental hydrographic records are enriched with their corresponding spatial cell's biodiversity profile.
5. **Agent Independence**: Raw occurrences remain available for specialized deep queries by the Marine Life Agent, preventing loss of granular taxonomic signals.
