"""
ORCA Marine Data Acquisition Script
Downloads official CalCOFI (NOAA/Scripps) oceanographic bottle data
and OBIS (UNESCO/IOC) marine biodiversity occurrence records.
"""

import os
import sys
import zipfile
import urllib.request
import json
import pandas as pd
import time

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

CALCOFI_URL = "https://calcofi.org/downloads/database/CalCOFI_Database_194903-202105_csv_16October2023.zip"
CALCOFI_ZIP = os.path.join(RAW_DIR, "calcofi_database.zip")

def download_calcofi():
    print("=" * 60)
    print("STEP 1: Downloading NOAA / Scripps CalCOFI Oceanographic Database")
    print(f"Source URL: {CALCOFI_URL}")
    print("=" * 60)

    if not os.path.exists(CALCOFI_ZIP):
        def reporthook(count, block_size, total_size):
            if total_size > 0:
                percent = int(count * block_size * 100 / total_size)
                mb = count * block_size / (1024 * 1024)
                sys.stdout.write(f"\rDownloading CalCOFI archive: {mb:.1f} MB / {total_size/(1024*1024):.1f} MB ({percent}%)")
                sys.stdout.flush()

        req = urllib.request.Request(CALCOFI_URL, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(CALCOFI_ZIP, "wb") as out_file:
            total_size = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            block_size = 1024 * 64
            while True:
                chunk = resp.read(block_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = int(downloaded * 100 / total_size)
                    sys.stdout.write(f"\rDownloading: {downloaded/(1024*1024):.1f}MB / {total_size/(1024*1024):.1f}MB ({pct}%)")
                    sys.stdout.flush()
        print("\nDownload complete!")
    else:
        print(f"Archive already exists at {CALCOFI_ZIP}")

    print("Extracting CalCOFI archive...")
    with zipfile.ZipFile(CALCOFI_ZIP, "r") as z:
        namelist = z.namelist()
        print(f"Archive contents: {namelist}")
        z.extractall(RAW_DIR)
    print(f"Extracted to {RAW_DIR}")


def download_obis_biodiversity(sample_size=6000):
    print("\n" + "=" * 60)
    print("STEP 2: Fetching UNESCO/IOC OBIS Marine Biodiversity Records")
    print("Study Region: California Current Marine Ecosystem (Lat: 31.0 - 38.5 N, Lon: -125.0 - -117.0 W)")
    print("=" * 60)
    
    csv_out = os.path.join(RAW_DIR, "obis_marine_biodiversity.csv")
    json_out = os.path.join(RAW_DIR, "obis_marine_biodiversity.json")
    
    if os.path.exists(csv_out) and os.path.getsize(csv_out) > 50000:
        print(f"OBIS biodiversity records already present at {csv_out}")
        return

    # Polygon for California Current / Monterey Bay / SoCal Bight
    polygon = "POLYGON((-125 31,-117 31,-117 38.5,-125 38.5,-125 31))"
    base_url = "https://api.obis.org/v3/occurrence"
    
    all_records = []
    # Fetch in batches of 1000
    batch_size = 1000
    max_batches = sample_size // batch_size
    
    for page in range(max_batches):
        offset = page * batch_size
        # Filter for marine species observations with valid coordinates and depths
        url = f"{base_url}?geometry={urllib.parse.quote(polygon)}&size={batch_size}&offset={offset}&has_depth=true"
        print(f"Querying OBIS API batch {page+1}/{max_batches} (offset={offset})...")
        req = urllib.request.Request(url, headers={"User-Agent": "ORCA-Ecosystem-Platform/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                results = data.get("results", [])
                if not results:
                    break
                for r in results:
                    all_records.append({
                        "id": r.get("id"),
                        "scientificName": r.get("scientificName"),
                        "vernacularName": r.get("vernacularName"),
                        "phylum": r.get("phylum"),
                        "class": r.get("class"),
                        "order": r.get("order"),
                        "family": r.get("family"),
                        "genus": r.get("genus"),
                        "species": r.get("species"),
                        "taxonRank": r.get("taxonRank"),
                        "decimalLatitude": r.get("decimalLatitude"),
                        "decimalLongitude": r.get("decimalLongitude"),
                        "eventDate": r.get("eventDate"),
                        "date_year": r.get("date_year"),
                        "depth": r.get("depth"),
                        "minimumDepthInMeters": r.get("minimumDepthInMeters"),
                        "maximumDepthInMeters": r.get("maximumDepthInMeters"),
                        "individualCount": r.get("individualCount"),
                        "basisOfRecord": r.get("basisOfRecord"),
                        "datasetName": r.get("datasetName"),
                    })
                print(f"  Retrieved {len(results)} records (Cumulative: {len(all_records)})")
            time.sleep(0.5)
        except Exception as e:
            print(f"Error fetching batch {page+1}: {e}")
            break

    df = pd.DataFrame(all_records)
    df.to_csv(csv_out, index=False)
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(all_records[:500], f, indent=2)
    print(f"Successfully saved {len(df)} OBIS occurrence records to {csv_out}")

if __name__ == "__main__":
    download_calcofi()
    download_obis_biodiversity()
    print("\nData download finished successfully!")
