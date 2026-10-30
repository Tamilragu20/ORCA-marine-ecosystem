import urllib.request
import json
import re
import os

print("--- Testing OBIS API ---")
obis_url = "https://api.obis.org/v3/occurrence?geometry=POLYGON((-124%2032,-117%2032,-117%2038,-124%2038,-124%2032))&size=10"
try:
    req = urllib.request.Request(obis_url, headers={"User-Agent": "ORCA-Marine-Research/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        obis_data = json.loads(resp.read().decode("utf-8"))
        print(f"OBIS total records in California Current study region: {obis_data.get('total')}")
        results = obis_data.get("results", [])
        if results:
            print("Sample record:", results[0].get("scientificName"), "| Taxa:", results[0].get("phylum"), "| Lat/Lon:", results[0].get("decimalLatitude"), results[0].get("decimalLongitude"))
except Exception as e:
    print(f"OBIS probe error: {e}")

print("\n--- Testing NOAA / Oceanographic Data Sources ---")
test_sources = [
    ("CalCOFI Scripps Oceanographic", "https://calcofi.org/downloads/database/"),
    ("CalCOFI Bottle Database Web", "https://calcofi.org/data/oceanographic-data/bottle-database/"),
    ("NOAA NCEI Ocean Data", "https://www.ncei.noaa.gov/data/oceans/wod/"),
    ("NOAA NDBC Monterey Buoy 46042", "https://www.ndbc.noaa.gov/data/realtime2/46042.txt"),
    ("Scripps Data Portal", "https://oceaninformatics.ucsd.edu/calcofi/"),
]

for name, url in test_sources:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urllib.request.urlopen(req, timeout=12) as resp:
            print(f"{name}: HTTP {resp.status}, Content-Type: {resp.headers.get('Content-Type')}")
            content = resp.read()[:5000].decode("utf-8", errors="ignore")
            hrefs = re.findall(r'href=[\'"]([^\'"]+\.(?:csv|zip|gz|txt))[\'"]', content, re.I)
            if hrefs:
                print(f"  Found data file links: {hrefs[:5]}")
    except Exception as e:
        print(f"{name}: Error ({e})")
