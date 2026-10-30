import pandas as pd
import glob

bottle_path = glob.glob('data/raw/**/194903-202105_Bottle.csv', recursive=True)[0]
cast_path = glob.glob('data/raw/**/194903-202105_Cast.csv', recursive=True)[0]

print('Reading Cast index...')
casts = pd.read_csv(cast_path, usecols=['Cst_Cnt', 'Date', 'Year', 'Month', 'Lat_Dec', 'Lon_Dec'])
min_yr, max_yr = casts['Year'].min(), casts['Year'].max()
print(f'Total casts: {len(casts)}, Years: {min_yr} - {max_yr}')

# Filter casts to modern era (1984 - 2021) where chlorophyll and nutrients were systematically measured
modern_casts = casts[casts['Year'] >= 1984]
print(f'Modern casts (>= 1984): {len(modern_casts)}')

# Inspect bottle data sample in chunks
print('Inspecting modern bottle measurements...')
cols = ['Cst_Cnt', 'Depthm', 'T_degC', 'Salnty', 'O2ml_L', 'O2Sat', 'ChlorA', 'PO4uM', 'SiO3uM', 'NO3uM']
chunks = pd.read_csv(bottle_path, usecols=cols, chunksize=100000, encoding='latin-1', low_memory=False)

sample_counts = {'total': 0, 'with_temp_sal_o2': 0, 'with_chla': 0, 'complete_all': 0}
for i, chunk in enumerate(chunks):
    sample_counts['total'] += len(chunk)
    c1 = chunk.dropna(subset=['T_degC', 'Salnty', 'O2ml_L'])
    sample_counts['with_temp_sal_o2'] += len(c1)
    c2 = chunk.dropna(subset=['ChlorA'])
    sample_counts['with_chla'] += len(c2)
    c3 = chunk.dropna(subset=['T_degC', 'Salnty', 'O2ml_L', 'ChlorA'])
    sample_counts['complete_all'] += len(c3)
    if i >= 4: # sample first 500k rows
        break

print('Sample profile counts:', sample_counts)
