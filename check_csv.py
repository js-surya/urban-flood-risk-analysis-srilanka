
import pandas as pd
import numpy as np

filepath = 'data/raw/buildings/google_buildings.csv.gz'
print(f"Checking {filepath}...")

try:
    # Read first 1000 rows to check structure and sample coordinates
    df = pd.read_csv(filepath, nrows=1000, compression='gzip')
    print("Columns:", df.columns.tolist())
    
    print("\nSample Data (first 5 rows):")
    print(df[['latitude', 'longitude']].head())
    
    print("\nReading full coordinate range (this might take a moment)...")
    # Use chunks to avoid memory issues if file is huge
    min_lat, max_lat = 90, -90
    min_lon, max_lon = 180, -180
    count = 0
    
    # Just read lat/lon columns
    for chunk in pd.read_csv(filepath, usecols=['latitude', 'longitude'], chunksize=100000, compression='gzip'):
        min_lat = min(min_lat, chunk['latitude'].min())
        max_lat = max(max_lat, chunk['latitude'].max())
        min_lon = min(min_lon, chunk['longitude'].min())
        max_lon = max(max_lon, chunk['longitude'].max())
        count += len(chunk)
        if count > 1000000: # check first 1 million rows only for speed
            print("Checked 1M rows...")
            break
            
    print(f"\nApproximate Bounds (first {count} rows):")
    print(f"Lat: {min_lat:.4f} to {max_lat:.4f}")
    print(f"Lon: {min_lon:.4f} to {max_lon:.4f}")
    
    # Colombo BBOX
    colombo = {'w': 79.82, 'e': 80.22, 's': 6.75, 'n': 7.05}
    print(f"\nTarget BBOX: Lat {colombo['s']}-{colombo['n']}, Lon {colombo['w']}-{colombo['e']}")
    
    # Check overlap
    if (max_lat < colombo['s'] or min_lat > colombo['n'] or 
        max_lon < colombo['w'] or min_lon > colombo['e']):
        print("❌ NO OVERLAP with Colombo!")
    else:
        print("✅ Overlap detected!")

except Exception as e:
    print(f"Error: {e}")
