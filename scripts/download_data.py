"""
Automated Data Download Script for Colombo District

Downloads all required datasets for flood vulnerability analysis:
- CHIRPS rainfall (daily, 0.05° resolution)
- SRTM DEM from AWS Open Data (30m resolution)
- Administrative boundaries from OSM (Overpass API)
- Google Open Buildings (direct download)

No API keys required!
"""

import os
import requests
import zipfile
import io
from pathlib import Path
from typing import Tuple, List, Optional
import geopandas as gpd
from shapely.geometry import box
import json

# Colombo District bounding box
COLOMBO_BBOX = {
    'west': 79.82,
    'east': 80.22,
    'south': 6.75,
    'north': 7.05
}


def create_data_directories(base_path: Path = Path('data')) -> dict:
    """Create directory structure for data storage."""
    dirs = {
        'raw': base_path / 'raw',
        'processed': base_path / 'processed',
        'chirps': base_path / 'raw' / 'chirps',
        'dem': base_path / 'raw' / 'dem',
        'buildings': base_path / 'raw' / 'buildings',
        'admin': base_path / 'raw' / 'admin'
    }
    
    for name, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        print(f"Created: {path}")
    
    return dirs


def download_file(url: str, output_path: Path, chunk_size: int = 8192) -> bool:
    """Download a file with progress indication."""
    try:
        print(f"Downloading: {url}")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\r  Progress: {percent:.1f}%", end="", flush=True)
        
        print(f"\n  Saved: {output_path}")
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False


# ============================================================
# CHIRPS RAINFALL DATA
# ============================================================

def download_chirps(
    years: List[int] = [2022, 2023],
    output_dir: Path = Path('data/raw/chirps')
) -> List[Path]:
    """
    Download CHIRPS daily rainfall data.
    
    Parameters
    ----------
    years : list
        Years to download
    output_dir : Path
        Output directory
    
    Returns
    -------
    list
        Paths to downloaded files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    
    base_url = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p05"
    
    for year in years:
        filename = f"chirps-v2.0.{year}.days_p05.nc"
        url = f"{base_url}/{filename}"
        output_path = output_dir / filename
        
        if output_path.exists():
            print(f"Already exists: {output_path}")
            downloaded.append(output_path)
            continue
        
        if download_file(url, output_path):
            downloaded.append(output_path)
    
    return downloaded


# ============================================================
# SRTM DEM FROM AWS
# ============================================================

def get_srtm_tiles(bbox: dict) -> List[str]:
    """Get list of SRTM tile names needed for bounding box."""
    tiles = []
    
    # SRTM tiles are 1x1 degree, named by SW corner
    for lat in range(int(bbox['south']), int(bbox['north']) + 1):
        for lon in range(int(bbox['west']), int(bbox['east']) + 1):
            # Format: N06E079 (latitude, longitude)
            lat_str = f"N{abs(lat):02d}" if lat >= 0 else f"S{abs(lat):02d}"
            lon_str = f"E{abs(lon):03d}" if lon >= 0 else f"W{abs(lon):03d}"
            tiles.append(f"{lat_str}{lon_str}")
    
    return tiles


def download_srtm_aws(
    bbox: dict = COLOMBO_BBOX,
    output_dir: Path = Path('data/raw/dem')
) -> List[Path]:
    """
    Download SRTM tiles from AWS Open Data.
    
    AWS SRTM URL pattern:
    https://elevation-tiles-prod.s3.amazonaws.com/skadi/{N|S}{lat}/{N|S}{lat}{E|W}{lon}.hgt.gz
    
    Parameters
    ----------
    bbox : dict
        Bounding box with west, east, south, north keys
    output_dir : Path
        Output directory
    
    Returns
    -------
    list
        Paths to downloaded files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    
    tiles = get_srtm_tiles(bbox)
    print(f"SRTM tiles needed: {tiles}")
    
    base_url = "https://elevation-tiles-prod.s3.amazonaws.com/skadi"
    
    for tile in tiles:
        # Extract lat/lon from tile name
        lat_dir = tile[:3]  # e.g., N06
        filename = f"{tile}.hgt.gz"
        url = f"{base_url}/{lat_dir}/{filename}"
        output_path = output_dir / filename
        
        if output_path.exists():
            print(f"Already exists: {output_path}")
            downloaded.append(output_path)
            continue
        
        if download_file(url, output_path):
            downloaded.append(output_path)
    
    return downloaded


# ============================================================
# OSM ADMINISTRATIVE BOUNDARIES
# ============================================================

def download_osm_admin(
    district_name: str = "Colombo",
    output_dir: Path = Path('data/raw/admin')
) -> Optional[Path]:
    """
    Download administrative boundary from OpenStreetMap using Overpass API.
    
    Parameters
    ----------
    district_name : str
        Name of the district
    output_dir : Path
        Output directory
    
    Returns
    -------
    Path or None
        Path to downloaded GeoJSON file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{district_name.lower()}_boundary.geojson"
    
    if output_path.exists():
        print(f"Already exists: {output_path}")
        return output_path
    
    # Overpass API query for district boundary
    overpass_url = "https://overpass-api.de/api/interpreter"
    
    query = f"""
    [out:json][timeout:60];
    area["name"="Sri Lanka"]->.country;
    (
      relation["name"="{district_name}"]["admin_level"="5"](area.country);
      relation["name"="{district_name} District"]["admin_level"="5"](area.country);
    );
    out geom;
    """
    
    try:
        print(f"Querying OSM for {district_name} boundary...")
        response = requests.post(overpass_url, data={'data': query}, timeout=120)
        response.raise_for_status()
        data = response.json()
        
        if not data.get('elements'):
            print(f"  No boundary found for {district_name}")
            return None
        
        # Convert to GeoJSON
        features = []
        for element in data['elements']:
            if element['type'] == 'relation' and 'bounds' in element:
                # Get geometry from relation members
                geometry = element.get('geometry', [])
                if geometry:
                    coords = [[p['lon'], p['lat']] for p in geometry]
                    feature = {
                        'type': 'Feature',
                        'properties': element.get('tags', {}),
                        'geometry': {
                            'type': 'Polygon',
                            'coordinates': [coords]
                        }
                    }
                    features.append(feature)
        
        if features:
            geojson = {'type': 'FeatureCollection', 'features': features}
            with open(output_path, 'w') as f:
                json.dump(geojson, f)
            print(f"  Saved: {output_path}")
            return output_path
        else:
            print(f"  Could not extract geometry for {district_name}")
            return None
            
    except Exception as e:
        print(f"  Error: {e}")
        return None


def download_osm_admin_simple(
    bbox: dict = COLOMBO_BBOX,
    output_dir: Path = Path('data/raw/admin')
) -> Optional[Path]:
    """
    Download all admin boundaries within bounding box using Nominatim/Overpass.
    
    This is a simpler alternative that gets all admin boundaries in the area.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "colombo_admin.geojson"
    
    if output_path.exists():
        print(f"Already exists: {output_path}")
        return output_path
    
    overpass_url = "https://overpass-api.de/api/interpreter"
    
    # Query for admin boundaries in bbox
    query = f"""
    [out:json][timeout:120];
    (
      relation["admin_level"~"5|6|7"]
        ({bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']});
    );
    out geom;
    """
    
    try:
        print("Querying OSM for admin boundaries in Colombo area...")
        response = requests.post(overpass_url, data={'data': query}, timeout=180)
        response.raise_for_status()
        data = response.json()
        
        print(f"  Found {len(data.get('elements', []))} boundary elements")
        
        # Save raw response
        with open(output_path.with_suffix('.json'), 'w') as f:
            json.dump(data, f)
        
        print(f"  Saved raw data: {output_path.with_suffix('.json')}")
        return output_path.with_suffix('.json')
        
    except Exception as e:
        print(f"  Error: {e}")
        return None


# ============================================================
# GOOGLE OPEN BUILDINGS
# ============================================================

def download_google_buildings_info():
    """
    Print instructions for Google Open Buildings download.
    
    Note: Google Buildings requires manual selection from their website
    as it's organized by S2 cells, not simple bounding boxes.
    """
    print("""
    ============================================================
    GOOGLE OPEN BUILDINGS - Manual Download Required
    ============================================================
    
    1. Go to: https://sites.research.google/open-buildings/#download
    2. Select region: South Asia → Sri Lanka
    3. Download the CSV for your area of interest
    4. Place the file in: data/raw/buildings/
    
    Alternatively, use OSM buildings:
    - Run: download_osm_buildings()
    ============================================================
    """)


def download_osm_buildings(
    bbox: dict = COLOMBO_BBOX,
    output_dir: Path = Path('data/raw/buildings')
) -> Optional[Path]:
    """
    Download building footprints from OpenStreetMap.
    
    Note: This may take a while for urban areas with many buildings.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "osm_buildings_colombo.geojson"
    
    if output_path.exists():
        print(f"Already exists: {output_path}")
        return output_path
    
    overpass_url = "https://overpass-api.de/api/interpreter"
    
    # Query for buildings - limiting to reduce load
    query = f"""
    [out:json][timeout:300];
    (
      way["building"]
        ({bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']});
    );
    out geom;
    """
    
    try:
        print("Querying OSM for buildings in Colombo (this may take several minutes)...")
        response = requests.post(overpass_url, data={'data': query}, timeout=600)
        response.raise_for_status()
        data = response.json()
        
        print(f"  Found {len(data.get('elements', []))} building elements")
        
        # Save raw response
        raw_path = output_dir / "osm_buildings_raw.json"
        with open(raw_path, 'w') as f:
            json.dump(data, f)
        
        print(f"  Saved: {raw_path}")
        return raw_path
        
    except Exception as e:
        print(f"  Error: {e}")
        return None


# ============================================================
# MAIN DOWNLOAD FUNCTION
# ============================================================

def download_all(
    years: List[int] = [2022, 2023],
    skip_buildings: bool = True
):
    """
    Download all required datasets for Colombo District.
    
    Parameters
    ----------
    years : list
        Years of CHIRPS data to download
    skip_buildings : bool
        Skip building download (can be slow)
    """
    print("=" * 60)
    print("AUTOMATED DATA DOWNLOAD - Colombo District")
    print("=" * 60)
    print(f"Bounding Box: {COLOMBO_BBOX}")
    print()
    
    # Create directories
    print("\n1. Creating directories...")
    dirs = create_data_directories()
    
    # Download CHIRPS
    print("\n2. Downloading CHIRPS rainfall data...")
    chirps_files = download_chirps(years=years)
    print(f"   Downloaded {len(chirps_files)} CHIRPS files")
    
    # Download SRTM
    print("\n3. Downloading SRTM DEM from AWS...")
    dem_files = download_srtm_aws()
    print(f"   Downloaded {len(dem_files)} DEM tiles")
    
    # Download admin boundaries
    print("\n4. Downloading administrative boundaries from OSM...")
    admin_file = download_osm_admin_simple()
    
    # Buildings
    if not skip_buildings:
        print("\n5. Downloading building footprints from OSM...")
        buildings_file = download_osm_buildings()
    else:
        print("\n5. Skipping buildings (set skip_buildings=False to download)")
        download_google_buildings_info()
    
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run preprocessing to clip data to Colombo boundary")
    print("2. Open notebooks/analysis.ipynb to start analysis")


if __name__ == "__main__":
    # Default: download 2022-2023 CHIRPS, DEM, and admin boundaries
    # Skip buildings by default (can be slow)
    download_all(years=[2022, 2023], skip_buildings=True)
