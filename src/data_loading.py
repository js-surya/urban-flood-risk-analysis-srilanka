"""
Data Loading Module

Functions for loading various geospatial datasets:
- CHIRPS rainfall data (NetCDF)
- SRTM DEM (GeoTIFF)
- OpenStreetMap features (Shapefile/GeoPackage)
"""



import xarray as xr
import rioxarray  # noqa: F401
from rioxarray.merge import merge_arrays
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import box, shape
from pathlib import Path
from typing import Union, Optional, List
import gzip
import struct
import requests
import json
import shutil


def load_srtm_tiles(
    tile_dir: Union[str, Path],
    bbox: Optional[tuple] = None
) -> xr.DataArray:
    """
    Load SRTM DEM from multiple .hgt.gz tiles.
    
    Parameters
    ----------
    tile_dir : str or Path
        Directory containing SRTM .hgt.gz files
    bbox : tuple, optional
        Bounding box to clip (west, south, east, north)
    
    Returns
    -------
    xr.DataArray
        Merged elevation data
    """
    tile_dir = Path(tile_dir)
    hgt_files = list(tile_dir.glob('*.hgt.gz')) + list(tile_dir.glob('*.hgt'))
    
    if not hgt_files:
        raise FileNotFoundError(f"No SRTM tiles found in {tile_dir}")
    
    tiles = []
    for hgt_file in sorted(hgt_files):
        tile = _read_single_hgt(hgt_file)
        if tile is not None:
            tiles.append(tile)
    
    if not tiles:
        raise ValueError("Could not read any SRTM tiles")
    
    # Merge tiles using rioxarray (handles 2D grid correctly)
    try:
        merged = merge_arrays(tiles)
    except Exception as e:
        print(f"Warning: merge_arrays failed ({e}), falling back to concat")
        merged = xr.concat(tiles, dim='y')

    # Assign CRS (SRTM is always WGS84)
    merged.rio.write_crs("EPSG:4326", inplace=True)
    
    # Clip to bbox if provided
    if bbox is not None:
        west, south, east, north = bbox
        merged = merged.rio.clip_box(minx=west, miny=south, maxx=east, maxy=north)
    
    merged.name = 'elevation'
    return merged


def _read_single_hgt(filepath: Path) -> xr.DataArray:
    """Read a single SRTM HGT file (compressed or uncompressed)."""
    # Parse coordinates from filename (e.g., N06E079.hgt.gz)
    name = filepath.stem.replace('.hgt', '')
    lat_char = name[0]  # N or S
    lat = int(name[1:3])
    lon_char = name[3]  # E or W
    lon = int(name[4:7])
    
    if lat_char == 'S':
        lat = -lat
    if lon_char == 'W':
        lon = -lon
    
    # Read file
    try:
        if filepath.suffix == '.gz':
            with gzip.open(filepath, 'rb') as f:
                data = f.read()
        else:
            with open(filepath, 'rb') as f:
                data = f.read()
        
        # SRTM1 = 3601x3601, SRTM3 = 1201x1201
        size = int(np.sqrt(len(data) / 2))
        elevation = np.frombuffer(data, dtype='>i2').reshape((size, size))
        
        # Create coordinate arrays
        lats = np.linspace(lat + 1, lat, size)
        lons = np.linspace(lon, lon + 1, size)
        
        da = xr.DataArray(
            data=elevation.astype(np.float32),
            dims=['y', 'x'],
            coords={'y': lats, 'x': lons}
        )
        
        # Replace void values
        da = da.where(da != -32768)
        
        # Assign CRS explicitly to enable safe merging
        da.rio.write_crs("EPSG:4326", inplace=True)
        
        return da
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None


def load_chirps_data(
    filepath: Union[str, Path],
    variable: str = "precip",
    time_slice: Optional[tuple] = None
) -> xr.DataArray:
    """
    Load CHIRPS rainfall data from NetCDF file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the CHIRPS NetCDF file
    variable : str
        Variable name in the NetCDF (default: 'precip')
    time_slice : tuple, optional
        Start and end dates for temporal subset, e.g., ('2020-01-01', '2020-12-31')
    
    Returns
    -------
    xr.DataArray
        Rainfall data as xarray DataArray with dimensions (time, lat, lon)
    
    Example
    -------
    >>> rainfall = load_chirps_data('data/chirps_2020.nc')
    >>> print(rainfall.dims)
    ('time', 'latitude', 'longitude')
    """
    # open the netcdf file
    ds = xr.open_dataset(filepath)
    data = ds[variable]
    
    # apply time slice if provided
    if time_slice is not None:
        start_date, end_date = time_slice
        data = data.sel(time=slice(start_date, end_date))
    
    # data = data.sel(time=slice(start_date, end_date))
    
    # Assign CRS (CHIRPS is EPSG:4326)
    if data.rio.crs is None:
        data.rio.write_crs("EPSG:4326", inplace=True)
    
    return data


def load_dem(
    filepath: Union[str, Path],
    clip_bounds: Optional[tuple] = None
) -> xr.DataArray:
    """
    Load Digital Elevation Model from GeoTIFF.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the DEM GeoTIFF file
    clip_bounds : tuple, optional
        Bounding box to clip (minx, miny, maxx, maxy)
    
    Returns
    -------
    xr.DataArray
        Elevation data with CRS information
    
    Example
    -------
    >>> dem = load_dem('data/srtm_sri_lanka.tif')
    >>> print(dem.rio.crs)
    EPSG:4326
    """
    # load raster using rioxarray
    dem = xr.open_dataarray(filepath, engine='rasterio')
    
    # clip to bounds if provided
    if clip_bounds is not None:
        minx, miny, maxx, maxy = clip_bounds
        dem = dem.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
    
    return dem



def load_osm_buildings(
    filepath: Union[str, Path],
    bbox: Optional[tuple] = None
) -> gpd.GeoDataFrame:
    """
    Load buildings from OpenStreetMap/Overpass GeoJSON.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the OSM buildings GeoJSON file
    bbox : tuple, optional
        Bounding box filter (minx, miny, maxx, maxy)
    
    Returns
    -------
    gpd.GeoDataFrame
        Building footprints as polygons
    """
    try:
        if bbox is not None:
            return gpd.read_file(filepath, bbox=bbox)
        else:
            return gpd.read_file(filepath)
    except Exception:
        # Fallback for raw Overpass JSON
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            elements = data.get('elements', [])
            if not elements:
                return gpd.GeoDataFrame(columns=['geometry'], crs="EPSG:4326")
                
            # Convert Overpass elements to GeoDataFrame
            # Overpass 'way' with 'geometry' (list of lat/lon) -> Polygon
            from shapely.geometry import Polygon, LineString
            
            geoms = []
            properties = []
            
            for el in elements:
                if 'geometry' in el:
                    coords = [(pt['lon'], pt['lat']) for pt in el['geometry']]
                    if len(coords) < 3:
                        geom = LineString(coords) # Fallback if not closed
                    else:
                        geom = Polygon(coords)
                    
                    geoms.append(geom)
                    tags = el.get('tags', {})
                    tags['building_id'] = el.get('id', 0)  # Inject ID
                    properties.append(tags)
            
            if not geoms:
                 return gpd.GeoDataFrame(columns=['geometry'], crs="EPSG:4326")
                 
            gdf = gpd.GeoDataFrame(properties, geometry=geoms, crs="EPSG:4326")
            
            if bbox:
                 minx, miny, maxx, maxy = bbox
                 gdf = gdf.cx[minx:maxx, miny:maxy]
                 
            return gdf
            
        except Exception as e:
            print(f"Error parsing OSM JSON: {e}")
            return gpd.GeoDataFrame(columns=['geometry'], crs="EPSG:4326")



def load_osm_roads(
    filepath: Union[str, Path],
    road_types: Optional[list] = None
) -> gpd.GeoDataFrame:
    """
    Load road network from OpenStreetMap data.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the OSM roads shapefile or geopackage
    road_types : list, optional
        Filter by highway types, e.g., ['primary', 'secondary', 'trunk']
    
    Returns
    -------
    gpd.GeoDataFrame
        Road network as LineStrings
    
    Example
    -------
    >>> roads = load_osm_roads('data/osm_roads.shp', road_types=['primary', 'secondary'])
    >>> print(f"Loaded {len(roads)} road segments")
    """
    filepath = Path(filepath)
    try:
        roads = gpd.read_file(filepath)
    except Exception:
        # Fallback for raw Overpass JSON (like buildings loader)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            elements = data.get('elements', [])
            if not elements:
                return gpd.GeoDataFrame(columns=['geometry'], crs="EPSG:4326")

            from shapely.geometry import LineString

            geoms = []
            properties = []
            for el in elements:
                geom_coords = el.get('geometry')
                if not geom_coords:
                    continue
                coords = [(pt['lon'], pt['lat']) for pt in geom_coords]
                if len(coords) < 2:
                    continue  # need at least 2 points for a LineString
                geoms.append(LineString(coords))
                tags = el.get('tags', {})
                tags['road_id'] = el.get('id', 0)
                properties.append(tags)

            if not geoms:
                return gpd.GeoDataFrame(columns=['geometry'], crs="EPSG:4326")

            roads = gpd.GeoDataFrame(properties, geometry=geoms, crs="EPSG:4326")
        except Exception as e:
            print(f"Error parsing OSM roads JSON: {e}")
            return gpd.GeoDataFrame(columns=['geometry'], crs="EPSG:4326")
    
    # filter by road type if specified
    if road_types is not None and 'highway' in roads.columns:
        roads = roads[roads['highway'].isin(road_types)]
    
    return roads


def load_osm_water(
    filepath: Union[str, Path],
    bbox: Optional[tuple] = None
) -> gpd.GeoDataFrame:
    """
    Load water bodies from OpenStreetMap/Overpass GeoJSON.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the OSM water bodies GeoJSON file
    bbox : tuple, optional
        Bounding box filter (minx, miny, maxx, maxy)
    
    Returns
    -------
    gpd.GeoDataFrame
        Water body features (rivers, canals, lakes) as polygons/linestrings
    """
    try:
        if bbox is not None:
            return gpd.read_file(filepath, bbox=bbox)
        else:
            return gpd.read_file(filepath)
    except Exception:
        # Fallback for raw Overpass JSON
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            elements = data.get('elements', [])
            if not elements:
                return gpd.GeoDataFrame(columns=['geometry', 'waterway', 'natural'], crs="EPSG:4326")
                
            # Convert Overpass elements to GeoDataFrame
            from shapely.geometry import Polygon, LineString
            
            geoms = []
            properties = []
            
            for el in elements:
                if 'geometry' in el:
                    coords = [(pt['lon'], pt['lat']) for pt in el['geometry']]
                    if len(coords) < 2:
                        continue
                    
                    # Waterways are typically LineStrings, water bodies can be Polygons
                    tags = el.get('tags', {})
                    if 'waterway' in tags and len(coords) >= 2:
                        # River, canal, stream - LineString
                        geom = LineString(coords)
                    elif len(coords) >= 3 and coords[0] == coords[-1]:
                        # Closed polygon - lake, reservoir
                        geom = Polygon(coords)
                    elif len(coords) >= 3:
                        # Try to close it for water bodies
                        geom = Polygon(coords)
                    else:
                        geom = LineString(coords)
                    
                    geoms.append(geom)
                    tags['water_id'] = el.get('id', 0)
                    properties.append(tags)
            
            if not geoms:
                 return gpd.GeoDataFrame(columns=['geometry', 'waterway', 'natural'], crs="EPSG:4326")
                 
            gdf = gpd.GeoDataFrame(properties, geometry=geoms, crs="EPSG:4326")
            
            if bbox:
                 minx, miny, maxx, maxy = bbox
                 gdf = gdf.cx[minx:maxx, miny:maxy]
                 
            return gdf
            
        except Exception as e:
            print(f"Error parsing OSM water JSON: {e}")
            return gpd.GeoDataFrame(columns=['geometry', 'waterway', 'natural'], crs="EPSG:4326")


def load_admin_boundaries(
    filepath: Union[str, Path],
    level: Optional[str] = None
) -> gpd.GeoDataFrame:
    """
    Load administrative boundaries.
    
    Parameters
    ----------
    filepath : str or Path
        Path to boundaries shapefile or geopackage
    level : str, optional
        Admin level to filter (e.g., 'district', 'division')
    
    Returns
    -------
    gpd.GeoDataFrame
        Administrative boundary polygons
    """
    boundaries = gpd.read_file(filepath)
    
    # filter by admin level if column exists
    if level is not None and 'admin_level' in boundaries.columns:
        boundaries = boundaries[boundaries['admin_level'] == level]
    
    return boundaries


def validate_crs_match(
    *datasets: Union[xr.DataArray, gpd.GeoDataFrame]
) -> bool:
    """
    Check if all datasets have matching CRS.
    
    Parameters
    ----------
    *datasets : xr.DataArray or gpd.GeoDataFrame
        Variable number of datasets to compare
    
    Returns
    -------
    bool
        True if all CRS match
    
    Example
    -------
    >>> if not validate_crs_match(buildings, admin_boundaries):
    ...     admin_boundaries = admin_boundaries.to_crs(buildings.crs)
    """
    crs_list = []
    
    for ds in datasets:
        if hasattr(ds, 'rio') and hasattr(ds.rio, 'crs'):
            # rioxarray raster
            crs_list.append(str(ds.rio.crs))
        elif hasattr(ds, 'crs'):
            # geopandas geodataframe
            crs_list.append(str(ds.crs))
        else:
            continue
    
    # check if all CRS are the same
    return len(set(crs_list)) <= 1




def download_file(url: str, dest_path: Path):
    """Helper to download a file with progress."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://overpass.openstreetmap.fr/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter"
]

def _run_overpass_query(query: str, initial_timeout: int = 180) -> dict:
    """
    Run an Overpass query with automatic retries and endpoint cycling.
    
    Parameters
    ----------
    query : str
        The Overpass QL query string
    initial_timeout : int
        Initial timeout for the request
        
    Returns
    -------
    dict
        Parsed JSON response or empty elements dict if all fail
    """
    import time
    import random

    # Ensure query has its own [timeout:...] if not present
    # But usually we'll build it in the caller
    
    max_retries = 2
    for endpoint in OVERPASS_ENDPOINTS:
        for attempt in range(max_retries + 1):
            # Calculate actual timeout for this attempt
            current_timeout = initial_timeout + (attempt * 60)
            
            try:
                print(f"  Trying Overpass mirror: {endpoint} (Attempt {attempt+1}/{max_retries+1})...")
                response = requests.post(
                    endpoint, 
                    data={'data': query}, 
                    timeout=current_timeout + 30
                )
                
                if response.status_code == 429: # Too many requests
                    print("  Server busy (429). Waiting...")
                    time.sleep(5 * (attempt + 1))
                    continue
                
                response.raise_for_status()
                data = response.json()
                
                if not data.get('elements'):
                    # Sometimes they return no elements on error without status code
                    if 'remark' in data:
                        print(f"  Server remark: {data['remark']}")
                        continue
                
                return data
                
            except requests.exceptions.RequestException as e:
                print(f"  Attempt {attempt+1} failed at {endpoint}: {e}")
                if attempt < max_retries:
                    wait = (attempt + 1) * 3
                    time.sleep(wait)
                else:
                    print(f"  Moving to next mirror...")
                    break
            except Exception as e:
                print(f"  Unexpected error: {e}")
                break
                
    print("CRITICAL: All Overpass mirrors failed.")
    return {"elements": []}


def download_srtm(
    bbox: dict,
    cache_dir: Union[str, Path]
) -> xr.DataArray:
    """
    Download SRTM Data.
    Attempts to download real SRTM 90m data from CGIAR-CSI (public).
    Tile srtm_44_09 covers Sri Lanka.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # CGIAR Tile for Sri Lanka (Lat 5-10N, Lon 75-80E -> 52_11)
    tile_url = "https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF/srtm_52_11.zip"
    zip_path = cache_dir / "srtm_52_11.zip"
    tif_path = cache_dir / "srtm_52_11.tif"
    
    # 1. Download Zip
    if not zip_path.exists() and not tif_path.exists():
        print(f"Downloading Real SRTM (CGIAR 90m) from {tile_url}...")
        try:
            download_file(tile_url, zip_path)
            print("Download complete.")
        except Exception as e:
            print(f"SRTM download failed: {e}")
            return None

    # 2. Extract
    import zipfile
    if not tif_path.exists():
        if zip_path.exists():
             print("Extracting SRTM...")
             try:
                 with zipfile.ZipFile(zip_path, 'r') as z:
                     z.extractall(cache_dir)
                 print("Extraction complete.")
             except Exception as e:
                 print(f"Extraction failed: {e}")
                 return None
        else:
             print("Zip file missing.")
             return None

    # 3. Load & Clip
    # The extracted file should be srtm_44_09.tif
    if tif_path.exists():
        # Load full tile
        da = load_dem(tif_path)
        
        # Assign CRS if missing (CGIAR usually 4326)
        if da.rio.crs is None:
            da.rio.write_crs("EPSG:4326", inplace=True)
            
        # Clip to Colombo BBox
        # Note: BBox is small compared to tile, so clipping is efficient
        minx, miny, maxx, maxy = bbox['west'], bbox['south'], bbox['east'], bbox['north']
        da_clipped = da.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
        
        # Save clipped version for speed?
        # optional
        
        return da_clipped
    else:
        print("SRTM TIF not found after extraction.")
        return None


def download_osm_buildings(
    bbox: dict,
    output_path: Union[str, Path]
) -> gpd.GeoDataFrame:
    """
    Download buildings from Overpass API and save to disk.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    query = f"""
    [out:json][timeout:180];
    (
      way["building"]({bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']});
      relation["building"]({bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']});
    );
    out geom;
    """
    print(f"Requesting Buildings from Overpass API...")
    data = _run_overpass_query(query)
    
    if data.get('elements'):
        with open(output_path, 'w') as f:
            json.dump(data, f)
        print(f"Downloaded raw buildings to {output_path}")
    else:
        print("Warning: No building data retrieved. Using cache if exists.")

    # Load it (will fallback to cache if data empty but file exists)
    return load_osm_buildings(output_path)


def download_osm_roads(
    bbox: dict,
    output_path: Union[str, Path],
    highway_types: Optional[list] = None
) -> gpd.GeoDataFrame:
    """
    Download OSM roads via Overpass and save to disk.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # build type filter
    type_filter = ""
    if highway_types:
        filters = "|".join(highway_types)
        type_filter = f'["highway"~"{filters}"]'
    else:
        type_filter = '["highway"]'

    query = f"""
    [out:json][timeout:300];
    (
      way{type_filter}({bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']});
      relation{type_filter}({bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']});
    );
    out geom;
    """
    print("Requesting roads from Overpass API...")
    data = _run_overpass_query(query, initial_timeout=360)
    
    if data.get('elements'):
        with open(output_path, 'w') as f:
            json.dump(data, f)
        print(f"Downloaded raw roads to {output_path}")
    else:
        print("Warning: No road data retrieved. Using cache if exists.")

    return load_osm_roads(output_path)

if __name__ == "__main__":
    # quick test
    print("Data loading module loaded successfully")
    print("Available functions:")
    print("  - load_chirps_data()")
    print("  - load_dem()")
    print("  - load_osm_buildings()")
    print("  - load_osm_roads()")
    print("  - load_admin_boundaries()")
    print("  - validate_crs_match()")
