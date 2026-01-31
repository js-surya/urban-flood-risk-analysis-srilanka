"""
Data Loading Module

Functions for loading various geospatial datasets:
- CHIRPS rainfall data (NetCDF)
- SRTM DEM (GeoTIFF)
- OpenStreetMap features (Shapefile/GeoPackage)
"""



import xarray as xr
import rioxarray  # noqa: F401
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
    
    # Merge tiles
    merged = xr.concat(tiles, dim='y')
    merged = merged.sortby('y', ascending=False)
    merged = merged.sortby('x')
    
    # helper: drop duplicates if any (common with tile overlap)
    _, index = np.unique(merged['x'], return_index=True)
    merged = merged.isel(x=index)
    _, index = np.unique(merged['y'], return_index=True)
    merged = merged.isel(y=index[::-1]) # Keep descending sort for y
    
    # Clip to bbox if provided
    if bbox is not None:
        west, south, east, north = bbox
        merged = merged.sel(x=slice(west, east), y=slice(north, south))
    
    merged.name = 'elevation'
    # Assign CRS (SRTM is always WGS84)
    merged.rio.write_crs("EPSG:4326", inplace=True)
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
    roads = gpd.read_file(filepath)
    
    # filter by road type if specified
    if road_types is not None and 'highway' in roads.columns:
        roads = roads[roads['highway'].isin(road_types)]
    
    return roads


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
