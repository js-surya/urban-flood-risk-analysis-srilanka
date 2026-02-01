"""
Integration Module

Raster-Vector integration operations including:
- Zonal statistics (raster to vector)
- Rasterization (vector to raster)
- Bidirectional data transfer
"""

import numpy as np
import xarray as xr
import geopandas as gpd
import pandas as pd
from rasterstats import zonal_stats, point_query
from geocube.api.core import make_geocube
import rasterio
from rasterio import features
from typing import Union, Optional, List, Dict, Tuple
from pathlib import Path


# ============================================================
# RASTER TO VECTOR (Zonal Statistics)
# ============================================================

def extract_zonal_statistics(
    vector: gpd.GeoDataFrame,
    raster_path: Union[str, Path, xr.DataArray],
    stats: List[str] = ['mean', 'max', 'min', 'std'],
    prefix: str = ''
) -> gpd.GeoDataFrame:
    """
    Extract raster statistics for each polygon feature.
    
    Parameters
    ----------
    vector : gpd.GeoDataFrame
        Polygon features for zonal analysis
    raster_path : str, Path, or xr.DataArray
        Path to raster file or xarray DataArray
    stats : list
        Statistics to calculate (options: mean, max, min, std, count, sum, median, etc.)
    prefix : str
        Prefix for output column names
    
    Returns
    -------
    gpd.GeoDataFrame
        Input features with statistics columns added
    
    Example
    -------
    >>> districts = extract_zonal_statistics(
    ...     districts, 'rainfall_max.tif',
    ...     stats=['mean', 'max'], prefix='rainfall_'
    ... )
    >>> # districts now has 'rainfall_mean' and 'rainfall_max' columns
    >>> # Or with a DataArray:
    >>> dist_da = xr.DataArray(dist_to_water, coords=water_raster.coords, dims=water_raster.dims)
    >>> districts = extract_zonal_statistics(
    ...     districts, dist_da,
    ...     stats=['mean'], prefix='proximity_'
    ... )
    """
    # Handle DataArray input
    if isinstance(raster_path, xr.DataArray):
        raster_da = raster_path
        raster_crs = raster_da.rio.crs if hasattr(raster_da, 'rio') else None
        data = raster_da.values
        nodata = -9999  # Default nodata value for DataArray
        
        # Get affine transform from coordinates
        coords = raster_da.coords
        if 'latitude' in coords and 'longitude' in coords:
            lons = coords['longitude'].values
            lats = coords['latitude'].values
        elif 'y' in coords and 'x' in coords:
            lons = coords['x'].values
            lats = coords['y'].values
        else:
            raise ValueError("DataArray must have latitude/longitude or x/y coordinates")
        
        # Build transform
        res_x = abs(float(lons[1] - lons[0])) if len(lons) > 1 else 0.01
        res_y = abs(float(lats[1] - lats[0])) if len(lats) > 1 else 0.01
        
        from rasterio.transform import from_origin
        minx, maxx = float(lons.min()), float(lons.max())
        miny, maxy = float(lats.min()), float(lats.max())
        
        # Handle both north-up and south-up orientations
        if lats[0] < lats[-1]:  # South-up (ascending)
            data = data[::-1, :]
            transform = from_origin(minx, maxy, res_x, res_y)
        else:  # North-up (descending)
            transform = from_origin(minx, maxy, res_x, res_y)
        
        if raster_crs is None:
            raster_crs = 'EPSG:4326'  # Default to WGS84
        
        bounds_minx, bounds_maxx = minx, maxx
        bounds_miny, bounds_maxy = miny, maxy
    else:
        # Handle file path input (existing logic)
        raster_path = Path(raster_path)

        # Open raster once to grab CRS/bounds; clip and reproject vector to avoid zero-sized windows
        with rasterio.open(raster_path) as src:
            raster_crs = src.crs
            bounds = src.bounds
            data = src.read(1)
            res_x = abs(src.transform.a)
            res_y = abs(src.transform.e)
            nodata = src.nodata if src.nodata is not None else -9999

            # Normalize bounds so min < max
            bounds_minx, bounds_maxx = sorted([bounds.left, bounds.right])
            bounds_miny, bounds_maxy = sorted([bounds.bottom, bounds.top])

            # If raster is south-up (positive y pixel size), flip to north-up for rasterstats
            from rasterio.transform import from_origin
            if src.transform.e > 0:
                data = data[::-1, :]
            transform = from_origin(bounds_minx, bounds_maxy, res_x, res_y)

    gdf = vector.copy()

    # Reproject to raster CRS if needed
    if gdf.crs is not None and raster_crs is not None and gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)

    # Clip geometries to raster extent to prevent empty windows
    from shapely.geometry import box
    raster_bounds_geom = gpd.GeoSeries([box(bounds_minx, bounds_miny, bounds_maxx, bounds_maxy)], crs=raster_crs)
    gdf = gpd.clip(gdf, raster_bounds_geom)
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()]

    if gdf.empty:
        raise ValueError("No vector features overlap the raster extent; cannot compute zonal statistics.")

    # calculate zonal stats using in-memory array to avoid malformed transforms
    zone_stats = zonal_stats(
        gdf.geometry,
        data,
        affine=transform,
        stats=stats,
        nodata=nodata,
        all_touched=True,
        boundless=True
    )
    
    # convert to dataframe and add prefix
    stats_df = pd.DataFrame(zone_stats)
    if prefix:
        stats_df.columns = [f'{prefix}{col}' for col in stats_df.columns]
    
    # join to original geodataframe
    result = vector.copy()
    for col in stats_df.columns:
        result[col] = stats_df[col].values
    
    return result


def sample_raster_at_points(
    points: gpd.GeoDataFrame,
    raster_path: Union[str, Path],
    column_name: str = 'raster_value'
) -> gpd.GeoDataFrame:
    """
    Sample raster values at point locations.
    
    Parameters
    ----------
    points : gpd.GeoDataFrame
        Point features (or polygons - centroids will be used)
    raster_path : str or Path
        Path to raster file
    column_name : str
        Name for output column
    
    Returns
    -------
    gpd.GeoDataFrame
        Points with raster values
    
    Example
    -------
    >>> buildings = sample_raster_at_points(
    ...     buildings, 'rainfall_p95.tif', column_name='rainfall_exposure'
    ... )
    """
    # use centroids for polygons
    sample_points = points.geometry.centroid
    
    # query raster at each point
    values = point_query(
        sample_points,
        str(raster_path),
        nodata=-9999
    )
    
    result = points.copy()
    result[column_name] = values
    
    return result


def assign_raster_classes(
    vector: gpd.GeoDataFrame,
    raster_path: Union[str, Path],
    breaks: List[float],
    labels: List[str],
    column_name: str = 'risk_class',
    stat: str = 'mean'
) -> gpd.GeoDataFrame:
    """
    Classify features based on zonal raster statistics.
    
    Parameters
    ----------
    vector : gpd.GeoDataFrame
        Input features
    raster_path : str or Path
        Path to raster
    breaks : list
        Classification break values
    labels : list
        Class labels (one more than breaks)
    column_name : str
        Output column name
    stat : str
        Statistic to use for classification
    
    Returns
    -------
    gpd.GeoDataFrame
        Features with classification
    
    Example
    -------
    >>> districts = assign_raster_classes(
    ...     districts, 'rainfall.tif',
    ...     breaks=[50, 100, 150],
    ...     labels=['Low', 'Moderate', 'High', 'Extreme']
    ... )
    """
    # get zonal stat
    vector = extract_zonal_statistics(vector, raster_path, stats=[stat])
    
    # classify
    values = vector[stat].values
    classes = pd.cut(values, bins=[-np.inf] + breaks + [np.inf], labels=labels)
    
    result = vector.copy()
    result[column_name] = classes
    
    return result


# ============================================================
# VECTOR TO RASTER (Rasterization)
# ============================================================

def rasterize_vector(
    vector: gpd.GeoDataFrame,
    value_column: Optional[str] = None,
    resolution: Tuple[float, float] = (-0.01, 0.01),
    bounds: Optional[Tuple[float, float, float, float]] = None,
    fill_value: float = 0,
    dtype: str = 'float32'
) -> xr.DataArray:
    """
    Convert vector features to raster using geocube.
    
    Parameters
    ----------
    vector : gpd.GeoDataFrame
        Input vector features
    value_column : str, optional
        Column containing values to rasterize. If None, creates a binary mask (1 where features exist).
    resolution : tuple
        Output resolution as (x_res, y_res), typically (-res, res)
    bounds : tuple, optional
        Output bounds (minx, miny, maxx, maxy)
    fill_value : float
        Value for areas without features
    dtype : str
        Output data type
    
    Returns
    -------
    xr.DataArray
        Rasterized values
    
    Example
    -------
    >>> building_density_raster = rasterize_vector(
    ...     districts, 'building_density',
    ...     resolution=(-0.01, 0.01)  # ~1km resolution
    ... )
    >>> water_mask = rasterize_vector(
    ...     water_bodies, value_column=None,  # Binary mask
    ...     resolution=(-0.0001, 0.0001)
    ... )
    """
    # If no value column specified, create a binary mask
    if value_column is None:
        # Add a temporary column with value 1 for all features
        vector = vector.copy()
        vector['_mask_'] = 1
        value_column = '_mask_'
    
    # make geocube
    cube = make_geocube(
        vector_data=vector,
        measurements=[value_column],
        resolution=resolution,
        fill=fill_value
    )
    
    return cube[value_column]


def rasterize_with_rasterio(
    vector: gpd.GeoDataFrame,
    reference_raster: Union[str, Path],
    value_column: Optional[str] = None,
    default_value: int = 1
) -> np.ndarray:
    """
    Rasterize vector to match reference raster grid.
    
    Parameters
    ----------
    vector : gpd.GeoDataFrame
        Input vector features
    reference_raster : str or Path
        Path to reference raster for grid alignment
    value_column : str, optional
        Column for values. If None, uses default_value
    default_value : int
        Value to assign if no value_column
    
    Returns
    -------
    np.ndarray
        Rasterized array matching reference grid
    
    Example
    -------
    >>> building_mask = rasterize_with_rasterio(
    ...     buildings, 'reference.tif', default_value=1
    ... )
    """
    # open reference to get transform and shape
    with rasterio.open(reference_raster) as src:
        transform = src.transform
        out_shape = (src.height, src.width)
    
    # prepare shapes
    if value_column is not None:
        shapes = ((geom, value) for geom, value in 
                  zip(vector.geometry, vector[value_column]))
    else:
        shapes = ((geom, default_value) for geom in vector.geometry)
    
    # rasterize
    rasterized = features.rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype='float32'
    )
    
    return rasterized


# ============================================================
# BIDIRECTIONAL INTEGRATION
# ============================================================

def calculate_vulnerability_scores(
    admin_boundaries: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame,
    rainfall_raster: Union[str, Path],
    dem_raster: Union[str, Path],
    weights: Tuple[float, float, float] = (0.4, 0.3, 0.3)
) -> gpd.GeoDataFrame:
    """
    Calculate composite vulnerability scores per admin unit.
    
    Combines:
    - Rainfall intensity from raster (raster -> vector)
    - Building density from vector
    - Elevation from raster (raster -> vector)
    
    Parameters
    ----------
    admin_boundaries : gpd.GeoDataFrame
        Administrative unit polygons
    buildings : gpd.GeoDataFrame
        Building footprints
    rainfall_raster : str or Path
        Path to rainfall raster (e.g., annual max)
    dem_raster : str or Path
        Path to DEM raster
    weights : tuple
        Weights for (rainfall, building_density, elevation)
    
    Returns
    -------
    gpd.GeoDataFrame
        Admin units with vulnerability scores
    
    Example
    -------
    >>> districts = calculate_vulnerability_scores(
    ...     districts, buildings, 'rainfall.tif', 'dem.tif'
    ... )
    >>> # districts now has 'vulnerability_score' column
    """
    from src.vector_analysis import calculate_building_density
    
    # Step 1: Calculate building density (vector processing)
    admin_with_buildings = calculate_building_density(
        buildings, admin_boundaries,
        admin_id_col='district_id' if 'district_id' in admin_boundaries.columns else admin_boundaries.columns[0]
    )
    
    # Step 2: Extract rainfall stats (raster -> vector)
    admin_with_rainfall = extract_zonal_statistics(
        admin_with_buildings, rainfall_raster,
        stats=['max'], prefix='rainfall_'
    )
    
    # Step 3: Extract elevation stats (raster -> vector)
    admin_with_elev = extract_zonal_statistics(
        admin_with_rainfall, dem_raster,
        stats=['mean'], prefix='elevation_'
    )
    
    # Step 4: Normalize values
    def normalize(series):
        return (series - series.min()) / (series.max() - series.min() + 1e-10)
    
    result = admin_with_elev.copy()
    
    rainfall_norm = normalize(result['rainfall_max'])
    density_norm = normalize(result['building_density'])
    elev_norm = normalize(result['elevation_mean'])
    
    # Step 5: Calculate composite vulnerability
    # Note: invert elevation (lower = more vulnerable)
    w_rain, w_building, w_elev = weights
    
    result['vulnerability_score'] = (
        w_rain * rainfall_norm +
        w_building * density_norm +
        w_elev * (1 - elev_norm)  # invert elevation
    )
    
    # Classify vulnerability
    result['vulnerability_class'] = pd.cut(
        result['vulnerability_score'],
        bins=[0, 0.3, 0.5, 0.7, 1.0],
        labels=['Low', 'Moderate', 'High', 'Extreme']
    )
    
    return result


def create_vulnerability_raster(
    vulnerability_vector: gpd.GeoDataFrame,
    resolution: Tuple[float, float] = (-0.01, 0.01),
    value_column: str = 'vulnerability_score'
) -> xr.DataArray:
    """
    Convert vulnerability scores to raster (vector -> raster).
    
    Parameters
    ----------
    vulnerability_vector : gpd.GeoDataFrame
        Admin units with vulnerability scores
    resolution : tuple
        Output raster resolution
    value_column : str
        Column containing vulnerability scores
    
    Returns
    -------
    xr.DataArray
        Vulnerability raster
    """
    return rasterize_vector(
        vulnerability_vector,
        value_column,
        resolution
    )


if __name__ == "__main__":
    print("Integration module loaded successfully")
    print("\nRaster -> Vector:")
    print("  - extract_zonal_statistics()")
    print("  - sample_raster_at_points()")
    print("  - assign_raster_classes()")
    print("\nVector -> Raster:")
    print("  - rasterize_vector()")
    print("  - rasterize_with_rasterio()")
    print("\nBidirectional:")
    print("  - calculate_vulnerability_scores()")
    print("  - create_vulnerability_raster()")
