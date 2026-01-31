"""
Vector Analysis Module

GeoPandas, Shapely, and Fiona operations for building and road analysis.
Implements at least 3 required geospatial operations:
1. Spatial Join
2. Buffer Analysis
3. Density Calculation
Plus additional operations for comprehensive analysis.
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union
import numpy as np
from typing import Optional, Union, List

# ============================================================
# OPERATION 0: PRE-PROCESSING (CLIPPING)
# ============================================================

def clip_vectors_to_boundary(
    gdf: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Clip vector geometries to a boundary polygon.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input vector data (e.g., buildings, roads)
    boundary : gpd.GeoDataFrame
        Boundary polygon (e.g., district border)
    
    Returns
    -------
    gpd.GeoDataFrame
        Clipped geometries
    """
    # ensure crs match
    if gdf.crs != boundary.crs:
        boundary = boundary.to_crs(gdf.crs)
        
    # Spatial filter: Keep features that intersect the boundary
    # This preserves the full geometry of edge buildings instead of cutting them
    mask = gdf.intersects(boundary.unary_union)
    filtered = gdf[mask].copy()
    
    return filtered


# ============================================================
# OPERATION 1: SPATIAL JOIN
# ============================================================

def spatial_join_buildings_to_admin(
    buildings: gpd.GeoDataFrame,
    admin_boundaries: gpd.GeoDataFrame,
    admin_id_col: str = 'district_id'
) -> gpd.GeoDataFrame:
    """
    Assign administrative unit to each building using spatial join.
    
    Parameters
    ----------
    buildings : gpd.GeoDataFrame
        Building footprints
    admin_boundaries : gpd.GeoDataFrame
        Administrative boundary polygons
    admin_id_col : str
        Column name for admin unit identifier
    
    Returns
    -------
    gpd.GeoDataFrame
        Buildings with admin unit IDs assigned
    
    Example
    -------
    >>> buildings_with_district = spatial_join_buildings_to_admin(
    ...     buildings, districts, admin_id_col='district_id'
    ... )
    >>> # each building now has 'district_id' column
    """
    # perform spatial join - 'within' predicate
    joined = gpd.sjoin(
        buildings,
        admin_boundaries[[admin_id_col, 'geometry']],
        how='left',
        predicate='within'
    )
    
    # clean up index column from join
    if 'index_right' in joined.columns:
        joined = joined.drop(columns=['index_right'])
    
    return joined


# ============================================================
# OPERATION 2: BUFFER ANALYSIS
# ============================================================

def create_road_buffers(
    roads: gpd.GeoDataFrame,
    buffer_distance: float = 50.0,
    road_types: Optional[List[str]] = None
) -> gpd.GeoDataFrame:
    """
    Create buffer zones around roads for accessibility analysis.
    
    Parameters
    ----------
    roads : gpd.GeoDataFrame
        Road network linestrings
    buffer_distance : float
        Buffer distance in CRS units (meters if projected)
    road_types : list, optional
        Filter by highway types e.g. ['primary', 'secondary']
    
    Returns
    -------
    gpd.GeoDataFrame
        Buffered road polygons
    
    Example
    -------
    >>> road_buffers = create_road_buffers(roads, buffer_distance=100)
    >>> # areas within 100m of roads
    """
    # filter by road type if specified
    if road_types is not None and 'highway' in roads.columns:
        roads = roads[roads['highway'].isin(road_types)].copy()
    
    # create buffer
    buffered = roads.copy()
    buffered['geometry'] = roads.geometry.buffer(buffer_distance)
    
    return buffered


def identify_buildings_near_roads(
    buildings: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    distance: float = 50.0
) -> gpd.GeoDataFrame:
    """
    Find buildings within specified distance of roads.
    
    Parameters
    ----------
    buildings : gpd.GeoDataFrame
        Building footprints
    roads : gpd.GeoDataFrame
        Road network
    distance : float
        Distance threshold
    
    Returns
    -------
    gpd.GeoDataFrame
        Buildings near roads with 'near_road' flag
    """
    # create unified road buffer
    road_buffer = unary_union(roads.buffer(distance))
    
    # check which buildings intersect
    buildings = buildings.copy()
    buildings['near_road'] = buildings.geometry.intersects(road_buffer)
    
    return buildings


# ============================================================
# OPERATION 3: DENSITY CALCULATION
# ============================================================

def calculate_building_density(
    buildings: gpd.GeoDataFrame,
    admin_boundaries: gpd.GeoDataFrame,
    admin_id_col: str = 'district_id'
) -> gpd.GeoDataFrame:
    """
    Calculate building density metrics per administrative unit.
    
    Parameters
    ----------
    buildings : gpd.GeoDataFrame
        Building footprints
    admin_boundaries : gpd.GeoDataFrame
        Administrative boundary polygons
    admin_id_col : str
        Column for admin unit identifier
    
    Returns
    -------
    gpd.GeoDataFrame
        Admin boundaries with density metrics:
        - building_count: number of buildings
        - building_area_km2: total built-up area in km²
        - building_density: buildings per km²
    
    Example
    -------
    >>> districts_with_density = calculate_building_density(
    ...     buildings, districts, admin_id_col='district_id'
    ... )
    >>> print(districts_with_density[['district_id', 'building_density']])
    """
    # first do spatial join
    buildings_joined = spatial_join_buildings_to_admin(
        buildings, admin_boundaries, admin_id_col
    )
    
    # calculate building area
    buildings_joined['building_area'] = buildings_joined.geometry.area
    
    # aggregate by admin unit
    stats = buildings_joined.groupby(admin_id_col).agg(
        building_count=('building_area', 'count'),
        total_building_area=('building_area', 'sum')
    ).reset_index()
    
    # merge back to admin boundaries
    result = admin_boundaries.merge(stats, on=admin_id_col, how='left')
    
    # fill NaN with 0 (areas with no buildings)
    result['building_count'] = result['building_count'].fillna(0)
    result['total_building_area'] = result['total_building_area'].fillna(0)
    
    # calculate area in km² and density
    result['admin_area_km2'] = result.geometry.area / 1e6
    result['building_area_km2'] = result['total_building_area'] / 1e6
    result['building_density'] = result['building_count'] / result['admin_area_km2']
    
    return result


# ============================================================
# ADDITIONAL OPERATIONS
# ============================================================

def overlay_intersection(
    gdf1: gpd.GeoDataFrame,
    gdf2: gpd.GeoDataFrame,
    keep_geom_type: bool = True
) -> gpd.GeoDataFrame:
    """
    Find intersection between two GeoDataFrames.
    
    Parameters
    ----------
    gdf1 : gpd.GeoDataFrame
        First dataset
    gdf2 : gpd.GeoDataFrame
        Second dataset
    keep_geom_type : bool
        Keep only geometries of same type as gdf1
    
    Returns
    -------
    gpd.GeoDataFrame
        Intersected geometries with attributes from both
    
    Example
    -------
    >>> exposed_buildings = overlay_intersection(buildings, flood_zones)
    """
    return gpd.overlay(gdf1, gdf2, how='intersection', keep_geom_type=keep_geom_type)


def calculate_centroid_coordinates(
    gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Add centroid coordinates to polygons.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input polygons
    
    Returns
    -------
    gpd.GeoDataFrame
        Original data with 'centroid_x' and 'centroid_y' columns
    """
    gdf = gdf.copy()
    centroids = gdf.geometry.centroid
    gdf['centroid_x'] = centroids.x
    gdf['centroid_y'] = centroids.y
    return gdf


def filter_by_area(
    gdf: gpd.GeoDataFrame,
    min_area: Optional[float] = None,
    max_area: Optional[float] = None
) -> gpd.GeoDataFrame:
    """
    Filter geometries by area.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input geometries
    min_area : float, optional
        Minimum area threshold
    max_area : float, optional
        Maximum area threshold
    
    Returns
    -------
    gpd.GeoDataFrame
        Filtered geometries
    """
    areas = gdf.geometry.area
    mask = pd.Series(True, index=gdf.index)
    
    if min_area is not None:
        mask = mask & (areas >= min_area)
    if max_area is not None:
        mask = mask & (areas <= max_area)
    
    return gdf[mask].copy()


def dissolve_by_attribute(
    gdf: gpd.GeoDataFrame,
    by: str,
    aggfunc: str = 'sum'
) -> gpd.GeoDataFrame:
    """
    Dissolve geometries by attribute.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input geometries
    by : str
        Column to group by
    aggfunc : str
        Aggregation function for numeric columns
    
    Returns
    -------
    gpd.GeoDataFrame
        Dissolved geometries
    """
    return gdf.dissolve(by=by, aggfunc=aggfunc)


def read_vector_with_fiona(
    filepath: str,
    layer: Optional[str] = None
) -> gpd.GeoDataFrame:
    """
    Read vector data using Fiona backend.
    
    Parameters
    ----------
    filepath : str
        Path to vector file
    layer : str, optional
        Layer name for multi-layer files
    
    Returns
    -------
    gpd.GeoDataFrame
        Vector data as GeoDataFrame
    
    Example
    -------
    >>> # demonstrates Fiona read as required by assignment
    >>> buildings = read_vector_with_fiona('data/buildings.gpkg', layer='buildings')
    """
    # using fiona explicitly as required
    with fiona.open(filepath, layer=layer) as src:
        crs = src.crs
        records = list(src)
    
    # convert to geopandas
    gdf = gpd.GeoDataFrame.from_features(records, crs=crs)
    
    return gdf


def write_vector_with_fiona(
    gdf: gpd.GeoDataFrame,
    filepath: str,
    driver: str = 'GPKG'
) -> None:
    """
    Write vector data using Fiona backend.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Data to write
    filepath : str
        Output file path
    driver : str
        Output format driver (default: GeoPackage)
    """
    # define schema from geodataframe
    schema = gpd.io.file.infer_schema(gdf)
    
    with fiona.open(filepath, 'w', driver=driver, crs=gdf.crs, schema=schema) as dst:
        for idx, row in gdf.iterrows():
            dst.write({
                'geometry': row.geometry.__geo_interface__,
                'properties': {k: v for k, v in row.items() if k != 'geometry'}
            })


if __name__ == "__main__":
    print("Vector analysis module loaded successfully")
    print("\nRequired Operations (3+):")
    print("  1. spatial_join_buildings_to_admin()")
    print("  2. create_road_buffers()")
    print("  3. calculate_building_density()")
    print("\nAdditional Operations:")
    print("  - overlay_intersection()")
    print("  - calculate_centroid_coordinates()")
    print("  - filter_by_area()")
    print("  - dissolve_by_attribute()")
    print("  - read_vector_with_fiona()")
    print("  - write_vector_with_fiona()")
