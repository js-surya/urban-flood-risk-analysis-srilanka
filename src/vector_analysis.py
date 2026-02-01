"""
Vector Analysis Module

GeoPandas, Shapely, and Fiona operations for building and road analysis.
Implements at least 3 required geospatial operations:
1. Spatial Join
2. Buffer Analysis
3. Density Calculation
Plus interactive map generation using Folium.
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union
import numpy as np
from typing import Optional, Union, List, Dict
import folium
from folium import plugins
import xarray as xr


# ============================================================
# OPERATION 0: PRE-PROCESSING (CLIPPING)
# ============================================================

def clip_vectors_to_boundary(
    gdf: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Clip vector geometries to a boundary polygon.
    """
    # ensure crs match
    if gdf.crs != boundary.crs:
        boundary = boundary.to_crs(gdf.crs)
        
    # Spatial filter: Keep features that intersect the boundary
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
    """
    # ensure area calc is valid (projected CRS)
    if buildings.crs is not None and buildings.crs.is_geographic:
        # temporary projection for area calc (World Mercator)
        build_proj = buildings.to_crs(epsg=3395)
        admin_proj = admin_boundaries.to_crs(epsg=3395)
    else:
        build_proj = buildings
        admin_proj = admin_boundaries
        
    # calculate areas
    admin_proj = admin_proj.copy()
    admin_proj['area_sqkm'] = admin_proj.geometry.area / 1e6
    area_lookup = admin_proj[[admin_id_col, 'area_sqkm']]
    
    # join to count
    joined = gpd.sjoin(build_proj, admin_proj[[admin_id_col, 'geometry']], how='left', predicate='within')
    counts = joined.groupby(admin_id_col).size().reset_index(name='building_count')
    
    # merge back
    result = admin_boundaries.merge(counts, on=admin_id_col, how='left')
    result = result.merge(area_lookup, on=admin_id_col, how='left')
    result['building_count'] = result['building_count'].fillna(0)
    result['density_per_sqkm'] = result['building_count'] / result['area_sqkm']
    
    return result


# ============================================================
# OPERATION 4: RISK SAMPLING & INTERACTIVE MAP
# ============================================================

def sample_raster_values(
    gdf: gpd.GeoDataFrame,
    raster: xr.DataArray,
    column_name: str = 'raster_val'
) -> gpd.GeoDataFrame:
    """
    Sample raster values at the centroid of each polygon.
    """
    if gdf.empty:
        return gdf.assign(**{column_name: []})

    # Normalize raster to xarray with rio accessor
    if not hasattr(raster, 'rio'):
        raster = xr.DataArray(raster)

    # Ensure CRS match
    if gdf.crs is not None and hasattr(raster, 'rio') and raster.rio.crs is not None and gdf.crs != raster.rio.crs:
        gdf = gdf.to_crs(raster.rio.crs)
    
    # Convert DataArray to numpy array and get affine transform
    if hasattr(raster, 'rio'):
        # It's a rioxarray DataArray
        data = raster.values
        
        # Build affine transform from coordinates
        coords = raster.coords
        if 'longitude' in coords and 'latitude' in coords:
            lons = coords['longitude'].values
            lats = coords['latitude'].values
        elif 'x' in coords and 'y' in coords:
            lons = coords['x'].values
            lats = coords['y'].values
        else:
            raise ValueError(f"Raster must have either ('x', 'y') or ('longitude', 'latitude') coordinates. Found: {list(coords.keys())}")
        
        res_x = abs(float(lons[1] - lons[0])) if len(lons) > 1 else 0.01
        res_y = abs(float(lats[1] - lats[0])) if len(lats) > 1 else 0.01
        
        from rasterio.transform import from_origin
        minx, maxx = float(lons.min()), float(lons.max())
        miny, maxy = float(lats.min()), float(lats.max())
        
        # Handle north-up vs south-up
        if lats[0] < lats[-1]:  # South-up
            data = data[::-1, :]
            transform = from_origin(minx, maxy, res_x, res_y)
        else:  # North-up
            transform = from_origin(minx, maxy, res_x, res_y)
        
        # Use point_query from rasterstats for point sampling
        from rasterstats import point_query
        values = [point_query(pt, data, affine=transform, nodata=-9999) for pt in gdf.geometry]
    else:
        raise TypeError("Raster must be an xarray DataArray with rio accessor")
    
    gdf = gdf.copy()
    gdf[column_name] = values
    
    return gdf


def assign_risk_category(
    gdf: gpd.GeoDataFrame,
    risk_col: str = 'risk_score',
    typology_col: Optional[str] = 'amenity'
) -> gpd.GeoDataFrame:
    """
    Classify risk into categories and apply typology multipliers.
    """
    gdf = gdf.copy()
    
    # 1. Apply Typology Multiplier (if available)
    gdf['final_risk'] = gdf[risk_col]
    if typology_col in gdf.columns:
        critical_infra = ['hospital', 'school', 'police', 'fire_station']
        # If amenity is critical, multiply risk by 1.5 (subject to ceiling of 1.0)
        mask_critical = gdf[typology_col].isin(critical_infra)
        gdf.loc[mask_critical, 'final_risk'] *= 1.5
    
    # Clip to max 1.0
    gdf['final_risk'] = gdf['final_risk'].clip(upper=1.0)
    
    # 2. Classify
    conditions = [
        (gdf['final_risk'] >= 0.7),
        (gdf['final_risk'] >= 0.4) & (gdf['final_risk'] < 0.7),
        (gdf['final_risk'] < 0.4)
    ]
    choices = ['High Risk', 'Medium Risk', 'Low Risk']
    
    gdf['risk_category'] = np.select(conditions, choices, default='Low Risk')
    
    return gdf


def generate_interactive_risk_map(
    buildings: gpd.GeoDataFrame,
    output_path: str = "interactive_risk_map.html"
):
    """
    Generate a Folium map with toggleable risk layers.
    """
    # Center map
    center_lat = buildings.geometry.centroid.y.mean()
    center_lon = buildings.geometry.centroid.x.mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles='CartoDB dark_matter')
    
    # Define styles
    colors = {
        'High Risk': '#ff0000',   # Red
        'Medium Risk': '#ffa500', # Orange
        'Low Risk': '#00ff00'     # Green
    }
    
    # Create FeatureGroups for toggle control
    fg_high = folium.FeatureGroup(name='High Risk Buildings')
    fg_med = folium.FeatureGroup(name='Medium Risk Buildings')
    fg_low = folium.FeatureGroup(name='Low Risk Buildings')
    
    # Helper to add features
    def add_to_group(subset, group, color):
        for idx, row in subset.iterrows():
            # Create popup content
            risk_score = f"{row.get('final_risk', 0):.2f}"
            cat = row.get('risk_category', 'Unknown')
            b_id = row.get('building_id', 'N/A')
            popup_html = f"<b>ID:</b> {b_id}<br><b>Risk:</b> {cat} ({risk_score})"
            
            # Add simple marker or polygon? 
            # For 4000 buildings, circle markers are faster than polygons
            if row.geometry.geom_type == 'Point':
                loc = [row.geometry.y, row.geometry.x]
            else:
                loc = [row.geometry.centroid.y, row.geometry.centroid.x]
                
            folium.CircleMarker(
                location=loc,
                radius=4,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=popup_html
            ).add_to(group)

    # Filter and add
    high_risk = buildings[buildings['risk_category'] == 'High Risk']
    med_risk = buildings[buildings['risk_category'] == 'Medium Risk']
    low_risk = buildings[buildings['risk_category'] == 'Low Risk']
    
    add_to_group(high_risk, fg_high, colors['High Risk'])
    add_to_group(med_risk, fg_med, colors['Medium Risk'])
    add_to_group(low_risk, fg_low, colors['Low Risk'])
    
    # Add to map
    fg_high.add_to(m)
    fg_med.add_to(m)
    fg_low.add_to(m)
    
    # Add Layer Control to enable toggling
    folium.LayerControl(collapsed=False).add_to(m)
    
    # Save
    m.save(output_path)
    return output_path


def generate_interactive_risk_map_clustered(
    buildings: gpd.GeoDataFrame,
    output_path: str = "interactive_risk_map_light.html",
    max_points_per_category: int = 20000,
    include_low: bool = True
):
    """
    Lightweight interactive map using clustering to keep file size manageable.

    Parameters
    ----------
    buildings : gpd.GeoDataFrame
        Buildings with columns 'final_risk' and 'risk_category'.
    output_path : str
        Destination HTML file.
    max_points_per_category : int
        Limit points per category to avoid huge HTML.
    include_low : bool
        Whether to include Low Risk points.
    """
    if buildings.empty:
        raise ValueError("No buildings supplied for mapping.")

    # Ensure WGS84 for web map
    if buildings.crs is not None and buildings.crs.to_epsg() != 4326:
        buildings = buildings.to_crs(epsg=4326)

    colors = {
        'High Risk': '#ff0000',   # Red
        'Medium Risk': '#ffa500', # Orange
        'Low Risk': '#00ff00'     # Green
    }

    # Sort by risk to keep highest first, then cap per category
    def subset(cat):
        df = buildings[buildings['risk_category'] == cat].copy()
        if 'final_risk' in df.columns:
            df = df.sort_values('final_risk', ascending=False)
        return df.head(max_points_per_category)

    high = subset('High Risk')
    med = subset('Medium Risk')
    low = subset('Low Risk') if include_low else gpd.GeoDataFrame(columns=buildings.columns, crs=buildings.crs)

    m = folium.Map(
        location=[buildings.geometry.centroid.y.mean(), buildings.geometry.centroid.x.mean()],
        zoom_start=12,
        tiles='CartoDB positron'
    )

    def add_cluster(df, name, color):
        if df.empty:
            return
        # prepare locations and popups
        locs = [[geom.y, geom.x] for geom in df.geometry.centroid]
        popups = [
            f"<b>ID:</b> {row.get('building_id', 'N/A')}<br>"
            f"<b>Risk:</b> {row.get('risk_category', '')} ({row.get('final_risk', 0):.2f})"
            for _, row in df.iterrows()
        ]
        fg = folium.FeatureGroup(name=name)
        plugins.FastMarkerCluster(
            data=locs,
            popups=popups,
            icon_create_function=None,  # default cluster icon
        ).add_to(fg)
        fg.add_to(m)

    add_cluster(high, "High Risk (clustered)", colors['High Risk'])
    add_cluster(med, "Medium Risk (clustered)", colors['Medium Risk'])
    if include_low:
        add_cluster(low, "Low Risk (clustered)", colors['Low Risk'])

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(output_path)
    return output_path


def generate_interactive_risk_map_geojson(
    buildings: gpd.GeoDataFrame,
    output_path: str = "interactive_risk_map_polygons.html",
    include_low: bool = False,
    max_features: int = 50000,
    simplify_tolerance: float = 0.00005
):
    """
    Interactive map with actual building geometries (colored by risk).
    Uses GeoJSON layer per category; caps feature count and simplifies geometry
    to keep file size manageable.

    Parameters
    ----------
    buildings : gpd.GeoDataFrame
        Must contain columns: geometry, risk_category, final_risk (and optional building_id)
    output_path : str
        Destination HTML file
    include_low : bool
        Whether to include Low Risk buildings (can be huge)
    max_features : int
        Maximum total features to include (highest risk first)
    simplify_tolerance : float
        Geometry simplification tolerance in degrees (0 disables)
    """
    if buildings.empty:
        raise ValueError("No buildings supplied for mapping.")

    # Ensure required columns
    gdf = buildings.copy()
    if 'building_id' not in gdf.columns:
        gdf['building_id'] = gdf.index.astype(str)
    if 'final_risk' not in gdf.columns:
        gdf['final_risk'] = gdf.get('risk_score', 0)

    # CRS -> WGS84 for web maps
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    # Filter categories
    if not include_low:
        gdf = gdf[gdf['risk_category'] != 'Low Risk']

    # Sort by risk descending and cap total features
    gdf = gdf.sort_values('final_risk', ascending=False)
    if len(gdf) > max_features:
        gdf = gdf.head(max_features)

    # Simplify geometries to reduce size
    if simplify_tolerance and simplify_tolerance > 0:
        gdf['geometry'] = gdf.geometry.simplify(simplify_tolerance, preserve_topology=True)

    colors = {
        'High Risk': '#ff0000',
        'Medium Risk': '#ffa500',
        'Low Risk': '#00ff00'
    }

    m = folium.Map(
        location=[gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()],
        zoom_start=13,
        tiles='CartoDB positron'
    )

    def add_layer(cat):
        subset = gdf[gdf['risk_category'] == cat]
        if subset.empty:
            return
        layer = folium.FeatureGroup(name=f"{cat} (polygons)")
        folium.GeoJson(
            subset,
            style_function=lambda feat, c=colors.get(cat, '#3388ff'): {
                "color": c,
                "weight": 1,
                "fillColor": c,
                "fillOpacity": 0.6
            },
            highlight_function=lambda feat: {"weight": 2, "fillOpacity": 0.8},
            tooltip=folium.GeoJsonTooltip(
                fields=['building_id', 'risk_category', 'final_risk'],
                aliases=['ID', 'Category', 'Risk Score'],
                localize=True,
                sticky=True
            )
        ).add_to(layer)
        layer.add_to(m)

    add_layer('High Risk')
    add_layer('Medium Risk')
    if include_low:
        add_layer('Low Risk')

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(output_path)
    return output_path
