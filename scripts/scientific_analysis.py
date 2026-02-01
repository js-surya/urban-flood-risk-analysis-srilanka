
import os
import json
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import LineString, Point
from pathlib import Path

# Import project modules
import src.data_loading as dl
import src.raster_analysis as ra
import src.vector_analysis as va
import src.integration as integration

def run_scientific_analysis():
    print("=== Starting Scientific Improvement Analysis (Stabilized) ===")
    
    # Setup paths
    BASE_DIR = Path(".")
    RAW_DIR = BASE_DIR / "data" / "raw"
    OUTPUT_DIR = BASE_DIR / "outputs"
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # 1. Load Data
    print("Loading core datasets...")
    admin_path = RAW_DIR / "admin" / "colombo_wards.json"
    admin_gdf = gpd.read_file(admin_path)
    
    buildings_gdf = dl.load_osm_buildings(RAW_DIR / "buildings" / "osm_buildings.json")
    print(f"Loaded {len(buildings_gdf)} buildings.")
    
    # Elevation - Load SRTM
    elev_path = RAW_DIR / "srtm" / "srtm_52_11.tif"
    elevation_ds = xr.open_dataset(elev_path, engine="rasterio")
    elevation = elevation_ds.band_data.squeeze()
    if elevation.rio.crs is None:
        elevation.rio.write_crs("EPSG:4326", inplace=True)
    
    # Rainfall
    rain_path = RAW_DIR / "chirps" / "chirps-v2.0.2025.monthly.nc"
    rainfall_cube = xr.open_dataset(rain_path).precip
    if rainfall_cube.rio.crs is None:
        rainfall_cube.rio.write_crs("EPSG:4326", inplace=True)
    
    # 2. Waterways & Proximity Layer
    print("Processing river proximity (WGS84)...")
    water_path = RAW_DIR / "landuse" / "colombo_water_raw.json"
    if water_path.exists():
        with open(water_path, 'r') as f:
            raw_water = json.load(f)
        
        water_feats = []
        for el in raw_water.get('elements', []):
            if el['type'] == 'way' and 'geometry' in el:
                coords = [(p['lon'], p['lat']) for p in el['geometry']]
                if len(coords) >= 2:
                    water_feats.append({'geometry': LineString(coords), 'type': 'waterway'})
        
        if water_feats:
            water_gdf = gpd.GeoDataFrame(water_feats, crs='EPSG:4326')
            
            # Rasterize (Targeting same grid as elevation)
            water_mask = integration.rasterize_with_rasterio(water_gdf, str(elev_path))
            target_idx = np.where(water_mask > 0)
            
            if len(target_idx[0]) > 0:
                # Calculate distance in Degrees
                res_deg = abs(elevation.rio.transform().a)
                dist_deg = ra.calculate_euclidean_distance(elevation.shape, target_idx, pixel_size=res_deg)
                
                # Conversion to meters approx
                dist_meters = dist_deg * 111000
                
                # Back to xarray
                dist_risk_raw = xr.DataArray(dist_meters, coords=elevation.coords, dims=elevation.dims)
                dist_risk_raw.rio.write_crs("EPSG:4326", inplace=True)
                dist_risk = 1 - ra.normalize_array(dist_risk_raw, 'minmax')
                print("River proximity layer generated.")
            else:
                print("No water features in extent.")
                dist_risk = xr.zeros_like(elevation)
                dist_risk_raw = xr.full_like(elevation, 10000)
        else:
            print("No water features parsed.")
            dist_risk = xr.zeros_like(elevation)
            dist_risk_raw = xr.full_like(elevation, 10000)
    else:
        print("Water data file missing.")
        dist_risk = xr.zeros_like(elevation)
        dist_risk_raw = xr.full_like(elevation, 10000)

    # 3. Model Refinement (AHP)
    print("Performing AHP Weighted Overlay...")
    # Fix Rainfall CRS before reproject
    rain_max = rainfall_cube.max(dim='time')
    if rain_max.rio.crs is None:
        rain_max.rio.write_crs("EPSG:4326", inplace=True)
    
    rain_max_aligned = rain_max.rio.reproject_match(elevation)
    rain_smooth = ra.smooth_raster(rain_max_aligned, method='gaussian', size=1)
    rain_risk = ra.normalize_array(rain_smooth, 'minmax')
    
    elev_risk = 1 - ra.normalize_array(elevation, 'minmax')
    
    # Weighted Hazard: 35% Rain, 35% Proximity, 30% Elevation
    hazard_map = (0.35 * rain_risk) + (0.35 * dist_risk) + (0.3 * elev_risk)
    
    # 5. Risk Scoring & Validation
    print("Joining exposure data...")
    buildings_joined = va.spatial_join_buildings_to_admin(buildings_gdf, admin_gdf, admin_id_col='id')
    
    # Filter to Colombo boundaries (bounding box)
    area_bounds = admin_gdf.total_bounds
    buildings_colombo = buildings_joined.cx[area_bounds[0]:area_bounds[2], area_bounds[1]:area_bounds[3]].copy()
    
    print(f"Sampling risk scores for {len(buildings_colombo)} buildings in Colombo...")
    buildings_risk = va.sample_raster_values(buildings_colombo, hazard_map, 'risk_score')
    buildings_risk = va.assign_risk_category(buildings_risk)
    
    high_risk_buildings = buildings_risk[buildings_risk['risk_category'] == 'High Risk']
    
    if not high_risk_buildings.empty:
        # Distance validation
        buildings_with_dist = va.sample_raster_values(high_risk_buildings, dist_risk_raw, 'river_dist')
        near_water = buildings_with_dist[buildings_with_dist['river_dist'] < 500] 
        consistency_score = (len(near_water) / len(high_risk_buildings)) * 100
        print(f"RESULT: {consistency_score:.1f}% of High-Risk buildings are within 500m of a waterway.")
    else:
        consistency_score = 0
        print("No High Risk buildings found.")
    
    # 6. Save Outputs
    output_map = OUTPUT_DIR / "scientific_risk_map.html"
    # Using clustered map for performance
    va.generate_interactive_risk_map_clustered(buildings_risk, str(output_map), include_low=False)
    print(f"Interactive map saved to: {output_map}")
    
    counts = buildings_risk['risk_category'].value_counts()
    print("Flood Risk Distribution (Buildings):")
    print(counts)
    
    report = {
        "spatial_consistency_score": float(consistency_score),
        "total_colombo_buildings": len(buildings_colombo),
        "high_risk_count": len(high_risk_buildings),
        "ahp_weights": {"rain": 0.35, "river_prox": 0.35, "elevation": 0.30}
    }
    with open(OUTPUT_DIR / "scientific_validation_report.json", 'w') as f:
        json.dump(report, f, indent=4)
    
    print("=== Scientific Analysis Complete ===")

if __name__ == "__main__":
    run_scientific_analysis()
