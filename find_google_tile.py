
import geopandas as gpd
import requests
from shapely.geometry import box

# Colombo BBOX
COLOMBO_BBOX = {
    'west': 79.82,
    'east': 80.22,
    'south': 6.75,
    'north': 7.05
}

def find_tile():
    print("Downloading tiles.geojson...")
    tiles_url = "https://sites.research.google/open-buildings/tiles.geojson"
    try:
        tiles_gdf = gpd.read_file(tiles_url)
    except Exception as e:
        print(f"Error reading tiles.geojson: {e}")
        return

    print(f"Loaded {len(tiles_gdf)} tiles.")
    
    # Create Colombo geometry
    colombo_geom = box(
        COLOMBO_BBOX['west'], 
        COLOMBO_BBOX['south'], 
        COLOMBO_BBOX['east'], 
        COLOMBO_BBOX['north']
    )
    
    # Find intersecting tiles
    intersecting = tiles_gdf[tiles_gdf.intersects(colombo_geom)]
    
    print(f"\nFound {len(intersecting)} intersecting tiles:")
    for idx, row in intersecting.iterrows():
        print(f"Tile ID: {row.get('tile_id')}")
        print(f"URL: {row.get('url')}")
        print(f"Size: {row.get('size_mb')} MB")
        print("-" * 30)

if __name__ == "__main__":
    find_tile()
