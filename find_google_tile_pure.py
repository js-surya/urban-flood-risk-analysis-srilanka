
import json

# Colombo BBOX
BBOX = {
    'west': 79.82,
    'east': 80.22,
    'south': 6.75,
    'north': 7.05
}

def is_intersecting(poly_coords, bbox):
    # poly_coords is [[[x,y], [x,y], ...]]
    # Simple separating axis theorem or bounding box check for quick filter
    # Get poly bounds
    try:
        ring = poly_coords[0]
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        
        poly_w, poly_e = min(xs), max(xs)
        poly_s, poly_n = min(ys), max(ys)
        
        # Check if bounding boxes overlap
        if (poly_e < bbox['west'] or poly_w > bbox['east'] or 
            poly_n < bbox['south'] or poly_s > bbox['north']):
            return False
            
        print(f"Match found! Tile BBOX: {poly_w:.4f},{poly_s:.4f},{poly_e:.4f},{poly_n:.4f}")
        return True
    except Exception as e:
        return False

def find_tile():
    print("Downloading tiles.geojson...")
    # Using curl to download to file first, as done in previous step
    # but here we'll just read the local file if it exists and is valid, 
    # otherwise suggest downloading from correct URL
    url = "https://sites.research.google.com/open-buildings/tiles.geojson"
    try:
        # We need to download it first because the previous download was a 404 HTML
        import urllib.request
        urllib.request.urlretrieve(url, 'tiles.geojson')
    except Exception as e:
        print(f"Download failed: {e}")
        return

    print("Loading tiles.geojson...")
    try:
        with open('tiles.geojson', 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print("Error: tiles.geojson is not valid JSON. Ensure it downloaded correctly.")
        return
    
    print(f"Loaded {len(data['features'])} tiles.")
    
    matches = []
    for feature in data['features']:
        coords = feature['geometry']['coordinates']
        props = feature['properties']
        
        if is_intersecting(coords, BBOX):
            matches.append(props)
            
    print(f"\nFound {len(matches)} intersecting tiles:")
    for m in matches:
        print(f"Tile ID: {m.get('tile_id')}")
        print(f"URL: {m.get('tile_url')}")
        print(f"Size: {m.get('size_mb')} MB")
        print("-" * 30)

if __name__ == "__main__":
    find_tile()
