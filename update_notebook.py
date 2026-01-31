
import json
import re

nb_path = "notebooks/analysis.ipynb"

with open(nb_path, 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    source_text = "".join(cell.get('source', []))
    
    # Update Header
    if "### 0.4 Download OSM Buildings" in source_text:
        cell['source'] = ["### 0.4 Download Google Open Buildings"]
        
    # Update Download Code
    elif "osm_buildings.json" in source_text and "overpass_url" in source_text:
        cell['source'] = [
            "# Download Google Open Buildings v3\n",
            "print(\"Downloading Google Open Buildings v3...\")\n",
            "\n",
            "google_buildings_path = data_loading.download_google_buildings(\n",
            "    bbox=COLOMBO_BBOX,\n",
            "    output_dir=RAW_DIR / 'buildings'\n",
            ")\n",
            "\n",
            "if google_buildings_path and google_buildings_path.exists():\n",
            "    print(\"Google Buildings ready.\")\n",
            "else:\n",
            "    print(\"Failed to prepare Google Buildings data.\")"
        ]
        
    # Update Loading Code (generic match for osm_buildings.json usage)
    elif "osm_buildings.json" in source_text:
        # We assume this is the loading cell if it's not the download cell
        # We will replace it directly
        cell['source'] = [
            "# Load Google Buildings\n",
            "print(\"Loading Google Open Buildings...\")\n",
            "google_buildings_file = RAW_DIR / 'buildings' / 'google_buildings.csv.gz'\n",
            "\n",
            "if google_buildings_file.exists():\n",
            "    buildings = data_loading.load_google_buildings(google_buildings_file, bbox=COLOMBO_BBOX)\n",
            "    print(f\"Loaded {len(buildings)} buildings\")\n",
            "    \n",
            "    # Compatibility alias for existing code\n",
            "    osm_buildings = buildings\n",
            "else:\n",
            "    print(\"Google Buildings file not found. Please run Section 0.4.\")"
        ]

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated.")
