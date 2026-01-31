
import json

nb_path = "notebooks/analysis.ipynb"

with open(nb_path, 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    source_text = "".join(cell.get('source', []))
    
    # Update Section 0.4 Header
    if "0.4 Download" in source_text and "Google" in source_text:
        cell['source'] = ["### 0.4 Download OSM Buildings"]
        
    # Update Section 0.4 Code (Restore Overpass Download)
    elif "google_buildings.csv.gz" in source_text and "# 1. Go to:" in source_text:
        cell['source'] = [
            "# Download OSM buildings (this may take 5-10 minutes for urban areas)\n",
            "print(\"Downloading building footprints from OpenStreetMap...\")\n",
            "print(\"(This may take several minutes for dense urban areas)\")\n",
            "\n",
            "buildings_path = RAW_DIR / 'buildings' / 'osm_buildings.json'\n",
            "\n",
            "if not buildings_path.exists():\n",
            "    overpass_url = \"https://overpass-api.de/api/interpreter\"\n",
            "    \n",
            "    buildings_query = f\"\"\"\n",
            "    [out:json][timeout:600];\n",
            "    (\n",
            "      way[\"building\"]({COLOMBO_BBOX['south']},{COLOMBO_BBOX['west']},{COLOMBO_BBOX['north']},{COLOMBO_BBOX['east']});\n",
            "    );\n",
            "    out geom;\n",
            "    \"\"\"\n",
            "    \n",
            "    try:\n",
            "        print(\"  Querying Overpass API...\")\n",
            "        response = requests.post(overpass_url, data={'data': buildings_query}, timeout=900)\n",
            "        response.raise_for_status()\n",
            "        data = response.json()\n",
            "        \n",
            "        with open(buildings_path, 'w') as f:\n",
            "            json.dump(data, f)\n",
            "        \n",
            "        print(f\"  Saved: {buildings_path.name}\")\n",
            "        print(f\"  Found {len(data.get('elements', []))} building elements\")\n",
            "    except Exception as e:\n",
            "        print(f\"  Error: {e}\")\n",
            "        print(\"  Will use sample buildings instead\")\n",
            "else:\n",
            "    print(f\"  Already exists: {buildings_path.name}\")"
        ]

    # Update Data Loading Section (Load OSM + Clip)
    elif "load_google_buildings" in source_text:
        cell['source'] = [
            "# Load and Clip OSM Buildings\n",
            "print(\"Loading OSM Buildings...\")\n",
            "osm_buildings_file = RAW_DIR / 'buildings' / 'osm_buildings.json'\n",
            "\n",
            "if osm_buildings_file.exists():\n",
            "    buildings = data_loading.load_osm_buildings(osm_buildings_file)\n",
            "    \n",
            "    # Clip to District (Masking requirement)\n",
            "    # Check if district_mask is available (defined in earlier cell)\n",
            "    if 'district_mask' in locals():\n",
            "        try:\n",
            "            print(\"Clipping buildings to district boundary...\")\n",
            "            buildings = vector_analysis.clip_vectors_to_boundary(buildings, district_mask)\n",
            "        except Exception as e:\n",
            "            print(f\"Clipping failed: {e}\")\n",
            "    \n",
            "    print(f\"Loaded {len(buildings)} buildings\")\n",
            "else:\n",
            "    print(\"OSM Buildings file not found. Please run Section 0.4.\")"
        ]

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook reverted to OSM.")
