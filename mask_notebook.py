
import json

nb_path = "notebooks/analysis.ipynb"

with open(nb_path, 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    source_text = "".join(cell.get('source', []))
    
    # 1. Update Admin Loading to Dissolve
    if "load_admin_boundaries" in source_text and "RAW_DIR" in source_text:
        cell['source'] = [
            "# Load Admin Boundaries\n",
            "print(\"Loading admin boundaries...\")\n",
            "admin_file = RAW_DIR / 'admin' / 'colombo_boundary.json'\n",
            "admin_boundaries = data_loading.load_admin_boundaries(admin_file, level='7')\n",
            "\n",
            "# Dissolve to get district boundary for masking\n",
            "district_mask = admin_boundaries.dissolve()\n",
            "print(f\"Admin boundaries loaded: {len(admin_boundaries)} units\")"
        ]
        
    # 2. Update Elevation Loading to Mask
    elif "load_srtm_tiles" in source_text:
        cell['source'] = [
            "# Mask Elevation with District Boundary\n",
            "elevation = data_loading.load_srtm_tiles(RAW_DIR / 'dem', bbox=tuple(COLOMBO_BBOX.values()))\n",
            "\n",
            "# Apply Mask\n",
            "elevation = raster_analysis.mask_raster_with_vector(elevation, district_mask)\n",
            "\n",
            "print(f\"Elevation data loaded and masked: {elevation.shape}\")\n",
            "elevation.plot(cmap='terrain', figsize=(10, 8), cbar_kwargs={'label': 'Elevation (m)'})"
        ]

    # 3. Update CHIRPS Loading to Mask
    elif "Load CHIRPS 2025" in source_text or ("rainfall_cube =" in source_text and "ds['precip'].sel" in source_text):
        # We need to find the specific cell again, reusing logic from previous replace if possible
        # but here we overwrite the cell content completely with the masked version
        cell['source'] = [
             "# Load CHIRPS 2025 rainfall data and clip to study area\n",
            "print(\"Loading CHIRPS 2025 monthly rainfall data...\")\n",
            "\n",
            "chirps_file = RAW_DIR / 'chirps' / 'chirps-v2.0.2025.monthly.nc'\n",
            "\n",
            "if chirps_file.exists():\n",
            "    # Load full dataset\n",
            "    ds = xr.open_dataset(chirps_file)\n",
            "    \n",
            "    # Get the precipitation variable and clip to BBOX first\n",
            "    # Slice latitude (handle both ascending and descending cases)\n",
            "    rainfall_cube = ds['precip'].sel(\n",
            "        latitude=slice(COLOMBO_BBOX['south'], COLOMBO_BBOX['north']),\n",
            "        longitude=slice(COLOMBO_BBOX['west'], COLOMBO_BBOX['east'])\n",
            "    )\n",
            "    if rainfall_cube.size == 0:\n",
            "       rainfall_cube = ds['precip'].sel(\n",
            "          latitude=slice(COLOMBO_BBOX['north'], COLOMBO_BBOX['south']),\n",
            "          longitude=slice(COLOMBO_BBOX['west'], COLOMBO_BBOX['east'])\n",
            "       )\n",
            "\n",
            "    # Apply District Mask\n",
            "    try:\n",
            "        rainfall_cube = raster_analysis.mask_raster_with_vector(rainfall_cube, district_mask)\n",
            "        print(f\"Rainfall data loaded and masked: {rainfall_cube.shape}\")\n",
            "    except Exception as e:\n",
            "        print(f\"Warning: Masking failed ({e}), using rectangular clip\")\n",
            "    \n",
            "else:\n",
            "    print(\"CHIRPS file not found\")"
        ]

    # 4. Update Buildings Loading to Clip (Google Buildings)
    elif "load_google_buildings" in source_text and "osm_buildings = " in source_text:
        cell['source'] = [
            "# Load and Clip Google Buildings\n",
            "print(\"Loading Google Open Buildings...\")\n",
            "google_buildings_file = RAW_DIR / 'buildings' / 'google_buildings.csv.gz'\n",
            "\n",
            "if google_buildings_file.exists():\n",
            "    buildings = data_loading.load_google_buildings(google_buildings_file, bbox=COLOMBO_BBOX)\n",
            "    \n",
            "    # Clip to District\n",
            "    buildings = vector_analysis.clip_vectors_to_boundary(buildings, district_mask)\n",
            "    \n",
            "    print(f\"Loaded and masked {len(buildings)} buildings\")\n",
            "    \n",
            "    # Compatibility alias\n",
            "    osm_buildings = buildings\n",
            "else:\n",
            "    print(\"Google Buildings file not found. Please run Section 0.4.\")"
        ]
        
    # 5. Fix Plotting (imshow needs extent fix or using xarray plot)
    elif "plt.imshow(max_monthly_rainfall" in source_text:
        cell['source'] = [
            "max_monthly_rainfall.plot(cmap='Blues', figsize=(10, 8), cbar_kwargs={'label': 'Max Monthly Rainfall (mm)'})\n",
            "plt.title('Max Monthly Rainfall (Masked)')"
        ]

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook masked.")
