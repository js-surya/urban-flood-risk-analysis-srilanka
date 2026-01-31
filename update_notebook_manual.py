
import json

nb_path = "notebooks/analysis.ipynb"

with open(nb_path, 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    source_text = "".join(cell.get('source', []))
    
    # Update Section 0.4 to be manual instructions
    if "Download Google Open Buildings" in source_text or "download_google_buildings" in source_text:
        cell['source'] = [
            "### 0.4 Download Google Open Buildings (Manual Step)\n",
            "\n",
            "# Due to network restrictions, please download the Google Buildings CSV manually.\n",
            "# 1. Go to: https://sites.research.google.com/open-buildings/\n",
            "# 2. Download the tile covering Colombo, Sri Lanka (S2 Level 4)\n",
            "# 3. Rename it to 'google_buildings.csv.gz'\n",
            "# 4. Place it in: ../data/raw/buildings/\n",
            "\n",
            "import shutil\n",
            "from pathlib import Path\n",
            "\n",
            "google_buildings_path = RAW_DIR / 'buildings' / 'google_buildings.csv.gz'\n",
            "\n",
            "if google_buildings_path.exists():\n",
            "    print(f\"Google Buildings data found: {google_buildings_path.name}\")\n",
            "else:\n",
            "    print(f\"Data not found at: {google_buildings_path}\")\n",
            "    print(\"Please download and place the file manually.\")"
        ]

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated for manual download.")
