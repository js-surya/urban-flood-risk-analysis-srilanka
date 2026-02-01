
import json
import pathlib

def fix_notebook():
    nb_path = pathlib.Path('notebooks/analysis.ipynb')
    if not nb_path.exists():
        print(f"Notebook not found at {nb_path}")
        return

    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Strategies to identify the cell
    marker1 = "admin_boundaries.plot"
    marker2 = "buildings.plot"
    
    # The new robust code
    new_source = [
        "# Plot (Fixed)\n",
        "try:\n",
        "    if 'admin_boundaries' in locals() and not admin_boundaries.empty:\n",
        "        fig, ax = plt.subplots(figsize=(10, 10))\n",
        "        \n",
        "        # Check if admin boundaries have valid geometry\n",
        "        # This prevents 'aspect must be finite' error in geopandas/matplotlib\n",
        "        valid_admin = admin_boundaries[~admin_boundaries.geometry.is_empty & admin_boundaries.geometry.is_valid]\n",
        "        if not valid_admin.empty:\n",
        "            try:\n",
        "                valid_admin.plot(ax=ax, color='white', edgecolor='black')\n",
        "            except Exception as e:\n",
        "                print(f\"Warning: Could not plot admin boundaries: {e}\")\n",
        "        else:\n",
        "             print(\"Warning: admin_boundaries has no valid/non-empty geometries to plot.\")\n",
        "        \n",
        "        if 'osm_buildings_file' in locals() and osm_buildings_file.exists() and 'buildings' in locals() and not buildings.empty:\n",
        "            valid_buildings = buildings[~buildings.geometry.is_empty & buildings.geometry.is_valid]\n",
        "            if not valid_buildings.empty:\n",
        "                try:\n",
        "                    valid_buildings.plot(ax=ax, markersize=1, color='red', alpha=0.5)\n",
        "                except Exception as e:\n",
        "                    print(f\"Warning: Could not plot buildings: {e}\")\n",
        "            else:\n",
        "                print(\"Warning: Buildings dataframe has no valid/non-empty geometries.\")\n",
        "            \n",
        "        ax.set_title(\"Buildings in Colombo (Clipped)\")\n",
        "        ax.set_axis_off()\n",
        "        plt.show()\n",
        "    else:\n",
        "        print(\"Skipping plot: admin_boundaries not available or empty.\")\n",
        "except Exception as e:\n",
        "    print(f\"Plotting completely failed: {e}\")\n"
    ]

    fixed_count = 0
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source_text = "".join(cell['source'])
            if marker1 in source_text and marker2 in source_text:
                print(f"Found target plotting cell at index {i}. Content preview:\n{source_text[:100]}...")
                cell['source'] = new_source
                fixed_count += 1
    
    if fixed_count > 0:
        with open(nb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"Successfully patched {fixed_count} cell(s) in {nb_path}")
    else:
        print("Target plotting code not found (markers: 'admin_boundaries.plot' and 'buildings.plot').")
        # debug: print code cells that contain 'plot'
        print("Debugging info: Cells containing 'plot':")
        for i, cell in enumerate(nb['cells']):
             if cell['cell_type'] == 'code' and 'plot' in "".join(cell['source']).lower():
                 print(f"Cell {i} preview: { ''.join(cell['source'])[:80]}...")

if __name__ == "__main__":
    fix_notebook()
