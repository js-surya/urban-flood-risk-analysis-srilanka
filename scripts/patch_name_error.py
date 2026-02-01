
import json
import pathlib

nb_path = pathlib.Path('notebooks/analysis.ipynb')
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell Index 55 (Exec 39)
target_cell = nb['cells'][55]
source = target_cell['source']

new_source = []
replaced = False
for line in source:
    if 'str(elev_path)' in line:
        new_line = line.replace('str(elev_path)', 'str(elevation_path)')
        new_source.append(new_line)
        replaced = True
    else:
        new_source.append(line)

if replaced:
    target_cell['source'] = new_source
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Successfully patched cell index 55: elev_path -> elevation_path")
else:
    print("Error: Could not find 'str(elev_path)' in cell source.")
