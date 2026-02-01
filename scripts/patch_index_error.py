
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
    if 'dist_deg = ra.calculate_euclidean_distance(elevation.shape,' in line:
        # Replace with 2D shape usage
        new_line = line.replace('elevation.shape', '(elevation.rio.height, elevation.rio.width)')
        new_source.append(new_line)
        
        # Add reshaping logic in the next line if needed
        # We'll look for where dist_meters is calculated and append the reshape there
        replaced = True
    elif 'dist_meters = dist_deg * 111000' in line:
        new_source.append(line)
        new_source.append('\n')
        new_source.append('            # Ensure shape matches elevation (handle band dimension)\n')
        new_source.append('            if len(elevation.shape) == 3:\n')
        new_source.append('                dist_meters = dist_meters.reshape(elevation.shape)\n')
    else:
        new_source.append(line)

if replaced:
    target_cell['source'] = new_source
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Successfully patched cell index 55: Fixed IndexError and added reshaping logic.")
else:
    print("Error: Could not find target line in cell source.")
