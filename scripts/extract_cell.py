
import json
import pathlib

nb_path = pathlib.Path('notebooks/analysis.ipynb')
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

indices = [45, 46]
for idx in indices:
    if idx < len(nb['cells']):
        cell = nb['cells'][idx]
        print(f"--- Cell Index {idx} (Exec {cell.get('execution_count', 'None')}) ---")
        print("".join(cell['source']))
