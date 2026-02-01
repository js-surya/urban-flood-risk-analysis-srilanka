
import json
import pathlib

nb_path = pathlib.Path('notebooks/analysis.ipynb')
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = cell['source']
        excerpt = "".join(source[:3]).replace('\n', ' ')
        exec_count = cell.get('execution_count', 'None')
        print(f"Index {i} (Exec {exec_count}): {excerpt[:80]}...")
