# Urban Flood Vulnerability Assessment - Colombo District, Sri Lanka

A scientific programming project for flood vulnerability mapping in **Colombo District**, 
combining CHIRPS rainfall data, SRTM elevation data, and Google Building footprints.

**Study Area:** Colombo District, Sri Lanka
- Bounding Box: 79.82°E - 80.22°E, 6.75°N - 7.05°N
- Area: ~676 km²

## Assignment 2 - Scientific Programming for Geospatial Sciences

This project implements flood exposure assessment combining raster and vector geospatial data to identify urban areas vulnerable to flooding in Sri Lankan cities.

## Project Structure

```
urban-flood-risk-analysis/
├── src/                    # Source code modules
│   ├── data_loading.py     # Data loading functions
│   ├── raster_analysis.py  # NumPy/Xarray raster operations
│   ├── tensor_operations.py # PyTorch tensor processing
│   ├── vector_analysis.py  # GeoPandas/Shapely operations
│   ├── integration.py      # Raster-vector integration

│   └── visualization.py    # Maps and charts
├── notebooks/              # Jupyter analysis notebook
│   └── analysis.ipynb      # Main analysis workflow
├── tests/                  # Unit tests
├── data/                   # Data directory (not tracked)
└── outputs/                # Generated outputs (not tracked)
```

## Requirements

- Python 3.10+
- Poetry for dependency management

## Installation

```bash
# install dependencies
poetry install

# register jupyter kernel
poetry run python -m ipykernel install --user --name urban-flood-analysis --display-name "Python (Urban Flood Analysis)"

# activate virtual environment
poetry shell

# or run commands directly
poetry run python -m src.data_loading
```

## Datasets

This project uses:
- **CHIRPS v2.0** - Rainfall data (0.05° resolution)
- **SRTM DEM** - Elevation data (30m resolution)
- **OpenStreetMap** - Building footprints, roads, and administrative boundaries

See `data/README.md` for download instructions.

> **Note:** Some large files are excluded from this repository to stay within GitHub's file size limits:
> *   `data/raw/buildings/osm_buildings.json` (~171 MB)
> *   `outputs/colombo_interactive_risk.html` (~241 MB)
> *   `outputs/colombo_interactive_risk_polygons.html` (~338 MB)
>
> These files will be generated locally when you run the notebook.

## Technical Components

1. **NumPy Arrays** - Raster masking, normalization, threshold analysis
2. **PyTorch Tensors** - Spatial convolution with GPU awareness
3. **GeoPandas/Shapely** - Spatial join, buffer, density calculation
4. **Xarray** - Multi-temporal data cube handling
5. **Raster-Vector Integration** - Zonal statistics, rasterization

## Usage

Run the main analysis notebook:
```bash
poetry run jupyter notebook notebooks/analysis.ipynb
```
### **Important: Selecting the Correct Kernel**
Once the notebook is open, you **must** select the correct environment to avoid `ModuleNotFoundError` errors:
1.  Click on the kernel name in the **top-right corner** of the interface (it may say "Python 3" or "Base").
2.  Select **"Python (Urban Flood Analysis)"** from the list.
3.  **Troubleshooting:** If you do not see the kernel in the list immediately, please **restart your IDE** (VS Code, Jupyter, etc.) or refresh your browser page to allow the kernel registry to update.

## Authors

- **Surya Jamuna Rani Subramaniyan** (S-3664414)
- **Sachin Ravi** (S-3563545)

Course: Scientific Programming for Geospatial Sciences - ITC, University of Twente (2026)
