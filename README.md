# Urban Flood Risk Analysis - Sri Lanka

Urban flood vulnerability assessment for Sri Lanka using CHIRPS rainfall data, Google Building footprints, and raster-vector integration.

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

# activate virtual environment
poetry shell

# or run commands directly
poetry run python -m src.data_loading
```

## Datasets

This project uses:
- **CHIRPS v2.0** - Rainfall data (0.05° resolution)
- **SRTM DEM** - Elevation data (30m resolution)
- **Google Open Buildings** - Building footprints
- **OpenStreetMap** - Roads and administrative boundaries

See `data/README.md` for download instructions.

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

## Authors

- **Surya Jamuna Rani Subramaniyan** (S3664414)
- **Sachin Ravi** (S3563545)

Scientific Programming for Geospatial Sciences - ITC, University of Twente (2025)
