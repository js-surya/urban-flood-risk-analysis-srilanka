# Data Directory - Colombo District

This directory contains the geospatial data files for flood vulnerability analysis in **Colombo District, Sri Lanka**.

## Study Area

**Bounding Box (EPSG:4326):**
- West: 79.82°E
- East: 80.22°E  
- South: 6.75°N
- North: 7.05°N

## Datasets

### 1. CHIRPS v2.0 (Rainfall)
- **Source:** https://data.chc.ucsb.edu/products/CHIRPS-2.0/
- **Resolution:** 0.05° (~5km), Daily
- **Format:** NetCDF (.nc)
- **Years:** 2018-2023 recommended
- **Download:** 
  ```
  # Example for 2023
  wget https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p05/chirps-v2.0.2023.days_p05.nc
  ```

### 2. SRTM DEM (Elevation)
- **Source:** https://dwtkns.com/srtm30m/ or https://earthexplorer.usgs.gov/
- **Tile:** N06E079, N06E080, N07E079, N07E080
- **Resolution:** 30m
- **Format:** GeoTIFF (.tif)

### 3. Google Open Buildings
- **Source:** https://sites.research.google/open-buildings/
- **Region:** South Asia → Sri Lanka
- **Format:** CSV with WKT geometry
- **Filter:** Clip to Colombo bounding box after download

### 4. Administrative Boundaries
- **Source:** https://download.geofabrik.de/asia/sri-lanka.html
- **Layer:** Administrative boundaries (district level)
- **Format:** Shapefile or GeoJSON

## Directory Structure

```
data/
├── raw/              # Original downloads (gitignored)
│   ├── chirps/
│   ├── dem/
│   ├── buildings/
│   └── admin/
├── processed/        # Clipped to study area (gitignored)
│   ├── chirps_colombo.nc
│   ├── dem_colombo.tif
│   ├── buildings_colombo.gpkg
│   └── admin_colombo.gpkg
└── sample/           # Small test files (tracked)
```

## Quick Start

1. Download the datasets from the sources above
2. Place raw files in `data/raw/`
3. Run the preprocessing notebook to clip to Colombo District
4. Processed files will be saved to `data/processed/`

## Note

Large data files are NOT tracked in git. Download them separately using the instructions above.
