"""
Raster Analysis Module

NumPy and Xarray operations for rainfall and elevation data analysis.
Implements array-based masking, normalization, and temporal statistics.
"""

import numpy as np
import xarray as xr
from scipy.ndimage import uniform_filter, gaussian_filter
from typing import Union, Optional, Tuple
import geopandas as gpd
from shapely.geometry import mapping


def mask_raster_with_vector(
    raster: xr.DataArray,
    vector_gdf: gpd.GeoDataFrame,
    nodata: float = np.nan
) -> xr.DataArray:
    """
    Mask raster data using vector polygons.
    Requires 'rioxarray' extension to be loaded.
    
    Parameters
    ----------
    raster : xr.DataArray
        Input raster
    vector_gdf : gpd.GeoDataFrame
        Vector mask (polygons)
    nodata : float
        Value to fill outside mask
    
    Returns
    -------
    xr.DataArray
        Masked raster
    """
    # ensure crs match
    if not hasattr(raster, 'rio'):
        raise AttributeError("DataArray doesn't have rio accesssor. Did you import rioxarray?")
        
    if raster.rio.crs != vector_gdf.crs:
        vector_gdf = vector_gdf.to_crs(raster.rio.crs)
        
    # mask
    masked = raster.rio.clip(
        vector_gdf.geometry.apply(mapping),
        vector_gdf.crs,
        drop=True,
        invert=False,
        all_touched=True
    )
    
    return masked


def create_extreme_rainfall_mask(
    rainfall: Union[np.ndarray, xr.DataArray],
    threshold: float = 100.0
) -> Union[np.ndarray, xr.DataArray]:
    """
    Create a boolean mask for extreme rainfall events.
    
    Parameters
    ----------
    rainfall : np.ndarray or xr.DataArray
        Rainfall data (can be 2D or 3D with time dimension)
    threshold : float
        Rainfall threshold in mm (default: 100mm)
    
    Returns
    -------
    np.ndarray or xr.DataArray
        Boolean mask where True = extreme rainfall
    
    Example
    -------
    >>> extreme_mask = create_extreme_rainfall_mask(daily_rainfall, threshold=50)
    >>> extreme_days_count = extreme_mask.sum(axis=0)  # count per pixel
    """
    # simple element-wise comparison - no loops needed
    mask = rainfall > threshold
    return mask


def count_extreme_events(
    rainfall: Union[np.ndarray, xr.DataArray],
    threshold: float = 100.0,
    time_axis: int = 0
) -> Union[np.ndarray, xr.DataArray]:
    """
    Count the number of extreme rainfall days per pixel.
    
    Parameters
    ----------
    rainfall : np.ndarray or xr.DataArray
        3D rainfall data (time, lat, lon)
    threshold : float
        Rainfall threshold in mm
    time_axis : int
        Axis representing time dimension
    
    Returns
    -------
    np.ndarray or xr.DataArray
        2D array with count of extreme events per pixel
    """
    extreme_mask = rainfall > threshold
    
    if isinstance(extreme_mask, xr.DataArray):
        return extreme_mask.sum(dim='time')
    else:
        return np.sum(extreme_mask, axis=time_axis)


def calculate_percentile_rainfall(
    rainfall: Union[np.ndarray, xr.DataArray],
    percentile: float = 95.0,
    time_axis: int = 0
) -> Union[np.ndarray, xr.DataArray]:
    """
    Calculate percentile rainfall value for each pixel.
    
    Parameters
    ----------
    rainfall : np.ndarray or xr.DataArray
        3D rainfall data (time, lat, lon)
    percentile : float
        Percentile to calculate (e.g., 95 for 95th percentile)
    time_axis : int
        Axis representing time dimension
    
    Returns
    -------
    np.ndarray or xr.DataArray
        2D array with percentile values
    
    Example
    -------
    >>> p95 = calculate_percentile_rainfall(annual_data, percentile=95)
    >>> # p95 shows the 95th percentile rainfall for each location
    """
    if isinstance(rainfall, xr.DataArray):
        return rainfall.quantile(percentile / 100.0, dim='time')
    else:
        return np.percentile(rainfall, percentile, axis=time_axis)


def normalize_array(
    data: Union[np.ndarray, xr.DataArray],
    method: str = 'minmax'
) -> Union[np.ndarray, xr.DataArray]:
    """
    Normalize array values to 0-1 range.
    
    Parameters
    ----------
    data : np.ndarray or xr.DataArray
        Input data to normalize
    method : str
        Normalization method: 'minmax' or 'zscore'
    
    Returns
    -------
    np.ndarray or xr.DataArray
        Normalized data
    
    Example
    -------
    >>> rainfall_norm = normalize_array(max_rainfall, method='minmax')
    >>> # values now range from 0 to 1
    """
    if method == 'minmax':
        # min-max normalization to [0, 1]
        data_min = np.nanmin(data)
        data_max = np.nanmax(data)
        if data_max - data_min == 0:
            return np.zeros_like(data)
        return (data - data_min) / (data_max - data_min)
    
    elif method == 'zscore':
        # z-score normalization
        mean = np.nanmean(data)
        std = np.nanstd(data)
        if std == 0:
            return np.zeros_like(data)
        return (data - mean) / std
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'minmax' or 'zscore'")


def identify_low_elevation_areas(
    dem: Union[np.ndarray, xr.DataArray],
    percentile: float = 25.0
) -> Union[np.ndarray, xr.DataArray]:
    """
    Identify low-lying areas based on elevation percentile.
    
    Parameters
    ----------
    dem : np.ndarray or xr.DataArray
        Digital Elevation Model data
    percentile : float
        Percentile threshold (areas below this are 'low')
    
    Returns
    -------
    np.ndarray or xr.DataArray
        Boolean mask where True = low elevation area
    
    Example
    -------
    >>> low_areas = identify_low_elevation_areas(dem_data, percentile=25)
    >>> # low_areas shows pixels in the lowest 25% elevation
    """
    threshold_value = np.nanpercentile(dem, percentile)
    low_mask = dem < threshold_value
    return low_mask


def calculate_annual_maximum(
    rainfall: xr.DataArray,
    dim: str = 'time'
) -> xr.DataArray:
    """
    Calculate annual maximum daily rainfall.
    
    Parameters
    ----------
    rainfall : xr.DataArray
        Daily rainfall data with time dimension
    dim : str
        Name of time dimension
    
    Returns
    -------
    xr.DataArray
        Annual maximum rainfall per year and location
    
    Example
    -------
    >>> annual_max = calculate_annual_maximum(daily_rainfall)
    >>> # annual_max has dimensions (year, lat, lon)
    """
    return rainfall.groupby(f'{dim}.year').max(dim=dim)


def smooth_raster(
    data: Union[np.ndarray, xr.DataArray],
    method: str = 'gaussian',
    size: int = 3
) -> Union[np.ndarray, xr.DataArray]:
    """
    Apply spatial smoothing to raster data.
    
    Parameters
    ----------
    data : np.ndarray or xr.DataArray
        2D raster data
    method : str
        Smoothing method: 'gaussian' or 'uniform'
    size : int
        Kernel size (for uniform) or sigma (for gaussian)
    
    Returns
    -------
    np.ndarray or xr.DataArray
        Smoothed raster
    
    Example
    -------
    >>> smoothed = smooth_raster(rainfall_max, method='gaussian', size=2)
    """
    # extract numpy array if xarray
    is_xarray = isinstance(data, xr.DataArray)
    if is_xarray:
        values = data.values
        coords = data.coords
        dims = data.dims
    else:
        values = data
    
    # apply filter
    if method == 'gaussian':
        smoothed = gaussian_filter(values.astype(float), sigma=size)
    elif method == 'uniform':
        smoothed = uniform_filter(values.astype(float), size=size)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # convert back to xarray if needed
    if is_xarray:
        return xr.DataArray(smoothed, coords=coords, dims=dims)
    return smoothed


def calculate_vulnerability_index(
    rainfall_norm: np.ndarray,
    building_density_norm: np.ndarray,
    elevation_norm: np.ndarray,
    weights: Tuple[float, float, float] = (0.4, 0.3, 0.3)
) -> np.ndarray:
    """
    Calculate composite vulnerability index.
    
    V = w1 * rainfall + w2 * building_density + w3 * (1 - elevation)
    
    Parameters
    ----------
    rainfall_norm : np.ndarray
        Normalized rainfall intensity (0-1)
    building_density_norm : np.ndarray
        Normalized building density (0-1)
    elevation_norm : np.ndarray
        Normalized elevation (0-1), will be inverted
    weights : tuple
        Weights for each component (must sum to 1)
    
    Returns
    -------
    np.ndarray
        Vulnerability index (0-1, higher = more vulnerable)
    
    Example
    -------
    >>> vulnerability = calculate_vulnerability_index(
    ...     rainfall_norm, building_norm, elevation_norm,
    ...     weights=(0.4, 0.3, 0.3)
    ... )
    """
    w_rain, w_building, w_elev = weights
    
    # invert elevation (lower elevation = higher vulnerability)
    elevation_inverted = 1.0 - elevation_norm
    
    # weighted sum
    vulnerability = (
        w_rain * rainfall_norm +
        w_building * building_density_norm +
        w_elev * elevation_inverted
    )
    
    return vulnerability


if __name__ == "__main__":
    print("Raster analysis module loaded successfully")
    print("Available functions:")
    print("  - create_extreme_rainfall_mask()")
    print("  - count_extreme_events()")
    print("  - calculate_percentile_rainfall()")
    print("  - normalize_array()")
    print("  - identify_low_elevation_areas()")
    print("  - calculate_annual_maximum()")
    print("  - smooth_raster()")
    print("  - calculate_vulnerability_index()")
