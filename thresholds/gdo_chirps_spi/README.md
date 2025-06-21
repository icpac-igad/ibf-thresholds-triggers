# SPI Virtual Concatenation for Extreme Value Analysis

## Overview

This document describes the implementation of virtual concatenation for Standardized Precipitation Index (SPI) datasets, enabling efficient extreme value analysis using Dask clusters without loading data into memory.

## Background

The SPI datasets are provided as individual NetCDF files, each containing 36 time steps. For extreme value analysis across multiple years, we need to concatenate these files virtually to create a unified dataset that can be processed efficiently on a Dask cluster.

### Dataset Structure
- **SPI Products**: spi1, spi3, spi6, spi9, spi12, spi24, spi48
- **Files per product**: 35 files (34 successfully loaded)
- **Time steps per file**: 36
- **Total time steps**: 1,224 (34 files × 36 steps)
- **Spatial grid**: 2400 × 7200 (lat × lon)
- **Data size per file**: ~2GB

## Virtual Concatenation Process

### 1. Initial Processing with `spi_processor.py`

The SPI processor creates individual Kerchunk JSON reference files for each NetCDF file:

```bash
python spi_processor.py --spi spi1 spi3 spi6 --full-processing
```

This creates a folder structure:
```
spi1/
├── spi1_file001.json
├── spi1_file002.json
├── ...
├── spi1_file035.json
└── spi1_summary.json
```

### 2. Virtual Concatenation Reference

Created a virtual concatenation reference file (`spi1_virtual_concat.json`) that contains:
- List of all JSON files to concatenate
- Total file count and time steps
- Usage instructions

```json
{
  "spi1_virtual_concat": {
    "description": "Virtual concatenation of SPI1 JSON files for extreme value analysis",
    "total_files": 35,
    "total_time_steps": 1260,
    "file_list": ["spi1/spi1_file001.json", ...],
    "usage": "Use xr.open_mfdataset(file_list, engine='kerchunk', concat_dim='time')"
  }
}
```

### 3. Loading Virtual Dataset

The virtual dataset is created using xarray's multi-file dataset capability:

```python
import xarray as xr
import json

# Load file list
with open('spi1/spi1_virtual_concat.json', 'r') as f:
    virtual_info = json.load(f)
file_list = virtual_info['spi1_virtual_concat']['file_list']

# Create virtual dataset (takes ~5-6 minutes but stays virtual)
ds = xr.open_mfdataset(
    file_list,
    engine='kerchunk',
    concat_dim='time',
    combine='nested',
    chunks={'time': 72, 'lat': 400, 'lon': 400},
    parallel=False
)
```

## Performance Characteristics

### Loading Times
- **Individual file**: 5-10 seconds
- **5 files**: ~52 seconds
- **35 files**: ~5-6 minutes (estimated)

### Virtual Operations (Instant)
- Subsetting by region: 0.002 seconds
- Creating analysis operations: < 1 second
- Rechunking: < 1 second

### Actual Computation
- Sample 10×10×5 pixels: 5.3 seconds
- Full computation: Depends on Dask cluster resources

## East Africa Regional Analysis Example

Successfully tested virtual subsetting to East Africa region:

```python
# Define East Africa bounds
east_africa_bounds = {
    'lat_min': -15.0,  # Southern boundary
    'lat_max': 20.0,   # Northern boundary  
    'lon_min': 25.0,   # Western boundary
    'lon_max': 55.0    # Eastern boundary
}

# Create virtual subset (instant!)
east_africa_ds = ds.sel(
    lat=slice(20.0, -15.0),  # Note: reversed for lat
    lon=slice(25.0, 55.0)
)

# Result: 700×600 grid covering East Africa
# All operations remain virtual until .compute() is called
```

## Dask Cluster Integration

The virtual dataset is perfectly suited for Dask cluster processing:

```python
from dask.distributed import Client

# Connect to cluster
client = Client('scheduler-address:8786')

# All operations are lazy/virtual
spi_data = ds['spc01']

# Extreme value analysis
temporal_max = spi_data.max(dim='time')      # Virtual
annual_max = spi_data.groupby('time.year').max()  # Virtual
percentiles = spi_data.quantile([0.01, 0.99], dim='time')  # Virtual

# Execute on cluster
results = temporal_max.compute()  # Distributed computation
```

## Key Benefits

1. **Memory Efficient**: No data loaded until explicitly computed
2. **Instant Subsetting**: Regional subsets created in milliseconds
3. **Dask Ready**: Preserves chunking for distributed computing
4. **Scalable**: Works with any number of files
5. **Flexible**: Easy to modify chunks for different analyses

## Files Created

1. **`create_spi1_virtual.py`**: Creates virtual concatenation reference
2. **`fast_spi_concat.py`**: Full concatenation implementation
3. **`test_spi1_virtual_east_africa.py`**: Tests virtual dataset and regional subsetting
4. **`spi1/spi1_virtual_concat.json`**: Virtual concatenation reference file
5. **`spi1/spi1_usage_example.py`**: Complete usage example with extreme value analysis

## Next Steps

1. Apply same process to other SPI products (spi3, spi6, etc.)
2. Upload JSON reference files to Google Cloud Storage
3. Set up Dask cluster for distributed extreme value analysis
4. Implement specific extreme value analysis workflows