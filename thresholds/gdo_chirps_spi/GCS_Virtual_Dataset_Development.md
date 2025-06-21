# GCS Virtual Dataset Development for xclim GEV Analysis

## Project Overview

This document describes the development process for creating a virtual dataset system to analyze CHIRPS SPI (Standardized Precipitation Index) data stored in Google Cloud Storage (GCS) using kerchunk, xarray, and eventually Coiled Dask for distributed extreme value analysis with xclim.

## Dataset Information

- **Source**: GCS bucket `cdi_arco` at path `gdo_chirps_spi_vz/spi1`
- **Data Format**: Kerchunk JSON reference files pointing to NetCDF data
- **Time Coverage**: 1991-2025 (35 years)
- **Spatial Coverage**: Global, 2400 × 7200 grid (lat × lon)
- **Variables**: `spc01` (1-month SPI values)
- **Total Size**: ~85GB (virtual references)

## Development Process

### 1. GCS Authentication and Access (✓ Completed)

**Script**: `test_gcs_access.py`

**Purpose**: Establish connection to GCS and list available JSON files.

**Key Features**:
- Uses service account authentication (`coiled-data-e4drr_202505.json`)
- Lists all JSON files in the specified bucket path
- Downloads sample files to inspect structure
- Creates inventory of available files

**Results**:
- Successfully connected to GCS
- Found 38 JSON files for spi1 (including 35 data files, concatenated, summary, and virtual_concat)
- Discovered additional spi12 files in the path

### 2. Download JSON Reference Files (✓ Completed)

**Script**: `download_all_spi1_json.py`

**Purpose**: Download all JSON reference files locally for virtual dataset creation.

**Key Features**:
- Parallel download using ThreadPoolExecutor
- Creates local `spi1/` directory structure
- Downloads only spi1 files (filters out spi12)
- Progress tracking and error handling

**Results**:
- Downloaded all 38 JSON files successfully
- Files stored in local `spi1/` directory
- Ready for virtual dataset creation

### 3. Virtual Dataset Creation (✓ Completed)

**Script**: `test_virtual_dataset.py`

**Purpose**: Create and test virtual dataset using xarray with kerchunk engine.

**Key Features**:
- Loads multiple JSON reference files
- Creates virtual concatenated dataset
- Tests subsetting for East Africa region
- Validates data access with small computations

**Performance**:
- Dataset loading: ~211 seconds
- Virtual subsetting: <0.01 seconds
- Small sample computation: ~3.5 seconds

**Issues Resolved**:
- Removed explicit chunking to avoid warnings
- Added missing file035 to concatenation list
- Handled NaN values in initial data samples

### 4. Polygon-based Subsetting (✓ Completed)

**Script**: `test_polygon_subset_fixed.py`

**Purpose**: Subset virtual dataset using polygons from GeoJSON file.

**Key Features**:
- Loads regions from `icpac_regions_admin1_20250630.geojson`
- Creates bounding box subsets for each region
- Implements polygon masking with regionmask
- Tests with first 3 regions (Djibouti, Gitega, Kirundo)

**Key Fixes**:
- Corrected regionmask API usage (removed incorrect parameters)
- Added proper error handling and traceback
- Implemented lazy masking to maintain virtual operations

## Next Steps: Coiled Dask Implementation Plan

### 5. Coiled Dask Setup and xclim GEV Analysis (Pending)

**Proposed Architecture**:

```python
# 1. Coiled cluster setup
import coiled
import dask.distributed

@coiled.cluster(
    name="spi-gev-analysis",
    n_workers=10,
    worker_memory="16 GB",
    region="us-east-1"
)
def create_cluster():
    return dask.distributed.Client()

# 2. GEV analysis function
@coiled.function(
    memory="32 GB",
    region="us-east-1"
)
def calculate_gev_for_region(region_polygon, spi_data, return_periods=[2, 5, 10, 20, 50, 100]):
    """Calculate GEV return periods for a single region"""
    import xclim.indices.stats
    
    # Subset data to region
    masked_data = subset_to_polygon(spi_data, region_polygon)
    
    # Calculate annual maxima/minima
    annual_extremes = masked_data.groupby('time.year').max()
    
    # Fit GEV distribution
    return_levels = xclim.indices.stats.frequency_analysis(
        annual_extremes,
        mode='max',
        t=return_periods,
        dist='gev'
    )
    
    return return_levels
```

### Implementation Steps:

1. **Environment Setup**
   - Install coiled: `pip install coiled`
   - Configure Coiled authentication
   - Set up environment with xclim, xarray, kerchunk

2. **Data Access Pattern**
   - Mount GCS credentials to Coiled workers
   - Use virtual dataset references directly from GCS
   - Implement efficient regional chunking strategy

3. **Parallel Processing**
   - One task per region (174 regions total)
   - Process multiple SPI timescales (spi1, spi3, spi6, etc.)
   - Calculate both drought and flood return periods

4. **Output Management**
   - Store results in GCS or S3
   - Generate CSV summaries per region
   - Create visualization outputs

## Code Usage Examples

### Loading Virtual Dataset
```python
import xarray as xr
import json

# Load file list
with open('spi1/spi1_virtual_concat.json', 'r') as f:
    virtual_info = json.load(f)
file_list = virtual_info['spi1_virtual_concat']['file_list']

# Create virtual dataset
ds = xr.open_mfdataset(
    file_list,
    engine='kerchunk',
    concat_dim='time',
    combine='nested',
    parallel=False
)
```

### Regional Subsetting
```python
import geopandas as gpd
import regionmask

# Load regions
regions = gpd.read_file('icpac_regions_admin1_20250630.geojson')

# Create subset for a region
region_bounds = regions.iloc[0].geometry.bounds
subset = ds.sel(
    lat=slice(bounds[3], bounds[1]),
    lon=slice(bounds[0], bounds[2])
)
```

## Performance Considerations

1. **Virtual Operations**: All subsetting and selection operations remain virtual (lazy) until `.compute()` is called
2. **Memory Efficiency**: No actual data loaded until computation
3. **Scalability**: Ready for distributed processing with Dask
4. **Optimization**: Consider rechunking after regional subsetting for optimal performance

## Dependencies

- xarray
- kerchunk
- geopandas
- regionmask
- google-cloud-storage
- numpy
- pandas
- coiled (for distributed processing)
- xclim (for GEV analysis)

## Conclusion

The virtual dataset system is successfully implemented and tested. The next phase involves setting up Coiled Dask for distributed GEV analysis across all 174 regions using xclim's statistical functions.