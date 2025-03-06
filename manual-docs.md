# IBF Threshold and Triggers

This document outlines the **IBF Drought Threshold and Trigger Settings**, detailing the transition from **local NetCDF files** to **remote Zarr access** and the **configuration for multiple regions, including Karamoja (KMJ)**.

## 1. System Overview

The system **analyzes climate data** (precipitation) to forecast drought conditions across different regions in East Africa. It calculates the **Standardized Precipitation Index (SPI)** and generates probability forecasts for drought events at different severity levels (moderate, severe, extreme).

### Key Files in the Implementation

The implementation consists of the following key files:

1. **run_main.py**: Main script that orchestrates the analysis workflow.
2. **vthree_utils.py**: Core utilities for data processing and analysis.
3. **.env**: Environment configuration file for credentials and paths.
4. **spi_utils.py**: Utilities for calculating the Standardized Precipitation Index.
5. **requirements.txt**: Dependencies for the system.
6. **xbootstrap.py**: Implementation of bootstrap analysis for uncertainty quantification.
7. **zarr_utils.py**: Utilities for accessing remote Zarr data stores.
8. **gcs_auth_helper.py**: Helpers for Google Cloud Storage authentication.
9. **vthree_utils_auth_patch.py**: Authentication patch for existing utilities.

---
## 1. Dependencies Management

The `requirements.txt` includes:

```bash
python-dotenv
numpy
pandas
xarray
scipy
matplotlib
dask
geopandas
shapely
pyarrow
h5netcdf
fsspec
gcsfs
google-auth
zarr
scikit-learn
```

Installation script (`install_dependencies.sh`):

```bash
#!/bin/bash
pip install -r requirements.txt
mkdir -p output
```

## 2. Execution Flow

The system executes as follows:

1. **Environment Setup**:
   - Load environment variables for authentication and data paths.
   - Set up logging.

2. **Data Loading**:
   - Authenticate with Google Cloud using credentials.
   - Load boundary data from **Parquet files**.
   - Connect to **remote Zarr stores**.

3. **Data Processing**:
   - For **Karamoja (KMJ)**:
     - Subset data based on region boundaries.
     - Compute **SPI values**.
     - Calculate **drought probability thresholds**.

4. **Metrics Calculation**:
   - Run `run_xhist2d()` for 2D contingency analysis.
   - Run `run_xhist1d()` for 1D probability metrics.

5. **Result Output**:
   - Save CSV files with drought forecasts for **Karamoja (KMJ)**.

---

## 3. Major Implementation Changes

### 3.1 Transition from Local NetCDF to Remote Zarr

#### Initial Problem
The system initially relied on **large NetCDF (.nc) files** stored locally, leading to several issues:
- **Storage limitations** on deployment environments.
- **Performance bottlenecks** during data loading.
- **Challenges in maintaining data consistency** across installations.

#### Solution Implementation
The **local NetCDF files** were replaced with **remote Zarr stores** hosted on **Google Cloud Storage (GCS)**:

1. **Implemented remote data access via `zarr_utils.py`**:

```python
def read_remote_zarr(zarr_url, storage_options=None):
    """
    Reads a remote Zarr store as an xarray Dataset.
    """
    try:
        logger.info(f"Reading Zarr store from {zarr_url}")
        if isinstance(storage_options, str) and os.path.exists(storage_options):
            logger.info(f"Using service account file: {storage_options}")
            storage_options = get_storage_options(storage_options)

        logger.info("Opening Zarr store...")
        ds = xr.open_zarr(zarr_url, storage_options=storage_options, consolidated=True)
        logger.info(f"Successfully loaded Zarr store with dims: {dict(ds.dims)}")

        return ds
    except Exception as e:
        logger.error(f"Error reading Zarr store: {str(e)}")
        raise
```

2. **Updated data loading in `vthree_utils.py` to leverage remote Zarr access**:

```python
# Previous approach:
# ds = xr.open_dataset(nc_path)

# New approach:
from zarr_utils import read_remote_zarr
ds = read_remote_zarr(zarr_url, storage_options)
```

---

### 3.2 Authentication Implementation

To securely access **remote Zarr stores**, authentication is handled using **Google Cloud credentials**:

1. **Environment variables in `.env` for authentication and paths**:

```bash
SERVICE_ACCOUNT_JSON=coiled-data_20241128.json
POLYGON_PQ_URI=gs://seas51/ea_admin0_2_custom_polygon_shapefile_v5.parquet
CHIRPS_ZARR_PATH=gs://seas51/chirps_v2_monthly_20241012.zarr
SEAS51_ZARR_PATH=gs://seas51/seas51_20241012_v3.zarr
OUTPUT_PATH=./output/
```

2. **Implemented `gcs_auth_helper.py` to manage authentication**:

```python
def get_credentials_from_json(service_account_path):
    """Creates GCP credentials from a service account JSON file."""
    try:
        logger.info(f"Loading credentials from {service_account_path}")
        credentials = service_account.Credentials.from_service_account_file(
            service_account_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        return credentials
    except Exception as e:
        logger.error(f"Error loading credentials: {e}")
        raise

def get_storage_options(service_account_path):
    """Retrieves storage options for GCS-based authentication."""
    credentials = get_credentials_from_json(service_account_path)
    project_id = get_project_id_from_json(service_account_path)

    return {
        'token': credentials,
        'project': project_id
    }
```

---

### 3.3 Region-Specific Configurations

The system supports **multiple regions**, including **Karamoja (KMJ)**.

1. **Modified `run_main.py` to include region configurations**:

```python
regions = [
    {
        "id": 1155, 
        "name": "Karamoja", 
        "filter": "kmj",
        "output_path": "./outputkmj/"
    }
]
```

2. **Added region-specific threshold values in `vthree_utils.py`**:

```python
def get_threshold(region_id, season):
    """
    Retrieves drought threshold values for a specified region and season.
    """
    data = """region_id,region,season,mod,sev,ext
    1155,kmj,mam,-0.43,-0.67,-0.84
    1155,kmj,ond,-0.55,-0.98,-0.99
    1155,kmj,jja,-0.43,-0.67,-0.84
    1155,kmj,jjas,-0.40,-0.98,-0.99
    """
```

3. **Implemented flexible region boundary loading using Parquet files**:

```python
def gcs_parquet_mask_creator(params):
    """
    Generates region/district masks using a Parquet file.
    """
    try:
        logger.info(f"Reading Parquet file from {params.gcs_file_url}")
        storage_options = get_storage_options(params.service_account_json)
        logger.info("Successfully created credentials")

        ddf = daskdf.read_parquet(params.gcs_file_url,
                                  storage_options=storage_options,
                                  engine='pyarrow')

        fdf = ddf[(ddf['level'] == params.level) & (ddf['gbid'] == params.region_filter)]
    except Exception as e:
        logger.error(f"Error in gcs_parquet_mask_creator: {str(e)}")
```

---

### 3.4 Enhanced SPI Calculation and Analysis

1. **Optimized SPI calculation in `spi_utils.py`**:

```python
def standardized_precipitation_index(data, freq="MS", window=3, dist="gamma", method="APP"):
    """
    Computes the Standardized Precipitation Index (SPI).
    """
    try:
        precip = data.copy()
        rolled = precip.rolling(time=window, center=False).sum()
    except Exception as e:
        logger.error(f"Error calculating SPI: {str(e)}")
        raise
```

---
---

## 4. Benefits of the Implementation

1. **Improved Performance** (remote Zarr reduces local storage).
2. **Scalability** (supports large datasets via cloud storage).
3. **Enhanced Collaboration** (centralized data for consistency).
4. **Simplified Deployment** (standardized authentication).
5. **Multi-Region Support** (**Karamoja (KMJ)** configurations integrated).

This implementation significantly enhances **drought prediction capabilities**, ensuring **efficient, scalable, and accurate forecasting** for Karamoja. 🚀
