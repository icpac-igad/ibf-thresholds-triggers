# SPI NetCDF to Zarr Conversion

This repository contains scripts to convert Standardized Precipitation Index (SPI) NetCDF files to Zarr format for efficient cloud storage and access.

## Overview

These scripts handle the following tasks:

1. **Examine NetCDF Files**: Analyze the structure of NetCDF files to determine optimal chunking for Zarr conversion.
2. **Convert to Zarr**: Convert NetCDF files to Zarr format with optimized chunking.
3. **Upload to GCS**: Upload Zarr files to Google Cloud Storage for cloud access.

## Scripts

- `examine_nc_files.py`: Examines NetCDF files to understand their structure and suggest optimal chunking.
- `nc_to_zarr.py`: Combines NetCDF files by SPI type and converts them to Zarr format.
- `upload_zarr_to_gcs.py`: Uploads Zarr files to a Google Cloud Storage bucket.

## Prerequisites

- Python 3.6+
- Required Python packages:
  - xarray
  - dask
  - zarr
  - netCDF4
  - pandas
  - numpy
  - gcsfs (for GCS access)

## Directory Structure

```
.
├── ecmwf_spi/               # Directory containing NetCDF SPI files
├── spi_zarr/                # Output directory for Zarr files
├── examine_nc_files.py      # Script to examine NetCDF files
├── nc_to_zarr.py            # Script to convert NetCDF to Zarr
└── upload_zarr_to_gcs.py    # Script to upload Zarr to GCS
```

## Usage

### Step 1: Examine NetCDF Files

```bash
python examine_nc_files.py
```

This will analyze the structure of NetCDF files in the `ecmwf_spi/` directory and suggest an optimal chunking strategy for Zarr conversion.

### Step 2: Convert NetCDF Files to Zarr

```bash
python nc_to_zarr.py
```

This will:
1. Combine NetCDF files for each SPI type (SPI1, SPI3, SPI6, etc.)
2. Convert them to Zarr format with optimized chunking
3. Save them to the `spi_zarr/` directory

### Step 3: Upload Zarr Files to GCS

```bash
python upload_zarr_to_gcs.py --gcs-bucket YOUR_BUCKET_NAME [--service-account-file PATH_TO_KEY.json]
```

This will upload the Zarr files to the specified GCS bucket. If you don't provide a service account file, it will use the default credentials.

## Accessing Zarr Data in GCS

After uploading to GCS, you can access the data directly from Python:

```python
import xarray as xr
import gcsfs

# Create GCS filesystem
fs = gcsfs.GCSFileSystem()  # Or with credentials: fs = gcsfs.GCSFileSystem(token=credentials)

# Open Zarr store
ds = xr.open_zarr('gs://your-bucket/spi_zarr/SPI1.zarr')

# Use the data
print(ds)
```

## Chunking Strategy

The optimal chunking strategy for the SPI data is:

- **Time**: Chunk by year (12 months) or the entire dimension if smaller
- **Latitude**: Chunk into blocks of ~20 degrees
- **Longitude**: Chunk into blocks of ~20 degrees

This strategy balances:
- Efficient I/O operations
- Memory usage
- Parallel processing capabilities

## Advantages of Zarr Format

1. **Compression**: Reduces storage requirements
2. **Chunked Storage**: Allows for partial data access
3. **Cloud-Optimized**: Works well with cloud storage services
4. **Lazy Loading**: Enables working with datasets larger than memory
5. **Parallel Access**: Supports parallel I/O operations

## Notes

- For large NetCDF collections, the conversion process can be memory-intensive. Consider using a machine with sufficient RAM or processing in batches.
- GCS upload requires proper authentication. Make sure to set up your credentials before uploading.
- The chunking strategy may need adjustment based on your specific access patterns.