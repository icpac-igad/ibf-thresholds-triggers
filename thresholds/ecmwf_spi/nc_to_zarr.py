#!/usr/bin/env python3
"""
Script to convert NetCDF SPI files to Zarr format and upload to a GCS bucket.

This script:
1. Combines multiple NetCDF files for each SPI type into a single xarray Dataset
2. Converts each Dataset to Zarr format with optimized chunking
3. Prepares for upload to a GCS bucket
"""

import os
import glob
import logging
import xarray as xr
import numpy as np
import pandas as pd
import dask
from dask.distributed import Client
import zarr
import gcsfs
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
INPUT_DIR = "ecmwf_spi"
OUTPUT_DIR = "spi_zarr"
SPI_TYPES = ["SPI1", "SPI3", "SPI6", "SPI12", "SPI24", "SPI36", "SPI48"]
GCS_BUCKET = "your-gcs-bucket-name"  # Change this to your GCS bucket name


def get_nc_files_for_spi(directory, spi_type):
    """
    Get a list of all NetCDF files for a specific SPI type.
    
    Args:
        directory: Directory containing NetCDF files
        spi_type: SPI type (e.g., "SPI1", "SPI3", etc.)
        
    Returns:
        List of file paths
    """
    pattern = f"{spi_type}_*.nc"
    file_paths = sorted(glob.glob(os.path.join(directory, pattern)))
    logger.info(f"Found {len(file_paths)} files for {spi_type}")
    return file_paths


def extract_date_from_filename(filename):
    """
    Extract date from filename and convert to pandas Timestamp.
    
    Args:
        filename: Filename (e.g., "SPI1_gamma_global_era5_moda_ref1991to2020_194001.area-subset.23.53.-12.21.nc")
        
    Returns:
        pandas.Timestamp
    """
    # Extract the date part (e.g., "194001")
    parts = os.path.basename(filename).split('_')
    date_str = None
    
    for part in parts:
        if len(part) == 6 and part.isdigit():
            date_str = part
            break
    
    if not date_str:
        raise ValueError(f"Could not extract date from filename: {filename}")
    
    # Convert to pandas Timestamp
    year = int(date_str[:4])
    month = int(date_str[4:6])
    
    return pd.Timestamp(year=year, month=month, day=1)


def combine_nc_files_to_dataset(file_paths, spi_type):
    """
    Combine multiple NetCDF files into a single xarray Dataset.
    
    Args:
        file_paths: List of NetCDF file paths
        spi_type: SPI type (e.g., "SPI1", "SPI3", etc.)
        
    Returns:
        xarray.Dataset
    """
    logger.info(f"Combining {len(file_paths)} files for {spi_type}")
    
    # Open the first file to get the template
    logger.info(f"Opening first file: {os.path.basename(file_paths[0])}")
    template_ds = xr.open_dataset(file_paths[0])
    
    # Create a list to store individual datasets
    datasets = []
    
    # Process files in batches to avoid using too much memory
    batch_size = 100
    num_batches = (len(file_paths) + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(file_paths))
        batch_files = file_paths[start_idx:end_idx]
        
        logger.info(f"Processing batch {batch_idx+1}/{num_batches} with {len(batch_files)} files")
        
        for file_path in batch_files:
            try:
                # Extract date from filename
                date = extract_date_from_filename(file_path)
                
                # Open the file and assign the date as the time coordinate
                ds = xr.open_dataset(file_path)
                
                # Ensure the dataset has the expected structure
                if spi_type not in ds.data_vars:
                    logger.warning(f"Variable {spi_type} not found in {os.path.basename(file_path)}")
                    continue
                
                # Replace the time coordinate with the extracted date
                ds = ds.assign_coords(time=[date])
                
                # Append to the list
                datasets.append(ds)
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
    
    if not datasets:
        raise ValueError(f"No valid datasets found for {spi_type}")
    
    # Combine all datasets along the time dimension
    combined_ds = xr.concat(datasets, dim="time")
    
    # Sort by time
    combined_ds = combined_ds.sortby("time")
    
    logger.info(f"Combined dataset for {spi_type}: {combined_ds.dims}")
    
    # Clean up
    template_ds.close()
    for ds in datasets:
        ds.close()
    
    return combined_ds


def save_to_zarr(dataset, spi_type, output_dir):
    """
    Save xarray Dataset to Zarr format with optimal chunking.
    
    Args:
        dataset: xarray.Dataset to save
        spi_type: SPI type (e.g., "SPI1", "SPI3", etc.)
        output_dir: Directory to save Zarr files
        
    Returns:
        Path to the saved Zarr store
    """
    logger.info(f"Saving {spi_type} to Zarr format")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the output path
    zarr_path = os.path.join(output_dir, f"{spi_type}.zarr")
    
    # Define optimal chunking
    chunks = {
        'time': min(12, len(dataset.time)),  # Chunk by year (12 months) if possible
        'lat': min(20, len(dataset.lat)),    # Chunk latitude into reasonable sized blocks
        'lon': min(20, len(dataset.lon))     # Chunk longitude into reasonable sized blocks
    }
    
    logger.info(f"Using chunks: {chunks}")
    
    # Make sure we have dask arrays
    dask_ds = dataset.chunk(chunks)
    
    # Set encoding for compression
    import numcodecs
    encoding = {var: {"compressor": numcodecs.Blosc(cname="zstd", clevel=3, shuffle=2)} 
                for var in dask_ds.data_vars}
    
    # Save to Zarr
    logger.info(f"Writing to {zarr_path}")
    start_time = time.time()
    dask_ds.to_zarr(zarr_path, mode="w", encoding=encoding)
    elapsed_time = time.time() - start_time
    
    logger.info(f"Saved {spi_type} to {zarr_path} in {elapsed_time:.2f} seconds")
    
    return zarr_path


def prepare_gcs_upload(zarr_path, spi_type, gcs_bucket):
    """
    Prepare for uploading Zarr store to GCS.
    This function demonstrates how to upload to GCS but doesn't actually upload
    without credentials.
    
    Args:
        zarr_path: Path to local Zarr store
        spi_type: SPI type (e.g., "SPI1", "SPI3", etc.)
        gcs_bucket: GCS bucket name
        
    Returns:
        Info about the upload
    """
    logger.info(f"Preparing to upload {spi_type} to GCS bucket: {gcs_bucket}")
    
    # Construct the GCS path
    gcs_path = f"gs://{gcs_bucket}/spi_zarr/{spi_type}.zarr"
    
    # Check if local Zarr store exists
    if not os.path.exists(zarr_path):
        logger.error(f"Local Zarr store not found: {zarr_path}")
        return None
    
    # Get info about the local Zarr store
    zarr_size_mb = sum(os.path.getsize(os.path.join(dirpath, filename)) 
                       for dirpath, _, filenames in os.walk(zarr_path) 
                       for filename in filenames) / (1024 * 1024)
    
    logger.info(f"Local Zarr store size: {zarr_size_mb:.2f} MB")
    
    # Info about the upload
    upload_info = {
        'local_path': zarr_path,
        'gcs_path': gcs_path,
        'size_mb': zarr_size_mb,
        'spi_type': spi_type
    }
    
    logger.info(f"To upload, you would use: gsutil -m cp -r {zarr_path} {gcs_path}")
    logger.info(f"Or with xarray: ds.to_zarr('{gcs_path}', storage_options={{'token': 'your-credentials'}})")
    
    return upload_info


def process_spi_type(spi_type):
    """
    Process a single SPI type from NetCDF to Zarr.
    
    Args:
        spi_type: SPI type (e.g., "SPI1", "SPI3", etc.)
        
    Returns:
        Info about the processed data
    """
    try:
        # Step 1: Get list of NetCDF files
        file_paths = get_nc_files_for_spi(INPUT_DIR, spi_type)
        
        if not file_paths:
            logger.warning(f"No files found for {spi_type}")
            return None
        
        # Step 2: Combine NetCDF files into a single dataset
        dataset = combine_nc_files_to_dataset(file_paths, spi_type)
        
        # Step 3: Save to Zarr format
        zarr_path = save_to_zarr(dataset, spi_type, OUTPUT_DIR)
        
        # Step 4: Prepare for GCS upload
        upload_info = prepare_gcs_upload(zarr_path, spi_type, GCS_BUCKET)
        
        return upload_info
    
    except Exception as e:
        logger.error(f"Error processing {spi_type}: {str(e)}")
        return None


def upload_to_gcs(local_path, gcs_path, credentials=None):
    """
    Upload Zarr store to GCS.
    This function would require proper GCS credentials to work.
    
    Args:
        local_path: Path to local Zarr store
        gcs_path: Path to GCS destination
        credentials: GCS credentials (optional)
        
    Returns:
        Boolean indicating success
    """
    logger.info(f"Uploading {local_path} to {gcs_path}")
    
    try:
        # This part would require GCS credentials
        # For demonstration, we're just showing the code structure
        
        # You would typically create a GCS filesystem with credentials
        # fs = gcsfs.GCSFileSystem(token=credentials)
        
        # Then you could upload with xarray
        # ds = xr.open_zarr(local_path)
        # ds.to_zarr(gcs_path, storage_options={'token': credentials})
        
        logger.info("To actually upload, you need to:")
        logger.info("1. Set up GCS credentials")
        logger.info("2. Use gcsfs to create a filesystem connection")
        logger.info("3. Use xarray's to_zarr with storage_options to upload")
        logger.info("Example: ds.to_zarr('gs://bucket/path.zarr', storage_options={'token': credentials})")
        
        return True
    
    except Exception as e:
        logger.error(f"Error uploading to GCS: {str(e)}")
        return False


def main():
    """Main function to convert all SPI NetCDF files to Zarr and prepare for GCS upload."""
    logger.info("Starting NetCDF to Zarr conversion for SPI files")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Set up dask client for parallel processing
    logger.info("Setting up dask client for parallel processing")
    client = Client()
    logger.info(f"Dask dashboard at {client.dashboard_link}")
    
    # Process each SPI type
    results = {}
    for spi_type in SPI_TYPES:
        logger.info(f"Processing {spi_type}")
        result = process_spi_type(spi_type)
        results[spi_type] = result
    
    # Print summary
    logger.info("\nSummary of Zarr conversion:")
    for spi_type, result in results.items():
        if result:
            logger.info(f"{spi_type}: {result['size_mb']:.2f} MB, saved to {result['local_path']}")
        else:
            logger.info(f"{spi_type}: Failed to process")
    
    logger.info("\nTo upload to GCS:")
    logger.info("1. Set up GCS credentials")
    logger.info("2. Run the upload_to_gcs function with proper credentials")
    logger.info("   or use gsutil: gsutil -m cp -r local_path gs://bucket/path")
    
    # Close dask client
    client.close()
    
    logger.info("Conversion completed")


if __name__ == "__main__":
    main()