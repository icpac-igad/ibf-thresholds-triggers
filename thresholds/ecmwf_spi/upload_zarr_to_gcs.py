#!/usr/bin/env python3
"""
Script to upload Zarr files to Google Cloud Storage (GCS).

This script takes local Zarr files and uploads them to a GCS bucket with proper
authentication. It requires GCS credentials to be set up before running.
"""

import os
import logging
import time
import argparse
import xarray as xr
import gcsfs
from google.oauth2 import service_account

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Upload Zarr files to GCS')
    parser.add_argument('--input-dir', type=str, default='spi_zarr',
                      help='Directory containing Zarr files')
    parser.add_argument('--gcs-bucket', type=str, required=True,
                      help='GCS bucket name')
    parser.add_argument('--gcs-prefix', type=str, default='spi_zarr',
                      help='Prefix path within the GCS bucket')
    parser.add_argument('--service-account-file', type=str,
                      help='Path to GCS service account key file (JSON)')
    return parser.parse_args()

def get_zarr_stores(input_dir):
    """
    Get a list of all Zarr stores in the input directory.
    
    Args:
        input_dir: Directory containing Zarr stores
        
    Returns:
        List of Zarr store names
    """
    if not os.path.isdir(input_dir):
        logger.error(f"Input directory not found: {input_dir}")
        return []
    
    # Look for directories with a .zgroup file, which indicates a Zarr store
    zarr_stores = []
    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)
        if os.path.isdir(item_path) and item.endswith('.zarr') and os.path.exists(os.path.join(item_path, '.zgroup')):
            zarr_stores.append(item)
    
    return zarr_stores

def get_gcs_credentials(service_account_file=None):
    """
    Get GCS credentials.
    
    Args:
        service_account_file: Path to service account key file
        
    Returns:
        GCS credentials object
    """
    if service_account_file and os.path.exists(service_account_file):
        logger.info(f"Using service account key from {service_account_file}")
        credentials = service_account.Credentials.from_service_account_file(
            service_account_file,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        return credentials
    
    # If no service account file is provided, try to use the default credentials
    logger.info("No service account file provided, using default credentials")
    logger.info("Make sure you've set up authentication with 'gcloud auth application-default login'")
    return None

def upload_zarr_to_gcs(local_path, gcs_path, credentials=None):
    """
    Upload a Zarr store to GCS.
    
    Args:
        local_path: Path to local Zarr store
        gcs_path: Path to GCS destination
        credentials: GCS credentials (optional)
        
    Returns:
        Boolean indicating success
    """
    logger.info(f"Uploading {local_path} to {gcs_path}")
    
    try:
        # Make sure the local Zarr store exists
        if not os.path.exists(local_path):
            logger.error(f"Local Zarr store not found: {local_path}")
            return False
        
        # Get size of local Zarr store
        zarr_size_mb = sum(os.path.getsize(os.path.join(dirpath, filename)) 
                           for dirpath, _, filenames in os.walk(local_path) 
                           for filename in filenames) / (1024 * 1024)
        
        logger.info(f"Local Zarr store size: {zarr_size_mb:.2f} MB")
        
        # Create GCS filesystem
        storage_options = {'token': credentials} if credentials else {}
        fs = gcsfs.GCSFileSystem(**storage_options)
        
        # Open local Zarr store
        start_time = time.time()
        ds = xr.open_zarr(local_path)
        
        # Upload to GCS
        logger.info(f"Writing to {gcs_path}")
        ds.to_zarr(gcs_path, storage_options=storage_options)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Upload completed in {elapsed_time:.2f} seconds")
        
        return True
    
    except Exception as e:
        logger.error(f"Error uploading to GCS: {str(e)}")
        return False

def main():
    """
    Main function to upload Zarr files to GCS.
    """
    args = parse_args()
    
    # Get GCS credentials
    credentials = get_gcs_credentials(args.service_account_file)
    
    # Get list of Zarr stores
    zarr_stores = get_zarr_stores(args.input_dir)
    logger.info(f"Found {len(zarr_stores)} Zarr stores in {args.input_dir}: {zarr_stores}")
    
    if not zarr_stores:
        logger.error("No Zarr stores found to upload")
        return
    
    # Upload each Zarr store to GCS
    success_count = 0
    for zarr_store in zarr_stores:
        local_path = os.path.join(args.input_dir, zarr_store)
        gcs_path = f"gs://{args.gcs_bucket}/{args.gcs_prefix}/{zarr_store}"
        
        success = upload_zarr_to_gcs(local_path, gcs_path, credentials)
        if success:
            logger.info(f"Successfully uploaded {zarr_store} to GCS")
            success_count += 1
        else:
            logger.error(f"Failed to upload {zarr_store} to GCS")
    
    # Print summary
    logger.info(f"Uploaded {success_count} of {len(zarr_stores)} Zarr stores to GCS")
    
    # Provide information about accessing the Zarr files in GCS
    if success_count > 0:
        logger.info("\nTo access the Zarr files in GCS from Python:")
        logger.info(f"import xarray as xr")
        logger.info(f"import gcsfs")
        logger.info(f"fs = gcsfs.GCSFileSystem()")
        logger.info(f"ds = xr.open_zarr(f'gs://{args.gcs_bucket}/{args.gcs_prefix}/{zarr_stores[0]}', storage_options={{'token': credentials}})")
        logger.info(f"# or without credentials if using default authentication:")
        logger.info(f"ds = xr.open_zarr(f'gs://{args.gcs_bucket}/{args.gcs_prefix}/{zarr_stores[0]}')")

if __name__ == "__main__":
    main()