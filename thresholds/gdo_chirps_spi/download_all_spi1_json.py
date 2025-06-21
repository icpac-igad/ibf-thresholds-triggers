#!/usr/bin/env python3
"""
Download all JSON files from the spi1 path in GCS bucket
"""

import os
import json
from google.cloud import storage
from google.oauth2 import service_account
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
SERVICE_ACCOUNT_FILE = "coiled-data-e4drr_202505.json"
BUCKET_NAME = "cdi_arco"
PREFIX = "gdo_chirps_spi_vz/spi1"
LOCAL_DIR = "spi1"

def setup_gcs_client():
    """Set up authenticated GCS client"""
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE
    )
    client = storage.Client(credentials=credentials)
    return client

def download_blob(client, bucket_name, blob_name, local_path):
    """Download a single blob"""
    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Download
        blob.download_to_filename(local_path)
        print(f"Downloaded: {blob_name} -> {local_path}")
        return True, blob_name
    except Exception as e:
        print(f"Error downloading {blob_name}: {str(e)}")
        return False, blob_name

def main():
    """Download all JSON files from spi1 path"""
    print("Downloading all spi1 JSON files from GCS...")
    
    # Create local directory
    os.makedirs(LOCAL_DIR, exist_ok=True)
    
    # Set up client
    client = setup_gcs_client()
    bucket = client.bucket(BUCKET_NAME)
    
    # List all blobs
    blobs = bucket.list_blobs(prefix=PREFIX)
    
    # Filter only spi1 JSON files (not spi12)
    spi1_blobs = []
    for blob in blobs:
        if blob.name.endswith('.json') and '/spi1/' in blob.name:
            spi1_blobs.append(blob.name)
    
    print(f"Found {len(spi1_blobs)} JSON files to download")
    
    # Download in parallel
    successful = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit download tasks
        futures = {}
        for blob_name in spi1_blobs:
            # Extract just the filename for local path
            filename = os.path.basename(blob_name)
            local_path = os.path.join(LOCAL_DIR, filename)
            
            future = executor.submit(download_blob, client, BUCKET_NAME, blob_name, local_path)
            futures[future] = blob_name
        
        # Process completed downloads
        for future in as_completed(futures):
            success, blob_name = future.result()
            if success:
                successful += 1
            else:
                failed += 1
    
    print(f"\nDownload complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    # List downloaded files
    downloaded_files = os.listdir(LOCAL_DIR)
    print(f"\nDownloaded files in {LOCAL_DIR}/:")
    for f in sorted(downloaded_files):
        print(f"  {f}")

if __name__ == "__main__":
    main()