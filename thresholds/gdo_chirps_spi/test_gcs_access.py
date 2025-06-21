#!/usr/bin/env python3
"""
Test script to access GCS bucket and list JSON files for SPI data
"""

import os
import json
from google.cloud import storage
from google.oauth2 import service_account

# Configuration
SERVICE_ACCOUNT_FILE = "coiled-data-e4drr_202505.json"
BUCKET_NAME = "cdi_arco"
PREFIX = "gdo_chirps_spi_vz/spi1"

def setup_gcs_client():
    """Set up authenticated GCS client"""
    # Load credentials from service account file
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE
    )
    
    # Create storage client
    client = storage.Client(credentials=credentials)
    return client

def list_json_files(client, bucket_name, prefix):
    """List all JSON files in the specified GCS path"""
    bucket = client.bucket(bucket_name)
    
    # List all blobs with the given prefix
    blobs = bucket.list_blobs(prefix=prefix)
    
    json_files = []
    for blob in blobs:
        if blob.name.endswith('.json'):
            json_files.append(blob.name)
            print(f"Found: {blob.name}")
    
    return json_files

def download_sample_json(client, bucket_name, blob_name, local_path):
    """Download a sample JSON file to inspect its structure"""
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    # Download to local file
    blob.download_to_filename(local_path)
    print(f"Downloaded {blob_name} to {local_path}")
    
    # Load and inspect the JSON
    with open(local_path, 'r') as f:
        data = json.load(f)
    
    # Print structure
    print(f"\nJSON structure keys: {list(data.keys())}")
    if 'refs' in data:
        print(f"Number of refs: {len(data['refs'])}")
        # Show a sample ref
        sample_ref = list(data['refs'].items())[0]
        print(f"Sample ref: {sample_ref}")
    
    return data

def main():
    """Main function to test GCS access"""
    print("Testing GCS access to SPI data...")
    
    # Check if service account file exists
    if not os.path.exists(SERVICE_ACCOUNT_FILE):
        print(f"Error: Service account file {SERVICE_ACCOUNT_FILE} not found!")
        return
    
    try:
        # Set up GCS client
        client = setup_gcs_client()
        print("Successfully authenticated with GCS")
        
        # List JSON files
        print(f"\nListing JSON files in gs://{BUCKET_NAME}/{PREFIX}")
        json_files = list_json_files(client, bucket_name=BUCKET_NAME, prefix=PREFIX)
        
        print(f"\nTotal JSON files found: {len(json_files)}")
        
        # Download a sample file if any found
        if json_files:
            sample_file = json_files[0]
            local_path = "sample_spi1.json"
            
            print(f"\nDownloading sample file: {sample_file}")
            download_sample_json(client, BUCKET_NAME, sample_file, local_path)
            
            # Create a file list for virtual concatenation
            gcs_paths = [f"gs://{BUCKET_NAME}/{blob}" for blob in json_files]
            
            # Save the file list
            file_list_info = {
                "bucket": BUCKET_NAME,
                "prefix": PREFIX,
                "total_files": len(json_files),
                "files": json_files,
                "gcs_paths": gcs_paths
            }
            
            with open("spi1_gcs_files.json", "w") as f:
                json.dump(file_list_info, f, indent=2)
            
            print(f"\nSaved file list to spi1_gcs_files.json")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()