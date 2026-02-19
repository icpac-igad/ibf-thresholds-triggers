#!/usr/bin/env python3
"""
VirtualiZarr to Icechunk GCS Demo

This script demonstrates:
1. Opening multiple NetCDF files using VirtualiZarr 
2. Concatenating them into a single virtual dataset
3. Saving the result to an Icechunk store on Google Cloud Storage

Prerequisites:
- List of file URLs (oisst_files in this example)
- Service account JSON key for GCS access
- Icechunk and VirtualiZarr installed
"""

import os
import json
from typing import List
import xarray as xr
import icechunk
from virtualizarr import open_virtual_dataset

def demo_virtualizarr_to_icechunk_gcs(
    file_urls: List[str],
    gcs_bucket: str,
    gcs_path: str,
    service_account_json: str,
    repo_name: str = "oisst_dataset"
):
    """
    Demonstrate VirtualiZarr to Icechunk GCS workflow
    
    Args:
        file_urls: List of URLs to NetCDF files
        gcs_bucket: GCS bucket name (without gs:// prefix)
        gcs_path: Path within the bucket for Icechunk store
        service_account_json: Path to service account JSON key file
        repo_name: Name for the Icechunk repository
    """
    
    print(f"Processing {len(file_urls)} files with VirtualiZarr...")
    
    # Step 1: Open multiple files as virtual datasets
    print("Opening virtual datasets...")
    virtual_datasets = [
        open_virtual_dataset(url, indexes={})
        for url in file_urls
    ]
    print(f"Opened {len(virtual_datasets)} virtual datasets")
    
    # Step 2: Concatenate virtual datasets along time dimension
    print("Concatenating virtual datasets...")
    virtual_ds = xr.concat(
        virtual_datasets,
        dim='time',
        coords='minimal',
        compat='override',
        combine_attrs='override'
    )
    print(f"Concatenated dataset shape: {virtual_ds.dims}")
    
    # Step 3: Set up Icechunk GCS storage
    print("Setting up Icechunk GCS storage...")
    
    # Configure GCS storage
    gcs_storage_path = f"gs://{gcs_bucket}/{gcs_path}"
    storage = icechunk.gcs_storage(
        bucket=gcs_bucket,
        prefix=gcs_path,
        service_account_json_path=service_account_json
    )
    
    # Set up repository configuration with virtual chunk container
    config = icechunk.RepositoryConfig.default()
    
    # Configure virtual chunk container to point to original data locations
    # This assumes the original data is also accessible (adjust as needed)
    virtual_chunk_container_url = "s3://your-source-bucket/path/"  # Adjust as needed
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(
            virtual_chunk_container_url,
            icechunk.s3_store(region="us-east-1")  # Adjust region as needed
        )
    )
    
    # Set up credentials for virtual chunk container access
    credentials = icechunk.containers_credentials({
        virtual_chunk_container_url: icechunk.s3_credentials(anonymous=True)
    })
    
    # Step 4: Create Icechunk repository
    print(f"Creating Icechunk repository at {gcs_storage_path}...")
    try:
        repo = icechunk.Repository.create(storage, config, credentials)
        print("Repository created successfully")
    except Exception as e:
        print(f"Repository creation failed: {e}")
        # Try to open existing repository
        print("Attempting to open existing repository...")
        repo = icechunk.Repository.open(storage, config, credentials)
        print("Opened existing repository")
    
    # Step 5: Write virtual dataset to Icechunk
    print("Writing virtual dataset to Icechunk...")
    
    # Create a new session for writing
    session = repo.writable_session("main")
    
    # Write the virtual dataset
    virtual_ds.to_zarr(
        store=session,
        mode='w',
        consolidated=True
    )
    
    # Commit the session
    snapshot_id = session.commit("Initial commit of concatenated OISST data")
    print(f"Committed data with snapshot ID: {snapshot_id}")
    
    # Step 6: Verify the written data
    print("Verifying written data...")
    readonly_session = repo.readonly_session("main")
    
    # Read back the data to verify
    reopened_ds = xr.open_zarr(readonly_session)
    print(f"Reopened dataset shape: {reopened_ds.dims}")
    print(f"Variables: {list(reopened_ds.data_vars)}")
    
    print("Demo completed successfully!")
    return repo, virtual_ds, reopened_ds


def demo_with_sample_files():
    """
    Demo with sample OISST file URLs
    """
    
    # Example OISST file URLs (replace with your actual URLs)
    oisst_files = [
        "s3://noaa-oisst-pds/data/v2.1/monthly/2023/oisst-avhrr-v02r01.20230101.nc",
        "s3://noaa-oisst-pds/data/v2.1/monthly/2023/oisst-avhrr-v02r01.20230201.nc", 
        "s3://noaa-oisst-pds/data/v2.1/monthly/2023/oisst-avhrr-v02r01.20230301.nc"
    ]
    
    # GCS configuration
    gcs_bucket = "your-gcs-bucket"  # Replace with your bucket
    gcs_path = "icechunk-stores/oisst-demo"
    service_account_json = "/path/to/your/service-account.json"  # Replace with your JSON key
    
    try:
        repo, virtual_ds, reopened_ds = demo_virtualizarr_to_icechunk_gcs(
            file_urls=oisst_files,
            gcs_bucket=gcs_bucket,
            gcs_path=gcs_path,
            service_account_json=service_account_json
        )
        
        print("\nDemo Results:")
        print(f"Original virtual dataset: {virtual_ds}")
        print(f"Reopened dataset: {reopened_ds}")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        raise


def demo_local_icechunk_alternative():
    """
    Alternative demo using local Icechunk storage (for testing)
    """
    
    print("Running local Icechunk demo...")
    
    # Sample file URLs (replace with your actual files)
    sample_files = [
        "sample_file_1.nc",  # Replace with actual URLs
        "sample_file_2.nc",
        "sample_file_3.nc"
    ]
    
    # Step 1: Open virtual datasets
    virtual_datasets = [
        open_virtual_dataset(url, indexes={})
        for url in sample_files
    ]
    
    # Step 2: Concatenate
    virtual_ds = xr.concat(
        virtual_datasets,
        dim='time',
        coords='minimal', 
        compat='override',
        combine_attrs='override'
    )
    
    # Step 3: Local Icechunk storage setup
    storage = icechunk.local_filesystem_storage(
        path='oisst_local',
    )
    
    config = icechunk.RepositoryConfig.default()
    
    # Configure virtual chunk container for original data location
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(
            "s3://mybucket/my/data/",  # Replace with your source
            icechunk.s3_store(region="us-east-1")
        )
    )
    
    credentials = icechunk.containers_credentials({
        "s3://mybucket/my/data/": icechunk.s3_credentials(anonymous=True)
    })
    
    # Step 4: Create repository
    repo = icechunk.Repository.create(storage, config, credentials)
    
    # Step 5: Write data
    session = repo.writable_session("main")
    virtual_ds.to_zarr(store=session, mode='w', consolidated=True)
    snapshot_id = session.commit("Local demo commit")
    
    print(f"Local demo completed with snapshot: {snapshot_id}")


if __name__ == "__main__":
    print("VirtualiZarr to Icechunk GCS Demo")
    print("=" * 40)
    
    # Check if service account JSON is available
    service_account_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    
    if service_account_path and os.path.exists(service_account_path):
        print("Found service account credentials, running GCS demo...")
        demo_with_sample_files()
    else:
        print("No service account found, running local demo instead...")
        print("To run GCS demo, set GOOGLE_APPLICATION_CREDENTIALS environment variable")
        demo_local_icechunk_alternative()