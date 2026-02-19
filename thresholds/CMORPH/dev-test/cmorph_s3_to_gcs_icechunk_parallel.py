#!/usr/bin/env python3
"""
CMORPH S3 to GCS Icechunk - Parallel Coiled Processing

This script creates and updates virtual Icechunk stores in Google Cloud Storage (GCS)
from CMORPH data in AWS S3 using parallel processing with Coiled Cluster.

Key features:
- Parallel virtual dataset creation using Coiled dask distributed
- Single Icechunk commit after collecting all virtual datasets
- Virtual Chunk Containers reference original S3 data (no duplication)
- GCS storage for the Icechunk metadata/store
- Credentials passed securely to workers

Author: AI Assistant
Date: 2025-01-30
"""

import os
import json
import logging
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import coiled
from dask.distributed import Client, as_completed
import fsspec

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cmorph_parallel_icechunk.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration constants
S3_BUCKET = "s3://noaa-cdr-precip-cmorph-pds/"
S3_REGION = "us-east-1"
DEFAULT_GCS_BUCKET = "cpc_awc"
DEFAULT_GCS_PREFIX = "cmorph_parallel_test"
DEFAULT_SERVICE_ACCOUNT_FILE = "coiled-data-e4drr_202505.json"


def get_s3_files_for_month(year: int, month: int) -> List[str]:
    """
    Get all S3 NetCDF files for a specific month using fsspec.

    Args:
        year: Year (e.g., 2024)
        month: Month (1-12)

    Returns:
        List of S3 URLs for the month
    """
    logger.info(f"Getting S3 files for {year}-{month:02d}...")

    try:
        fs = fsspec.filesystem("s3", anon=True)
        pattern = f"noaa-cdr-precip-cmorph-pds/data/30min/8km/{year}/{month:02d}/**/*.nc"
        s3_files = fs.glob(pattern)
        s3_urls = sorted([f"s3://{f}" for f in s3_files])

        logger.info(f"Found {len(s3_urls)} S3 files for {year}-{month:02d}")
        return s3_urls

    except Exception as e:
        logger.error(f"Error getting S3 files for {year}-{month:02d}: {e}")
        return []


def get_s3_files_for_year(year: int) -> List[str]:
    """
    Get all S3 NetCDF files for an entire year.

    Args:
        year: Year to get files for

    Returns:
        List of S3 URLs
    """
    logger.info(f"Getting all S3 files for year {year}...")

    try:
        fs = fsspec.filesystem("s3", anon=True)
        pattern = f"noaa-cdr-precip-cmorph-pds/data/30min/8km/{year}/**/*.nc"
        s3_files = fs.glob(pattern)
        s3_urls = sorted([f"s3://{f}" for f in s3_files])

        logger.info(f"Found {len(s3_urls)} S3 files for year {year}")
        return s3_urls

    except Exception as e:
        logger.error(f"Error getting S3 files for year {year}: {e}")
        return []


def process_batch_worker(args: Tuple) -> Dict[str, Any]:
    """
    Worker function to process a batch of S3 files and write to Icechunk.

    Each worker processes multiple files, concatenates them, and writes to
    a unique branch to avoid concurrency conflicts.

    Args:
        args: Tuple of (batch_id, s3_urls, s3_bucket, s3_region, creds_content, gcs_bucket, gcs_prefix, group_name)

    Returns:
        Dictionary with batch processing results
    """
    import os
    import tempfile
    import time

    batch_id, s3_urls, s3_bucket, s3_region, creds_content, gcs_bucket, gcs_prefix, group_name = args
    branch_name = f"batch_{batch_id}"
    creds_file = None

    try:
        start_time = time.time()

        # Write credentials to temp file
        creds_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        creds_file.write(creds_content)
        creds_file.close()

        # Import required libraries
        import icechunk
        import xarray as xr
        from virtualizarr import open_virtual_dataset
        from virtualizarr.parsers import HDFParser
        from virtualizarr.registry import ObjectStoreRegistry
        from obstore.store import from_url

        # Create S3 store and registry for VirtualiZarr
        s3_store = from_url(s3_bucket, region=s3_region, skip_signature=True)
        registry = ObjectStoreRegistry({s3_bucket: s3_store})
        parser = HDFParser()

        # Process all files in the batch
        virtual_datasets = []
        processed_files = []
        failed_files = []

        for s3_url in s3_urls:
            filename = s3_url.split('/')[-1]
            try:
                vds = open_virtual_dataset(
                    url=s3_url,
                    parser=parser,
                    registry=registry
                )
                virtual_datasets.append(vds)
                processed_files.append(filename)
            except Exception as e:
                failed_files.append({'file': filename, 'error': str(e)})

        if not virtual_datasets:
            return {
                'batch_id': batch_id,
                'status': 'error',
                'error': 'No files processed successfully',
                'failed_files': failed_files
            }

        # Concatenate virtual datasets
        if len(virtual_datasets) > 1:
            combined_ds = xr.concat(virtual_datasets, dim='time')
        else:
            combined_ds = virtual_datasets[0]

        # Setup Icechunk
        config = icechunk.RepositoryConfig.default()
        container = icechunk.VirtualChunkContainer(
            s3_bucket,
            store=icechunk.s3_store(region=s3_region, anonymous=True),
        )
        config.set_virtual_chunk_container(container)

        storage = icechunk.gcs_storage(
            bucket=gcs_bucket,
            prefix=gcs_prefix,
            service_account_file=creds_file.name
        )

        # Open repository and create a new branch for this batch
        try:
            repo = icechunk.Repository.open(storage, config=config)
        except:
            repo = icechunk.Repository.create(storage, config=config)

        # Create branch from main
        try:
            repo.create_branch(branch_name, repo.lookup_branch("main"))
        except:
            pass  # Branch may already exist

        session = repo.writable_session(branch_name)

        # Write to Icechunk
        combined_ds.virtualize.to_icechunk(session.store, group=group_name)

        # Commit
        commit_msg = f"Batch {batch_id}: Added {len(processed_files)} files"
        snapshot_id = session.commit(message=commit_msg)

        processing_time = time.time() - start_time

        return {
            'batch_id': batch_id,
            'branch_name': branch_name,
            'status': 'success',
            'processed_files': processed_files,
            'failed_files': failed_files,
            'snapshot_id': str(snapshot_id),
            'processing_time': processing_time
        }

    except Exception as e:
        import traceback
        return {
            'batch_id': batch_id,
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc()
        }
    finally:
        if creds_file is not None:
            try:
                os.unlink(creds_file.name)
            except:
                pass


def process_single_file_worker(args: Tuple) -> Dict[str, Any]:
    """
    Worker function to create a virtual dataset from a single S3 file.

    This runs on Coiled workers and returns serialized virtual dataset info.

    Args:
        args: Tuple of (s3_url, s3_bucket, s3_region)

    Returns:
        Dictionary with virtual dataset references or error info
    """
    import time
    s3_url, s3_bucket, s3_region = args
    filename = s3_url.split('/')[-1]

    try:
        start_time = time.time()

        # Import required libraries
        from virtualizarr import open_virtual_dataset
        from virtualizarr.parsers import HDFParser
        from virtualizarr.registry import ObjectStoreRegistry
        from obstore.store import from_url

        # Create S3 store and registry
        s3_store = from_url(s3_bucket, region=s3_region, skip_signature=True)
        registry = ObjectStoreRegistry({s3_bucket: s3_store})

        # Create parser
        parser = HDFParser()

        # Create virtual dataset
        virtual_ds = open_virtual_dataset(
            url=s3_url,
            parser=parser,
            registry=registry
        )

        # Serialize the virtual dataset to kerchunk dict for transport
        refs_dict = virtual_ds.virtualize.to_kerchunk(format='dict')

        processing_time = time.time() - start_time

        return {
            'file': filename,
            's3_url': s3_url,
            'status': 'success',
            'refs': refs_dict,
            'processing_time': processing_time
        }

    except Exception as e:
        import traceback
        return {
            'file': filename,
            's3_url': s3_url,
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc()
        }


def write_to_icechunk_worker(args: Tuple) -> Dict[str, Any]:
    """
    Worker function to write a virtual dataset directly to Icechunk.

    This runs on Coiled workers and writes directly to the shared Icechunk store.
    Note: This approach requires careful handling of concurrent writes.

    Args:
        args: Tuple of (s3_url, s3_bucket, s3_region, creds_content, gcs_bucket, gcs_prefix, group_name)

    Returns:
        Dictionary with processing result
    """
    import os
    import tempfile
    import time

    s3_url, s3_bucket, s3_region, creds_content, gcs_bucket, gcs_prefix, group_name = args
    filename = s3_url.split('/')[-1]
    creds_file = None

    try:
        start_time = time.time()

        # Write credentials to temp file
        creds_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        creds_file.write(creds_content)
        creds_file.close()

        # Import required libraries
        import icechunk
        from virtualizarr import open_virtual_dataset
        from virtualizarr.parsers import HDFParser
        from virtualizarr.registry import ObjectStoreRegistry
        from obstore.store import from_url

        # Create S3 store and registry for VirtualiZarr
        s3_store = from_url(s3_bucket, region=s3_region, skip_signature=True)
        registry = ObjectStoreRegistry({s3_bucket: s3_store})

        # Create virtual dataset
        parser = HDFParser()
        virtual_ds = open_virtual_dataset(
            url=s3_url,
            parser=parser,
            registry=registry
        )

        # Setup Icechunk repository config with virtual chunk container
        config = icechunk.RepositoryConfig.default()
        container = icechunk.VirtualChunkContainer(
            s3_bucket,
            store=icechunk.s3_store(region=s3_region, anonymous=True),
        )
        config.set_virtual_chunk_container(container)

        # Setup GCS storage
        storage = icechunk.gcs_storage(
            bucket=gcs_bucket,
            prefix=gcs_prefix,
            service_account_file=creds_file.name
        )

        # Open existing repository (should be created beforehand)
        repo = icechunk.Repository.open(storage, config=config)
        session = repo.writable_session("main")

        # Write virtual dataset to Icechunk (appends to existing data)
        virtual_ds.virtualize.to_icechunk(session.store, group=group_name)

        # Commit this individual file
        commit_msg = f"Added {filename}"
        snapshot_id = session.commit(message=commit_msg)

        processing_time = time.time() - start_time

        return {
            'file': filename,
            's3_url': s3_url,
            'status': 'success',
            'snapshot_id': str(snapshot_id),
            'processing_time': processing_time
        }

    except Exception as e:
        import traceback
        return {
            'file': filename,
            's3_url': s3_url,
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc()
        }
    finally:
        if creds_file is not None:
            try:
                os.unlink(creds_file.name)
            except:
                pass


def create_icechunk_repository(
    gcs_bucket: str,
    gcs_prefix: str,
    service_account_file: str,
    force_new: bool = False
) -> str:
    """
    Create or verify Icechunk repository exists in GCS.

    Args:
        gcs_bucket: GCS bucket name
        gcs_prefix: GCS prefix/path
        service_account_file: Path to service account JSON
        force_new: If True, create new repository even if one exists

    Returns:
        Repository status message
    """
    import icechunk

    logger.info(f"Setting up Icechunk repository at gs://{gcs_bucket}/{gcs_prefix}")

    # Create config with virtual chunk container
    config = icechunk.RepositoryConfig.default()
    container = icechunk.VirtualChunkContainer(
        S3_BUCKET,
        store=icechunk.s3_store(region=S3_REGION, anonymous=True),
    )
    config.set_virtual_chunk_container(container)

    # Setup GCS storage
    storage = icechunk.gcs_storage(
        bucket=gcs_bucket,
        prefix=gcs_prefix,
        service_account_file=service_account_file
    )

    try:
        if force_new:
            raise Exception("Force new repository")
        repo = icechunk.Repository.open(storage, config=config)
        logger.info("Opened existing Icechunk repository")
        return "opened"
    except Exception as e:
        logger.info(f"Creating new Icechunk repository ({e})")
        repo = icechunk.Repository.create(storage, config=config)
        logger.info("Created new Icechunk repository")
        return "created"


def process_files_parallel_collect(
    s3_files: List[str],
    gcs_bucket: str,
    gcs_prefix: str,
    service_account_file: str,
    n_workers: int = 10,
    batch_size: int = 100,
    group_name: str = "cmorph"
) -> Dict[str, Any]:
    """
    Process S3 files in parallel using Coiled Cluster, collecting virtual datasets.

    This approach:
    1. Creates virtual datasets on workers in parallel
    2. Collects kerchunk references back to main process
    3. Combines and writes to Icechunk in a single commit

    Args:
        s3_files: List of S3 URLs to process
        gcs_bucket: GCS bucket for Icechunk store
        gcs_prefix: GCS prefix for Icechunk store
        service_account_file: Path to GCS service account JSON
        n_workers: Number of Coiled workers
        batch_size: Number of files to process before committing
        group_name: Zarr group name in Icechunk store

    Returns:
        Dictionary with processing results
    """
    logger.info(f"Processing {len(s3_files)} files with {n_workers} workers")
    start_time = time.time()

    results = {
        'total_files': len(s3_files),
        'successful_files': 0,
        'failed_files': 0,
        'file_results': [],
        'errors': [],
        'start_time': datetime.now().isoformat()
    }

    # Create Coiled cluster with package sync to get virtualizarr
    logger.info("Starting Coiled cluster...")
    cluster = coiled.Cluster(
        name=f"cmorph-parallel-{int(time.time()) % 10000}",
        n_workers=n_workers,
        worker_vm_types="n2-standard-2",
        package_sync=True,  # Sync local environment including virtualizarr
        region="us-east1",
        workspace="e4drr",
        idle_timeout="10 minutes",
    )
    client = Client(cluster)
    logger.info(f"Cluster ready: {client.dashboard_link}")

    try:
        # Prepare task arguments
        task_args = [(s3_url, S3_BUCKET, S3_REGION) for s3_url in s3_files]

        # Submit all tasks
        logger.info(f"Submitting {len(task_args)} tasks...")
        futures = client.map(process_single_file_worker, task_args)

        # Collect results with progress tracking
        successful_refs = []
        for i, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
                results['file_results'].append({
                    'file': result['file'],
                    's3_url': result['s3_url'],
                    'status': result['status'],
                    'processing_time': result.get('processing_time', 0)
                })

                if result['status'] == 'success':
                    results['successful_files'] += 1
                    successful_refs.append(result)
                else:
                    results['failed_files'] += 1
                    results['errors'].append(f"{result['s3_url']}: {result.get('error', 'Unknown')}")

                if i % 50 == 0:
                    logger.info(f"Progress: {i}/{len(futures)} tasks completed "
                              f"({results['successful_files']} success, {results['failed_files']} failed)")

            except Exception as e:
                logger.error(f"Error getting result: {e}")
                results['failed_files'] += 1
                results['errors'].append(f"Future error: {str(e)}")

        logger.info(f"Collected {len(successful_refs)} virtual dataset references")

        # Now write collected references to Icechunk
        if successful_refs:
            logger.info("Writing to Icechunk store...")
            write_results = write_refs_to_icechunk(
                successful_refs,
                gcs_bucket,
                gcs_prefix,
                service_account_file,
                group_name
            )
            results['icechunk_write'] = write_results

    except Exception as e:
        logger.error(f"Error during parallel processing: {e}")
        results['errors'].append(f"Cluster error: {str(e)}")
    finally:
        logger.info("Cleaning up cluster...")
        client.close()
        cluster.close()

    # Final results
    end_time = time.time()
    processing_time = end_time - start_time

    results.update({
        'end_time': datetime.now().isoformat(),
        'processing_time_seconds': processing_time,
        'processing_time_minutes': processing_time / 60,
        'success_rate': (results['successful_files'] / results['total_files']) * 100 if results['total_files'] > 0 else 0,
        'gcs_location': f"gs://{gcs_bucket}/{gcs_prefix}"
    })

    logger.info("Processing Complete!")
    logger.info(f"Total files: {results['total_files']}")
    logger.info(f"Successful: {results['successful_files']}")
    logger.info(f"Failed: {results['failed_files']}")
    logger.info(f"Success rate: {results['success_rate']:.1f}%")
    logger.info(f"Processing time: {results['processing_time_minutes']:.2f} minutes")

    return results


def write_refs_to_icechunk(
    refs_list: List[Dict],
    gcs_bucket: str,
    gcs_prefix: str,
    service_account_file: str,
    group_name: str
) -> Dict[str, Any]:
    """
    Write collected virtual dataset references to Icechunk.

    Args:
        refs_list: List of dictionaries containing kerchunk refs
        gcs_bucket: GCS bucket name
        gcs_prefix: GCS prefix
        service_account_file: Path to service account JSON
        group_name: Zarr group name

    Returns:
        Dictionary with write results
    """
    import icechunk
    import xarray as xr
    import tempfile
    from virtualizarr import open_virtual_dataset
    from virtualizarr.parsers import KerchunkJSONParser
    from virtualizarr.registry import ObjectStoreRegistry
    from obstore.store import LocalStore

    logger.info(f"Writing {len(refs_list)} datasets to Icechunk...")
    start_time = time.time()

    # Create config with virtual chunk container
    config = icechunk.RepositoryConfig.default()
    container = icechunk.VirtualChunkContainer(
        S3_BUCKET,
        store=icechunk.s3_store(region=S3_REGION, anonymous=True),
    )
    config.set_virtual_chunk_container(container)

    # Setup GCS storage
    storage = icechunk.gcs_storage(
        bucket=gcs_bucket,
        prefix=gcs_prefix,
        service_account_file=service_account_file
    )

    # Open or create repository
    try:
        repo = icechunk.Repository.open(storage, config=config)
        logger.info("Opened existing repository")
    except:
        repo = icechunk.Repository.create(storage, config=config)
        logger.info("Created new repository")

    session = repo.writable_session("main")

    # Write refs to temp files and load using KerchunkJSONParser
    virtual_datasets = []
    temp_files = []

    try:
        for ref_item in refs_list:
            try:
                # Write refs to temp JSON file
                temp_file = tempfile.NamedTemporaryFile(
                    mode='w', suffix='.json', delete=False
                )
                json.dump(ref_item['refs'], temp_file)
                temp_file.close()
                temp_files.append(temp_file.name)

                # Load virtual dataset from kerchunk JSON file
                local_store = LocalStore(prefix="/")
                registry = ObjectStoreRegistry({"file://": local_store})
                parser = KerchunkJSONParser()

                vds = open_virtual_dataset(
                    url=f"file://{temp_file.name}",
                    parser=parser,
                    registry=registry
                )
                virtual_datasets.append(vds)
                logger.info(f"Loaded refs for {ref_item['file']}")

            except Exception as e:
                logger.warning(f"Could not load refs for {ref_item['file']}: {e}")

        if virtual_datasets:
            # Concatenate along time dimension
            logger.info(f"Concatenating {len(virtual_datasets)} virtual datasets...")
            combined_ds = xr.concat(virtual_datasets, dim='time')

            # Write to Icechunk
            combined_ds.virtualize.to_icechunk(session.store, group=group_name)

            # Commit
            commit_msg = f"Added {len(virtual_datasets)} CMORPH files"
            snapshot_id = session.commit(message=commit_msg)

            write_time = time.time() - start_time

            return {
                'status': 'success',
                'datasets_written': len(virtual_datasets),
                'snapshot_id': str(snapshot_id),
                'write_time': write_time
            }
        else:
            return {
                'status': 'error',
                'error': 'No valid datasets to write'
            }
    finally:
        # Clean up temp files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass


def process_files_parallel_batch(
    s3_files: List[str],
    gcs_bucket: str,
    gcs_prefix: str,
    service_account_file: str,
    n_workers: int = 10,
    batch_size: int = 50,
    group_name: str = "cmorph"
) -> Dict[str, Any]:
    """
    Process S3 files in parallel using batches with separate branches.

    This approach:
    1. Splits files into batches
    2. Each worker processes a batch and writes to a unique branch
    3. Avoids concurrency conflicts by using separate branches
    4. Branches can be merged later

    Args:
        s3_files: List of S3 URLs to process
        gcs_bucket: GCS bucket for Icechunk store
        gcs_prefix: GCS prefix for Icechunk store
        service_account_file: Path to GCS service account JSON
        n_workers: Number of Coiled workers
        batch_size: Number of files per batch
        group_name: Zarr group name in Icechunk store

    Returns:
        Dictionary with processing results
    """
    import math

    logger.info(f"Processing {len(s3_files)} files in batches of {batch_size}")
    start_time = time.time()

    # Load credentials
    with open(service_account_file, 'r') as f:
        creds_content = f.read()

    # Create initial repository
    create_icechunk_repository(gcs_bucket, gcs_prefix, service_account_file)

    # Split into batches
    n_batches = math.ceil(len(s3_files) / batch_size)
    batches = [s3_files[i*batch_size:(i+1)*batch_size] for i in range(n_batches)]
    logger.info(f"Created {len(batches)} batches")

    results = {
        'total_files': len(s3_files),
        'n_batches': len(batches),
        'successful_batches': 0,
        'failed_batches': 0,
        'processed_files': 0,
        'failed_files': 0,
        'batch_results': [],
        'errors': [],
        'branches': [],
        'start_time': datetime.now().isoformat()
    }

    # Create Coiled cluster
    logger.info("Starting Coiled cluster...")
    cluster = coiled.Cluster(
        name=f"cmorph-batch-{int(time.time()) % 10000}",
        n_workers=min(n_workers, len(batches)),
        worker_vm_types="n2-standard-4",  # Larger VMs for batch processing
        package_sync=True,
        region="us-east1",
        workspace="e4drr",
        idle_timeout="15 minutes",
    )
    client = Client(cluster)
    logger.info(f"Cluster ready: {client.dashboard_link}")

    try:
        # Prepare batch arguments
        task_args = [
            (i, batch, S3_BUCKET, S3_REGION, creds_content, gcs_bucket, gcs_prefix, group_name)
            for i, batch in enumerate(batches)
        ]

        # Submit all batch tasks
        logger.info(f"Submitting {len(task_args)} batch tasks...")
        futures = client.map(process_batch_worker, task_args)

        # Collect results
        for i, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
                results['batch_results'].append(result)

                if result['status'] == 'success':
                    results['successful_batches'] += 1
                    results['processed_files'] += len(result.get('processed_files', []))
                    results['failed_files'] += len(result.get('failed_files', []))
                    results['branches'].append(result.get('branch_name'))
                    logger.info(f"Batch {result['batch_id']} completed: "
                              f"{len(result.get('processed_files', []))} files")
                else:
                    results['failed_batches'] += 1
                    results['errors'].append(f"Batch {result.get('batch_id')}: {result.get('error', 'Unknown')}")
                    logger.error(f"Batch {result.get('batch_id')} failed: {result.get('error')}")

                logger.info(f"Progress: {i}/{len(futures)} batches completed")

            except Exception as e:
                logger.error(f"Error getting batch result: {e}")
                results['failed_batches'] += 1
                results['errors'].append(f"Future error: {str(e)}")

    except Exception as e:
        logger.error(f"Error during batch processing: {e}")
        results['errors'].append(f"Cluster error: {str(e)}")
    finally:
        logger.info("Cleaning up cluster...")
        client.close()
        cluster.close()

    # Final results
    end_time = time.time()
    processing_time = end_time - start_time

    results.update({
        'end_time': datetime.now().isoformat(),
        'processing_time_seconds': processing_time,
        'processing_time_minutes': processing_time / 60,
        'success_rate': (results['processed_files'] / results['total_files']) * 100 if results['total_files'] > 0 else 0,
        'gcs_location': f"gs://{gcs_bucket}/{gcs_prefix}"
    })

    logger.info("Batch Processing Complete!")
    logger.info(f"Total files: {results['total_files']}")
    logger.info(f"Processed: {results['processed_files']}")
    logger.info(f"Failed: {results['failed_files']}")
    logger.info(f"Success rate: {results['success_rate']:.1f}%")
    logger.info(f"Branches created: {results['branches']}")
    logger.info(f"Processing time: {results['processing_time_minutes']:.2f} minutes")

    return results


def process_files_parallel_direct(
    s3_files: List[str],
    gcs_bucket: str,
    gcs_prefix: str,
    service_account_file: str,
    n_workers: int = 10,
    group_name: str = "cmorph"
) -> Dict[str, Any]:
    """
    Process S3 files in parallel, with each worker writing directly to Icechunk.

    This approach has each worker:
    1. Create virtual dataset
    2. Open Icechunk repository
    3. Write and commit individually

    Note: This can lead to many small commits but handles large datasets better.

    Args:
        s3_files: List of S3 URLs to process
        gcs_bucket: GCS bucket for Icechunk store
        gcs_prefix: GCS prefix for Icechunk store
        service_account_file: Path to GCS service account JSON
        n_workers: Number of Coiled workers
        group_name: Zarr group name in Icechunk store

    Returns:
        Dictionary with processing results
    """
    logger.info(f"Processing {len(s3_files)} files directly with {n_workers} workers")
    start_time = time.time()

    # Load credentials content
    with open(service_account_file, 'r') as f:
        creds_content = f.read()

    # First, ensure repository exists
    create_icechunk_repository(gcs_bucket, gcs_prefix, service_account_file)

    results = {
        'total_files': len(s3_files),
        'successful_files': 0,
        'failed_files': 0,
        'file_results': [],
        'errors': [],
        'start_time': datetime.now().isoformat()
    }

    # Create Coiled cluster with package sync to get virtualizarr
    logger.info("Starting Coiled cluster...")
    cluster = coiled.Cluster(
        name=f"cmorph-direct-{int(time.time()) % 10000}",
        n_workers=n_workers,
        worker_vm_types="n2-standard-2",
        package_sync=True,  # Sync local environment including virtualizarr
        region="us-east1",
        workspace="e4drr",
        idle_timeout="10 minutes",
    )
    client = Client(cluster)
    logger.info(f"Cluster ready: {client.dashboard_link}")

    try:
        # Prepare task arguments with credentials
        task_args = [
            (s3_url, S3_BUCKET, S3_REGION, creds_content, gcs_bucket, gcs_prefix, group_name)
            for s3_url in s3_files
        ]

        # Submit all tasks
        logger.info(f"Submitting {len(task_args)} tasks...")
        futures = client.map(write_to_icechunk_worker, task_args)

        # Collect results with progress tracking
        for i, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
                results['file_results'].append(result)

                if result['status'] == 'success':
                    results['successful_files'] += 1
                else:
                    results['failed_files'] += 1
                    results['errors'].append(f"{result['s3_url']}: {result.get('error', 'Unknown')}")

                if i % 50 == 0:
                    logger.info(f"Progress: {i}/{len(futures)} tasks completed "
                              f"({results['successful_files']} success, {results['failed_files']} failed)")

            except Exception as e:
                logger.error(f"Error getting result: {e}")
                results['failed_files'] += 1
                results['errors'].append(f"Future error: {str(e)}")

    except Exception as e:
        logger.error(f"Error during parallel processing: {e}")
        results['errors'].append(f"Cluster error: {str(e)}")
    finally:
        logger.info("Cleaning up cluster...")
        client.close()
        cluster.close()

    # Final results
    end_time = time.time()
    processing_time = end_time - start_time

    results.update({
        'end_time': datetime.now().isoformat(),
        'processing_time_seconds': processing_time,
        'processing_time_minutes': processing_time / 60,
        'success_rate': (results['successful_files'] / results['total_files']) * 100 if results['total_files'] > 0 else 0,
        'gcs_location': f"gs://{gcs_bucket}/{gcs_prefix}"
    })

    logger.info("Processing Complete!")
    logger.info(f"Total files: {results['total_files']}")
    logger.info(f"Successful: {results['successful_files']}")
    logger.info(f"Failed: {results['failed_files']}")
    logger.info(f"Success rate: {results['success_rate']:.1f}%")
    logger.info(f"Processing time: {results['processing_time_minutes']:.2f} minutes")

    return results


def process_month_parallel(
    year: int,
    month: int,
    gcs_bucket: str = DEFAULT_GCS_BUCKET,
    gcs_prefix: str = DEFAULT_GCS_PREFIX,
    service_account_file: str = DEFAULT_SERVICE_ACCOUNT_FILE,
    n_workers: int = 10,
    max_files: Optional[int] = None,
    mode: str = "batch",
    batch_size: int = 50
) -> Dict[str, Any]:
    """
    Process one month of CMORPH data using parallel Coiled processing.

    Args:
        year: Year to process
        month: Month to process (1-12)
        gcs_bucket: GCS bucket name
        gcs_prefix: GCS prefix/path
        service_account_file: Service account JSON file
        n_workers: Number of Coiled workers
        max_files: Maximum files to process (None for all)
        mode: Processing mode - "batch" (recommended), "collect", or "direct"
              - "batch": Files split into batches, each batch writes to separate branch (recommended)
              - "collect": Collect refs, then write single commit (has serialization issues)
              - "direct": Each worker writes directly (has concurrency issues)
        batch_size: Number of files per batch (for batch mode)

    Returns:
        Dictionary with processing results
    """
    logger.info(f"Processing {year}-{month:02d} with {n_workers} workers (mode: {mode})")

    # Get S3 files
    s3_files = get_s3_files_for_month(year, month)

    if not s3_files:
        logger.error(f"No S3 files found for {year}-{month:02d}")
        return {'error': f'No files found for {year}-{month:02d}'}

    if max_files:
        s3_files = s3_files[:max_files]
        logger.info(f"Limited to {max_files} files")

    # Process based on mode
    if mode == "batch":
        results = process_files_parallel_batch(
            s3_files=s3_files,
            gcs_bucket=gcs_bucket,
            gcs_prefix=gcs_prefix,
            service_account_file=service_account_file,
            n_workers=n_workers,
            batch_size=batch_size
        )
    elif mode == "collect":
        results = process_files_parallel_collect(
            s3_files=s3_files,
            gcs_bucket=gcs_bucket,
            gcs_prefix=gcs_prefix,
            service_account_file=service_account_file,
            n_workers=n_workers
        )
    else:  # direct mode
        results = process_files_parallel_direct(
            s3_files=s3_files,
            gcs_bucket=gcs_bucket,
            gcs_prefix=gcs_prefix,
            service_account_file=service_account_file,
            n_workers=n_workers
        )

    # Add metadata
    results['year'] = year
    results['month'] = month
    results['mode'] = mode

    # Save results
    results_file = Path(f"cmorph_{year}{month:02d}_parallel_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {results_file}")

    return results


def main():
    """Main entry point with example usage."""
    import argparse

    parser = argparse.ArgumentParser(description='CMORPH S3 to GCS Icechunk - Parallel Processing')
    parser.add_argument('--year', type=int, default=2024, help='Year to process')
    parser.add_argument('--month', type=int, default=1, help='Month to process (1-12)')
    parser.add_argument('--gcs-bucket', default=DEFAULT_GCS_BUCKET, help='GCS bucket name')
    parser.add_argument('--gcs-prefix', default=DEFAULT_GCS_PREFIX, help='GCS prefix/path')
    parser.add_argument('--service-account', default=DEFAULT_SERVICE_ACCOUNT_FILE, help='Service account JSON file')
    parser.add_argument('--n-workers', type=int, default=10, help='Number of Coiled workers')
    parser.add_argument('--max-files', type=int, default=None, help='Max files to process (for testing)')
    parser.add_argument('--mode', choices=['batch', 'collect', 'direct'], default='batch',
                       help='Processing mode: batch (recommended), collect, or direct')
    parser.add_argument('--batch-size', type=int, default=50, help='Files per batch (for batch mode)')

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("CMORPH S3 to GCS Icechunk - Parallel Processing")
    logger.info("=" * 80)
    logger.info(f"Year: {args.year}")
    logger.info(f"Month: {args.month}")
    logger.info(f"GCS: gs://{args.gcs_bucket}/{args.gcs_prefix}")
    logger.info(f"Workers: {args.n_workers}")
    logger.info(f"Mode: {args.mode}")
    if args.max_files:
        logger.info(f"Max files: {args.max_files}")
    logger.info("=" * 80)

    results = process_month_parallel(
        year=args.year,
        month=args.month,
        gcs_bucket=args.gcs_bucket,
        gcs_prefix=args.gcs_prefix,
        service_account_file=args.service_account,
        n_workers=args.n_workers,
        max_files=args.max_files,
        mode=args.mode,
        batch_size=args.batch_size
    )

    logger.info("=" * 80)
    logger.info("Processing finished!")
    logger.info("=" * 80)

    return results


if __name__ == "__main__":
    main()
