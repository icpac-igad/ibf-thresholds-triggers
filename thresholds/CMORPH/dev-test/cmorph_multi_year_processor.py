#!/usr/bin/env python3
"""
CMORPH Multi-Year Processor - 20 Workers for 2020-2024

This script processes CMORPH precipitation data from AWS S3 to GCS Icechunk
using parallel Coiled processing with 20 workers.

Based on the CMORPH_FULL_PROCESSING_PLAN.md specifications:
- S3 Bucket: s3://noaa-cdr-precip-cmorph-pds/
- Path: data/30min/8km/{year}/{month}/{day}/*.nc
- Year-wise branches: year_2020, year_2021, etc.
- 20 workers, batch size 100

Expected data for 2020-2024:
- 2020: 8,784 files (leap year)
- 2021: 8,760 files
- 2022: 8,760 files
- 2023: 8,760 files
- 2024: 8,784 files (leap year)
- Total: ~43,848 files

Author: AI Assistant
Date: 2026-02-04
"""

import os
import json
import logging
import tempfile
import time
import math
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
        logging.FileHandler('cmorph_multi_year_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration constants
S3_BUCKET = "s3://noaa-cdr-precip-cmorph-pds/"
S3_REGION = "us-east-1"
DEFAULT_GCS_BUCKET = "cpc_awc"
DEFAULT_GCS_PREFIX = "cmorph_2020_2024"
DEFAULT_SERVICE_ACCOUNT_FILE = "coiled-data-e4drr_202505.json"
DEFAULT_LOCAL_STORE_PATH = "./icechunk_cmorph_local"

# Processing configuration
DEFAULT_N_WORKERS = 20
DEFAULT_BATCH_SIZE = 100


def get_s3_files_for_year(year: int) -> List[str]:
    """
    Get all S3 NetCDF files for an entire year.

    Args:
        year: Year to get files for

    Returns:
        List of S3 URLs sorted by filename (chronological)
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
    a unique batch branch to avoid concurrent commit conflicts.

    Args:
        args: Tuple of (batch_id, year, s3_urls, s3_bucket, s3_region,
                       creds_content, gcs_bucket, gcs_prefix, group_name)

    Returns:
        Dictionary with batch processing results
    """
    import os
    import tempfile
    import time

    (batch_id, year, s3_urls, s3_bucket, s3_region,
     creds_content, gcs_bucket, gcs_prefix, group_name) = args

    # Use unique branch per batch to avoid concurrent commit conflicts
    branch_name = f"batch_{year}_{batch_id}"
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
                'year': year,
                'status': 'error',
                'error': f'No files processed successfully. First errors: {failed_files[:3]}',
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

        # Open repository
        repo = icechunk.Repository.open(storage, config=config)

        # Create unique branch for this batch (from main)
        # Each batch gets its own branch to avoid concurrent commit conflicts
        try:
            repo.create_branch(branch_name, repo.lookup_branch("main"))
        except Exception as branch_err:
            # Branch might exist from a previous run - delete and recreate
            if "already exists" in str(branch_err).lower():
                try:
                    repo.delete_branch(branch_name)
                    repo.create_branch(branch_name, repo.lookup_branch("main"))
                except:
                    pass  # If delete fails, try to use existing branch

        session = repo.writable_session(branch_name)

        # Write to Icechunk - each batch branch has fresh group
        combined_ds.virtualize.to_icechunk(session.store, group=group_name)

        # Commit - no conflicts since each batch has its own branch
        commit_msg = f"Year {year} batch {batch_id}: Added {len(processed_files)} files"
        snapshot_id = session.commit(message=commit_msg)

        processing_time = time.time() - start_time

        return {
            'batch_id': batch_id,
            'year': year,
            'branch_name': branch_name,
            'status': 'success',
            'processed_files': len(processed_files),
            'failed_files_count': len(failed_files),
            'failed_files': failed_files[:5],  # Only keep first 5 failures for logging
            'snapshot_id': str(snapshot_id),
            'processing_time': processing_time
        }

    except Exception as e:
        import traceback
        return {
            'batch_id': batch_id,
            'year': year,
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
    service_account_file: str
) -> str:
    """
    Create or verify Icechunk repository exists in GCS.

    Args:
        gcs_bucket: GCS bucket name
        gcs_prefix: GCS prefix/path
        service_account_file: Path to service account JSON

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
        repo = icechunk.Repository.open(storage, config=config)
        logger.info("Opened existing Icechunk repository")
        return "opened"
    except Exception as e:
        logger.info(f"Creating new Icechunk repository ({e})")
        repo = icechunk.Repository.create(storage, config=config)
        logger.info("Created new Icechunk repository")
        return "created"


def process_year(
    year: int,
    client: Client,
    gcs_bucket: str,
    gcs_prefix: str,
    creds_content: str,
    batch_size: int = 100,
    group_name: str = "cmorph",
    max_files: Optional[int] = None
) -> Dict[str, Any]:
    """
    Process one year of CMORPH data using an existing Coiled cluster.

    Args:
        year: Year to process
        client: Dask distributed client
        gcs_bucket: GCS bucket name
        gcs_prefix: GCS prefix
        creds_content: Service account JSON content
        batch_size: Files per batch
        group_name: Zarr group name
        max_files: Optional limit on files (for testing)

    Returns:
        Dictionary with year processing results
    """
    logger.info(f"Processing year {year}...")
    start_time = time.time()

    # Get S3 files for this year
    s3_files = get_s3_files_for_year(year)

    if not s3_files:
        return {
            'year': year,
            'status': 'error',
            'error': f'No files found for year {year}'
        }

    if max_files:
        s3_files = s3_files[:max_files]
        logger.info(f"Limited to {max_files} files for testing")

    # Split into batches
    n_batches = math.ceil(len(s3_files) / batch_size)
    batches = [s3_files[i*batch_size:(i+1)*batch_size] for i in range(n_batches)]
    logger.info(f"Year {year}: {len(s3_files)} files in {n_batches} batches")

    # Prepare batch arguments
    task_args = [
        (i, year, batch, S3_BUCKET, S3_REGION, creds_content,
         gcs_bucket, gcs_prefix, group_name)
        for i, batch in enumerate(batches)
    ]

    # Submit all batch tasks
    logger.info(f"Submitting {len(task_args)} batch tasks for year {year}...")
    futures = client.map(process_batch_worker, task_args)

    # Collect results
    results = {
        'year': year,
        'total_files': len(s3_files),
        'n_batches': n_batches,
        'successful_batches': 0,
        'failed_batches': 0,
        'processed_files': 0,
        'failed_files': 0,
        'errors': [],
        'batch_branches': []  # Track individual batch branches
    }

    for i, future in enumerate(as_completed(futures), 1):
        try:
            result = future.result()

            if result['status'] == 'success':
                results['successful_batches'] += 1
                results['processed_files'] += result.get('processed_files', 0)
                results['failed_files'] += result.get('failed_files_count', 0)
                results['batch_branches'].append(result.get('branch_name'))
                logger.info(f"Year {year} batch {result['batch_id']}: "
                          f"{result.get('processed_files', 0)} files in "
                          f"{result.get('processing_time', 0):.1f}s -> {result.get('branch_name')}")
            else:
                results['failed_batches'] += 1
                error_msg = f"Batch {result.get('batch_id')}: {result.get('error', 'Unknown')}"
                results['errors'].append(error_msg)
                logger.error(f"Year {year} {error_msg}")
                # Log detailed file errors if available
                if result.get('failed_files'):
                    for ff in result.get('failed_files', [])[:3]:
                        logger.error(f"  File error: {ff}")
                if result.get('traceback'):
                    logger.error(f"  Traceback: {result.get('traceback')[:500]}")

            if i % 10 == 0:
                logger.info(f"Year {year} progress: {i}/{n_batches} batches completed")

        except Exception as e:
            logger.error(f"Error getting batch result for year {year}: {e}")
            results['failed_batches'] += 1
            results['errors'].append(f"Future error: {str(e)}")

    # Calculate final stats
    processing_time = time.time() - start_time
    results['processing_time_seconds'] = processing_time
    results['processing_time_minutes'] = processing_time / 60
    results['success_rate'] = (
        (results['processed_files'] / results['total_files']) * 100
        if results['total_files'] > 0 else 0
    )

    logger.info(f"Year {year} completed: {results['processed_files']}/{results['total_files']} files "
              f"({results['success_rate']:.1f}%) in {results['processing_time_minutes']:.1f} minutes")

    return results


def process_multi_year(
    years: List[int],
    gcs_bucket: str = DEFAULT_GCS_BUCKET,
    gcs_prefix: str = DEFAULT_GCS_PREFIX,
    service_account_file: str = DEFAULT_SERVICE_ACCOUNT_FILE,
    n_workers: int = DEFAULT_N_WORKERS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    group_name: str = "cmorph",
    max_files_per_year: Optional[int] = None,
    resume_from_year: Optional[int] = None
) -> Dict[str, Any]:
    """
    Process multiple years of CMORPH data with a shared Coiled cluster.

    Args:
        years: List of years to process
        gcs_bucket: GCS bucket name
        gcs_prefix: GCS prefix/path
        service_account_file: Service account JSON file
        n_workers: Number of Coiled workers
        batch_size: Files per batch
        group_name: Zarr group name
        max_files_per_year: Max files per year (for testing)
        resume_from_year: Skip years before this (for resuming)

    Returns:
        Dictionary with all processing results
    """
    logger.info("=" * 80)
    logger.info("CMORPH Multi-Year Processing - 20 Workers")
    logger.info("=" * 80)
    logger.info(f"Years: {years}")
    logger.info(f"GCS: gs://{gcs_bucket}/{gcs_prefix}")
    logger.info(f"Workers: {n_workers}")
    logger.info(f"Batch size: {batch_size}")
    if max_files_per_year:
        logger.info(f"Max files per year: {max_files_per_year}")
    if resume_from_year:
        logger.info(f"Resuming from year: {resume_from_year}")
    logger.info("=" * 80)

    start_time = time.time()

    # Load credentials
    with open(service_account_file, 'r') as f:
        creds_content = f.read()

    # Create repository if needed
    create_icechunk_repository(gcs_bucket, gcs_prefix, service_account_file)

    # Filter years if resuming
    if resume_from_year:
        years = [y for y in years if y >= resume_from_year]
        logger.info(f"Processing years: {years}")

    # Create Coiled cluster with full package sync
    logger.info(f"Starting Coiled cluster with {n_workers} workers...")
    cluster = coiled.Cluster(
        name=f"cmorph-multi-year-{int(time.time()) % 10000}",
        n_workers=n_workers,
        worker_vm_types="n2-standard-4",  # 4 vCPU, 16GB RAM per worker
        package_sync=True,  # Sync full local environment
        region="us-east1",  # Close to S3 data
        workspace="e4drr",
        idle_timeout="30 minutes",
    )
    client = Client(cluster)
    logger.info(f"Cluster ready: {client.dashboard_link}")

    # Initialize results
    all_results = {
        'start_time': datetime.now().isoformat(),
        'years': years,
        'n_workers': n_workers,
        'batch_size': batch_size,
        'gcs_location': f"gs://{gcs_bucket}/{gcs_prefix}",
        'year_results': {},
        'summary': {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'successful_years': 0,
            'failed_years': 0
        }
    }

    try:
        # Process each year
        for year in years:
            logger.info(f"\n{'='*40}")
            logger.info(f"Starting year {year}")
            logger.info(f"{'='*40}")

            year_result = process_year(
                year=year,
                client=client,
                gcs_bucket=gcs_bucket,
                gcs_prefix=gcs_prefix,
                creds_content=creds_content,
                batch_size=batch_size,
                group_name=group_name,
                max_files=max_files_per_year
            )

            all_results['year_results'][str(year)] = year_result

            # Update summary
            all_results['summary']['total_files'] += year_result.get('total_files', 0)
            all_results['summary']['processed_files'] += year_result.get('processed_files', 0)
            all_results['summary']['failed_files'] += year_result.get('failed_files', 0)

            if year_result.get('success_rate', 0) > 95:
                all_results['summary']['successful_years'] += 1
            else:
                all_results['summary']['failed_years'] += 1

            # Save checkpoint after each year
            checkpoint_file = Path(f"cmorph_multi_year_checkpoint_{year}.json")
            with open(checkpoint_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            logger.info(f"Checkpoint saved: {checkpoint_file}")

    except Exception as e:
        logger.error(f"Error during multi-year processing: {e}")
        all_results['error'] = str(e)
    finally:
        logger.info("Cleaning up cluster...")
        client.close()
        cluster.close()

    # Final results
    end_time = time.time()
    total_time = end_time - start_time

    all_results['end_time'] = datetime.now().isoformat()
    all_results['total_processing_time_seconds'] = total_time
    all_results['total_processing_time_minutes'] = total_time / 60
    all_results['total_processing_time_hours'] = total_time / 3600

    if all_results['summary']['total_files'] > 0:
        all_results['summary']['overall_success_rate'] = (
            all_results['summary']['processed_files'] /
            all_results['summary']['total_files'] * 100
        )
        all_results['summary']['files_per_minute'] = (
            all_results['summary']['processed_files'] / (total_time / 60)
        )

    # Save final results
    results_file = Path("cmorph_2020_2024_final_results.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Years processed: {len(years)}")
    logger.info(f"Total files: {all_results['summary']['total_files']}")
    logger.info(f"Processed: {all_results['summary']['processed_files']}")
    logger.info(f"Failed: {all_results['summary']['failed_files']}")
    logger.info(f"Success rate: {all_results['summary'].get('overall_success_rate', 0):.1f}%")
    logger.info(f"Total time: {all_results['total_processing_time_hours']:.2f} hours")
    logger.info(f"Rate: {all_results['summary'].get('files_per_minute', 0):.1f} files/minute")
    logger.info(f"Results saved to: {results_file}")
    logger.info("=" * 80)

    return all_results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='CMORPH Multi-Year Processor - 20 Workers for 2020-2024'
    )
    parser.add_argument(
        '--years',
        type=int,
        nargs='+',
        default=[2020, 2021, 2022, 2023, 2024],
        help='Years to process (default: 2020-2024)'
    )
    parser.add_argument(
        '--gcs-bucket',
        default=DEFAULT_GCS_BUCKET,
        help='GCS bucket name'
    )
    parser.add_argument(
        '--gcs-prefix',
        default=DEFAULT_GCS_PREFIX,
        help='GCS prefix/path'
    )
    parser.add_argument(
        '--service-account',
        default=DEFAULT_SERVICE_ACCOUNT_FILE,
        help='Service account JSON file'
    )
    parser.add_argument(
        '--n-workers',
        type=int,
        default=DEFAULT_N_WORKERS,
        help='Number of Coiled workers (default: 20)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help='Files per batch (default: 100)'
    )
    parser.add_argument(
        '--max-files-per-year',
        type=int,
        default=None,
        help='Max files per year (for testing)'
    )
    parser.add_argument(
        '--resume-from',
        type=int,
        default=None,
        help='Resume from specific year'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode: process only 50 files per year'
    )

    args = parser.parse_args()

    # Test mode
    if args.test:
        args.max_files_per_year = 50
        logger.info("TEST MODE: Processing 50 files per year")

    results = process_multi_year(
        years=args.years,
        gcs_bucket=args.gcs_bucket,
        gcs_prefix=args.gcs_prefix,
        service_account_file=args.service_account,
        n_workers=args.n_workers,
        batch_size=args.batch_size,
        max_files_per_year=args.max_files_per_year,
        resume_from_year=args.resume_from
    )

    return results


if __name__ == "__main__":
    main()
