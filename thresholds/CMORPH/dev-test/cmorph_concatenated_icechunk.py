#!/usr/bin/env python3
"""
CMORPH Concatenated Icechunk Store Creator (Coiled-accelerated)
===============================================================
Creates a properly concatenated Icechunk store from CMORPH S3 data.

Strategy: Parallel metadata reads via dask.delayed on Coiled,
sequential Icechunk writes with append_dim='time'.

- open_virtual_mfdataset with parallel='dask' distributes per-file
  S3 metadata reads across Coiled workers (~6 min -> faster)
- to_icechunk append is sequential (~4 sec per month, no conflicts)

Reference: https://virtualizarr.readthedocs.io/en/stable/usage.html#manual-concatenation-ordering

Usage:
    # New store: process 1998 (creates store, first month has no append_dim)
    micromamba run -n aifs-etl python cmorph_concatenated_icechunk.py \
        --start-year 1998 --end-year 1998 --n-workers 10

    # Append to existing store: process 1999 onwards
    micromamba run -n aifs-etl python cmorph_concatenated_icechunk.py \
        --start-year 1999 --end-year 2024 --n-workers 10 --append

    # Verify the store
    micromamba run -n aifs-etl python cmorph_concatenated_icechunk.py --verify

Author: AI Assistant
Date: 2026-01-23 (updated 2026-02-07 for Coiled parallel='dask')
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
from datetime import datetime
import re

import fsspec
import icechunk
import xarray as xr
import coiled
from dask.distributed import Client
from virtualizarr import open_virtual_dataset, open_virtual_mfdataset
from virtualizarr.parsers import HDFParser
from virtualizarr.registry import ObjectStoreRegistry
from obstore.store import from_url

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cmorph_concatenated_icechunk.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CMORPHConcatenatedIcechunkCreator:
    """
    Creates a properly concatenated Icechunk store from CMORPH S3 data.

    Uses DaskDelayedExecutor on a Coiled cluster to parallelize S3 metadata
    reads, then writes to Icechunk sequentially with append_dim='time'.
    """

    def __init__(
        self,
        gcs_bucket: str = "cpc_awc",
        gcs_prefix: str = "cmorph_concatenated_test",
        service_account_file: str = "coiled-data-e4drr_202505.json",
        n_workers: int = 10
    ):
        self.gcs_bucket = gcs_bucket
        self.gcs_prefix = gcs_prefix
        self.service_account_file = service_account_file
        self.n_workers = n_workers

        # S3 source configuration
        self.s3_bucket = "s3://noaa-cdr-precip-cmorph-pds/"
        self.s3_region = "us-east-1"

        # VirtualiZarr components
        self.s3_store = None
        self.registry = None
        self.parser = HDFParser()

        # Coiled/Dask components
        self.cluster = None
        self.client = None

        logger.info(f"Initialized CMORPH Concatenated Icechunk Creator")
        logger.info(f"  Output: gs://{gcs_bucket}/{gcs_prefix}")
        logger.info(f"  Source: {self.s3_bucket}")
        logger.info(f"  Workers: {n_workers}")

    def _setup_virtualizarr(self):
        """Setup VirtualiZarr S3 store and registry."""
        if self.s3_store is None:
            self.s3_store = from_url(
                self.s3_bucket,
                region=self.s3_region,
                skip_signature=True
            )
            self.registry = ObjectStoreRegistry({self.s3_bucket: self.s3_store})
            logger.info("VirtualiZarr S3 store and registry initialized")

    def _setup_coiled_cluster(self):
        """Create Coiled cluster and Dask client."""
        if self.client is not None:
            return

        logger.info(f"Starting Coiled cluster with {self.n_workers} workers...")
        cluster_start = time.time()

        self.cluster = coiled.Cluster(
            name=f"cmorph-concat-{int(time.time()) % 10000}",
            n_workers=self.n_workers,
            worker_vm_types="n2-standard-2",
            package_sync=True,
            region="us-east1",
            workspace="e4drr",
            idle_timeout="30 minutes",
        )
        self.client = Client(self.cluster)
        self.client.wait_for_workers(n_workers=self.n_workers, timeout=300)

        cluster_time = time.time() - cluster_start
        logger.info(f"Coiled cluster ready in {cluster_time:.1f}s")
        logger.info(f"  Dashboard: {self.client.dashboard_link}")

    def _shutdown_cluster(self):
        """Shutdown Coiled cluster."""
        if self.client is not None:
            logger.info("Shutting down Coiled cluster...")
            self.client.close()
            self.cluster.close()
            self.client = None
            self.cluster = None

    def _create_icechunk_config(self) -> icechunk.RepositoryConfig:
        """Create Icechunk configuration with Virtual Chunk Container."""
        config = icechunk.RepositoryConfig.default()
        container = icechunk.VirtualChunkContainer(
            self.s3_bucket,
            store=icechunk.s3_store(region=self.s3_region, anonymous=True),
        )
        config.set_virtual_chunk_container(container)
        return config

    def _get_gcs_storage(self):
        """Create GCS storage configuration."""
        return icechunk.gcs_storage(
            bucket=self.gcs_bucket,
            prefix=self.gcs_prefix,
            service_account_file=self.service_account_file,
        )

    def _open_or_create_repo(self, config: icechunk.RepositoryConfig) -> icechunk.Repository:
        """Open existing or create new Icechunk repository."""
        storage = self._get_gcs_storage()
        try:
            repo = icechunk.Repository.open(storage, config=config)
            logger.info("Opened existing Icechunk repository")
        except Exception as e:
            logger.info(f"Creating new repository ({e})")
            repo = icechunk.Repository.create(storage, config=config)
            logger.info("Created new Icechunk repository")
        return repo

    def get_s3_files_for_date_range(
        self,
        start_year: int,
        start_month: int,
        end_year: int,
        end_month: int
    ) -> List[str]:
        """Get all S3 files for a date range, sorted by timestamp."""
        logger.info(f"Getting S3 files for {start_year}-{start_month:02d} to {end_year}-{end_month:02d}")

        fs = fsspec.filesystem("s3", anon=True)
        all_urls = []

        current_year = start_year
        current_month = start_month

        while (current_year < end_year) or (current_year == end_year and current_month <= end_month):
            pattern = f"noaa-cdr-precip-cmorph-pds/data/30min/8km/{current_year}/{current_month:02d}/**/*.nc"
            try:
                files = fs.glob(pattern)
                urls = [f"s3://{f}" for f in files]
                all_urls.extend(urls)
                logger.info(f"  {current_year}-{current_month:02d}: {len(urls)} files")
            except Exception as e:
                logger.warning(f"  {current_year}-{current_month:02d}: Error - {e}")

            current_month += 1
            if current_month > 12:
                current_month = 1
                current_year += 1

        # Sort by timestamp from filename
        def extract_timestamp(url):
            filename = url.split('/')[-1]
            match = re.search(r'(\d{10})\.nc$', filename)
            if match:
                return match.group(1)
            return filename

        all_urls.sort(key=extract_timestamp)

        logger.info(f"Total: {len(all_urls)} files (sorted by timestamp)")
        return all_urls

    def get_files_by_month(self, all_urls: List[str]) -> Dict[str, List[str]]:
        """Group URLs by year-month for batch processing."""
        by_month = {}

        for url in all_urls:
            parts = url.split('/')
            try:
                for i, part in enumerate(parts):
                    if part == '8km' and i + 2 < len(parts):
                        year = parts[i + 1]
                        month = parts[i + 2]
                        key = f"{year}-{month}"
                        if key not in by_month:
                            by_month[key] = []
                        by_month[key].append(url)
                        break
            except Exception:
                logger.warning(f"Could not parse month from: {url}")

        sorted_months = sorted(by_month.keys())
        return {k: by_month[k] for k in sorted_months}

    def _get_parallel_mode(self):
        """Get the parallel mode based on whether Coiled cluster is available."""
        if self.client is not None:
            logger.info("  Using parallel='dask' (dask.delayed via Coiled)")
            return "dask"
        else:
            logger.info("  Using serial mode (no Coiled cluster)")
            return False

    def process_month_batch(
        self,
        urls: List[str],
        month_key: str,
        is_first_batch: bool,
        repo: icechunk.Repository
    ) -> Dict[str, Any]:
        """
        Process a single month's worth of files.

        Metadata reads are parallelized via DaskDelayedExecutor.
        Icechunk write is sequential (append_dim='time').
        """
        logger.info(f"Processing {month_key}: {len(urls)} files")
        start_time = time.time()

        self._setup_virtualizarr()

        result = {
            'month': month_key,
            'total_files': len(urls),
            'status': 'pending',
            'error': None
        }

        try:
            # Parallel metadata reads via dask.delayed on Coiled
            parallel_mode = self._get_parallel_mode()
            vds_start = time.time()
            logger.info(f"  Opening virtual multi-file dataset with {len(urls)} files...")

            combined_vds = open_virtual_mfdataset(
                urls,
                registry=self.registry,
                parser=self.parser,
                combine="nested",
                concat_dim="time",
                parallel=parallel_mode
            )

            vds_time = time.time() - vds_start
            logger.info(f"  Virtual dataset ready in {vds_time:.1f}s - dims: {dict(combined_vds.dims)}")

            # Sequential Icechunk write (~4 sec)
            write_start = time.time()
            session = repo.writable_session("main")

            if is_first_batch:
                logger.info("  Writing first batch (no append_dim)...")
                combined_vds.virtualize.to_icechunk(session.store, group='cmorph')
            else:
                logger.info("  Appending with append_dim='time'...")
                combined_vds.virtualize.to_icechunk(
                    session.store,
                    group='cmorph',
                    append_dim='time'
                )

            commit_msg = f"Added {month_key}: {len(urls)} files"
            snapshot_id = session.commit(message=commit_msg)
            write_time = time.time() - write_start

            result['status'] = 'success'
            result['snapshot_id'] = str(snapshot_id)
            result['time_dim_size'] = combined_vds.dims.get('time', 0)
            result['vds_time_sec'] = vds_time
            result['write_time_sec'] = write_time

            logger.info(f"  Committed: {commit_msg} (vds: {vds_time:.1f}s, write: {write_time:.1f}s)")

        except Exception as e:
            import traceback
            result['status'] = 'error'
            result['error'] = str(e)
            result['traceback'] = traceback.format_exc()
            logger.error(f"  Error processing {month_key}: {e}")
            logger.error(traceback.format_exc())

        result['processing_time_sec'] = time.time() - start_time
        return result

    def create_concatenated_store(
        self,
        start_year: int,
        start_month: int,
        end_year: int,
        end_month: int,
        max_months: Optional[int] = None,
        append: bool = False
    ) -> Dict[str, Any]:
        """
        Create a concatenated Icechunk store for a date range.

        Args:
            start_year: Starting year
            start_month: Starting month (1-12)
            end_year: Ending year
            end_month: Ending month (1-12)
            max_months: Optional limit on number of months to process
            append: If True, always use append_dim (store already has data)
        """
        logger.info("=" * 70)
        logger.info("Creating Concatenated CMORPH Icechunk Store (Coiled-accelerated)")
        logger.info("=" * 70)
        logger.info(f"  Date range: {start_year}-{start_month:02d} to {end_year}-{end_month:02d}")
        logger.info(f"  Output: gs://{self.gcs_bucket}/{self.gcs_prefix}")
        logger.info(f"  Append mode: {append}")

        overall_start = time.time()

        results = {
            'start_date': f"{start_year}-{start_month:02d}",
            'end_date': f"{end_year}-{end_month:02d}",
            'output_location': f"gs://{self.gcs_bucket}/{self.gcs_prefix}",
            'start_time': datetime.now().isoformat(),
            'append_mode': append,
            'n_workers': self.n_workers,
            'months_processed': [],
            'total_files': 0,
            'successful_files': 0,
            'failed_months': []
        }

        # Get all files and group by month
        all_urls = self.get_s3_files_for_date_range(
            start_year, start_month, end_year, end_month
        )

        if not all_urls:
            logger.error("No files found!")
            results['status'] = 'error'
            results['error'] = 'No files found'
            return results

        files_by_month = self.get_files_by_month(all_urls)
        months_to_process = list(files_by_month.keys())

        if max_months is not None:
            months_to_process = months_to_process[:max_months]
            logger.info(f"Limiting to first {max_months} months")

        total_files = sum(len(files_by_month[m]) for m in months_to_process)
        logger.info(f"\nTotal months to process: {len(months_to_process)}")
        logger.info(f"Total files: {total_files}")

        # Start Coiled cluster
        try:
            self._setup_coiled_cluster()
        except Exception as e:
            logger.error(f"Failed to start Coiled cluster: {e}")
            logger.info("Falling back to serial execution")

        # Create Icechunk repository
        config = self._create_icechunk_config()
        repo = self._open_or_create_repo(config)

        # Process each month sequentially
        # (parallel metadata reads inside each month via DaskDelayedExecutor)
        try:
            for i, month_key in enumerate(months_to_process):
                urls = files_by_month[month_key]
                # First batch only when not appending and it's the first month
                is_first = (i == 0) and (not append)

                month_result = self.process_month_batch(
                    urls=urls,
                    month_key=month_key,
                    is_first_batch=is_first,
                    repo=repo
                )

                results['months_processed'].append(month_result)
                results['total_files'] += month_result['total_files']

                if month_result['status'] == 'success':
                    results['successful_files'] += month_result['total_files']
                else:
                    results['failed_months'].append(month_key)

                elapsed = time.time() - overall_start
                rate = results['successful_files'] / elapsed * 60 if elapsed > 0 else 0
                logger.info(
                    f"Progress: {i+1}/{len(months_to_process)} months "
                    f"({results['successful_files']}/{total_files} files, "
                    f"{rate:.0f} files/min, {elapsed/60:.1f} min elapsed)"
                )
        finally:
            self._shutdown_cluster()

        # Final summary
        results['end_time'] = datetime.now().isoformat()
        results['total_processing_time_sec'] = time.time() - overall_start
        results['total_processing_time_min'] = results['total_processing_time_sec'] / 60
        results['success_rate'] = (
            results['successful_files'] / results['total_files'] * 100
            if results['total_files'] > 0 else 0
        )

        logger.info("\n" + "=" * 70)
        logger.info("PROCESSING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"  Months processed: {len(results['months_processed'])}")
        logger.info(f"  Total files: {results['total_files']}")
        logger.info(f"  Successful: {results['successful_files']}")
        logger.info(f"  Failed months: {len(results['failed_months'])}")
        logger.info(f"  Success rate: {results['success_rate']:.1f}%")
        logger.info(f"  Total time: {results['total_processing_time_min']:.1f} minutes")
        logger.info(f"  Output: {results['output_location']}")

        # Save results
        results_file = Path(f"cmorph_concatenated_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"  Results saved: {results_file}")

        return results


def verify_virtual_store(
    gcs_bucket="cpc_awc",
    gcs_prefix="cmorph_1998_2024_virtual",
    service_account_file="coiled-data-e4drr_202505.json"
):
    """Verify the virtual Icechunk store is queryable."""
    import numpy as np

    logger.info("=" * 70)
    logger.info(f"VERIFYING gs://{gcs_bucket}/{gcs_prefix}")
    logger.info("=" * 70)

    s3_bucket = "s3://noaa-cdr-precip-cmorph-pds/"

    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(
            s3_bucket,
            store=icechunk.s3_store(region="us-east-1", anonymous=True),
        )
    )

    s3_creds = icechunk.containers_credentials({
        s3_bucket: icechunk.s3_credentials(anonymous=True)
    })

    storage = icechunk.gcs_storage(
        bucket=gcs_bucket,
        prefix=gcs_prefix,
        service_account_file=service_account_file,
    )

    repo = icechunk.Repository.open(
        storage, config=config,
        authorize_virtual_chunk_access=s3_creds
    )

    session = repo.readonly_session(branch="main")
    ds = xr.open_zarr(session.store, group='cmorph', consolidated=False)

    logger.info(f"Dimensions: {dict(ds.sizes)}")
    logger.info(f"Variables: {list(ds.data_vars)}")
    logger.info(f"Time: {ds.time.values[0]} to {ds.time.values[-1]} ({ds.sizes['time']} steps)")

    # Single timestep
    t0 = time.time()
    field = ds['cmorph'].isel(time=0, lat=slice(0, 10), lon=slice(0, 10)).load()
    logger.info(f"Single timestep load: {time.time()-t0:.2f}s, shape={field.shape}")

    # Time slice (48 steps = 1 day)
    t0 = time.time()
    mid = ds.sizes['time'] // 2
    day = ds['cmorph'].isel(time=slice(mid, mid + 48), lat=slice(0, 10), lon=slice(0, 10)).load()
    logger.info(f"1-day time slice: {time.time()-t0:.2f}s, shape={day.shape}")
    logger.info(f"  Range: {day.time.values[0]} -> {day.time.values[-1]}")

    logger.info("VERIFICATION PASSED")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Create concatenated CMORPH Icechunk store (Coiled-accelerated)'
    )
    parser.add_argument('--start-year', type=int, default=2024,
                        help='Start year (default: 2024)')
    parser.add_argument('--start-month', type=int, default=1,
                        help='Start month (default: 1)')
    parser.add_argument('--end-year', type=int, default=2024,
                        help='End year (default: 2024)')
    parser.add_argument('--end-month', type=int, default=12,
                        help='End month (default: 12)')
    parser.add_argument('--max-months', type=int, default=None,
                        help='Max months to process (default: all)')
    parser.add_argument('--gcs-bucket', type=str, default='cpc_awc',
                        help='GCS bucket (default: cpc_awc)')
    parser.add_argument('--gcs-prefix', type=str, default='cmorph_1998_2024_virtual',
                        help='GCS prefix (default: cmorph_1998_2024_virtual)')
    parser.add_argument('--service-account', type=str,
                        default='coiled-data-e4drr_202505.json',
                        help='Service account JSON file')
    parser.add_argument('--n-workers', type=int, default=10,
                        help='Number of Coiled workers (default: 10)')
    parser.add_argument('--append', action='store_true',
                        help='Append to existing store (use when store already has data)')
    parser.add_argument('--verify', action='store_true',
                        help='Verify existing store instead of processing')

    args = parser.parse_args()

    if args.verify:
        verify_virtual_store(
            gcs_bucket=args.gcs_bucket,
            gcs_prefix=args.gcs_prefix,
            service_account_file=args.service_account
        )
    else:
        creator = CMORPHConcatenatedIcechunkCreator(
            gcs_bucket=args.gcs_bucket,
            gcs_prefix=args.gcs_prefix,
            service_account_file=args.service_account,
            n_workers=args.n_workers
        )

        results = creator.create_concatenated_store(
            start_year=args.start_year,
            start_month=args.start_month,
            end_year=args.end_year,
            end_month=args.end_month,
            max_months=args.max_months,
            append=args.append
        )


if __name__ == "__main__":
    main()
