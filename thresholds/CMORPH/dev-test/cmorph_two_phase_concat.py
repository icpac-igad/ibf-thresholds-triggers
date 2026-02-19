#!/usr/bin/env python3
"""
CMORPH Two-Phase Concatenation (OOM-free)
==========================================

Phase 1: Create monthly virtual datasets in parallel on Coiled workers.
          Each worker reads S3 file headers and returns a lightweight VDS (~70 KB).

Phase 2: Collect all monthly VDS locally, xr.concat into one dataset,
          then write to Icechunk in a SINGLE to_icechunk() call.
          No append_dim, no growing metadata, no OOM.

See CMORPH_TWO_PHASE_RATIONALE.md for why this approach avoids the OOM
that killed all previous append_dim-based attempts.

Usage:
    # Full 1998-2024 (10 Coiled workers)
    micromamba run -n aifs-etl python cmorph_two_phase_concat.py \
        --start-year 1998 --end-year 2024 --n-workers 10

    # Test with 3 months
    micromamba run -n aifs-etl python cmorph_two_phase_concat.py \
        --start-year 1998 --end-year 1998 --n-workers 5 --max-months 3

    # Custom output prefix
    micromamba run -n aifs-etl python cmorph_two_phase_concat.py \
        --start-year 1998 --end-year 2024 --n-workers 10 \
        --gcs-prefix cmorph_1998_2024_final

Author: AI Assistant
Date: 2026-02-07
"""

import json
import logging
import time
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import fsspec

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cmorph_two_phase_concat.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

S3_BUCKET = "s3://noaa-cdr-precip-cmorph-pds/"
S3_REGION = "us-east-1"
SERVICE_ACCOUNT_FILE = "coiled-data-e4drr_202505.json"


# ─── Phase 1: Worker function (runs on Coiled) ───────────────────────────────

def create_month_vds(args: Tuple) -> Dict[str, Any]:
    """
    Coiled worker function: create a virtual dataset for one month.

    Reads S3 file headers (metadata only, no data download).
    Returns the lightweight VDS object (~70 KB) back to the coordinator.
    """
    month_key, urls, s3_bucket, s3_region = args
    import time as _time

    try:
        start = _time.time()

        from virtualizarr import open_virtual_mfdataset
        from virtualizarr.parsers import HDFParser
        from virtualizarr.registry import ObjectStoreRegistry
        from obstore.store import from_url

        s3_store = from_url(s3_bucket, region=s3_region, skip_signature=True)
        registry = ObjectStoreRegistry({s3_bucket: s3_store})

        vds = open_virtual_mfdataset(
            urls,
            registry=registry,
            parser=HDFParser(),
            combine="nested",
            concat_dim="time"
        )

        elapsed = _time.time() - start
        n_time = vds.sizes.get('time', 0)

        return {
            'status': 'success',
            'month_key': month_key,
            'n_files': len(urls),
            'n_timesteps': n_time,
            'elapsed_sec': elapsed,
            'vds': vds
        }

    except Exception as e:
        import traceback
        return {
            'status': 'error',
            'month_key': month_key,
            'n_files': len(urls),
            'error': str(e),
            'traceback': traceback.format_exc()
        }


# ─── File discovery ──────────────────────────────────────────────────────────

def get_all_files_by_month(
    start_year: int, start_month: int,
    end_year: int, end_month: int,
    max_months: Optional[int] = None
) -> Dict[str, List[str]]:
    """Get S3 files grouped by month, sorted by timestamp."""
    logger.info(f"Listing S3 files for {start_year}-{start_month:02d} to {end_year}-{end_month:02d}")
    fs = fsspec.filesystem("s3", anon=True)
    by_month = {}

    y, m = start_year, start_month
    while (y < end_year) or (y == end_year and m <= end_month):
        key = f"{y}-{m:02d}"
        pattern = f"noaa-cdr-precip-cmorph-pds/data/30min/8km/{y}/{m:02d}/**/*.nc"
        try:
            files = fs.glob(pattern)
            urls = [f"s3://{f}" for f in files]

            # Sort by timestamp in filename
            def ts(url):
                match = re.search(r'(\d{10})\.nc$', url)
                return match.group(1) if match else url
            urls.sort(key=ts)

            by_month[key] = urls
            logger.info(f"  {key}: {len(urls)} files")
        except Exception as e:
            logger.warning(f"  {key}: Error - {e}")

        m += 1
        if m > 12:
            m = 1
            y += 1

    sorted_keys = sorted(by_month.keys())
    if max_months:
        sorted_keys = sorted_keys[:max_months]

    total = sum(len(by_month[k]) for k in sorted_keys)
    logger.info(f"Total: {len(sorted_keys)} months, {total} files")
    return {k: by_month[k] for k in sorted_keys}


# ─── Main pipeline ───────────────────────────────────────────────────────────

def run_two_phase(
    start_year: int,
    end_year: int,
    start_month: int = 1,
    end_month: int = 12,
    n_workers: int = 10,
    gcs_bucket: str = "cpc_awc",
    gcs_prefix: str = "cmorph_1998_2024_twophase",
    max_months: Optional[int] = None,
    service_account: str = SERVICE_ACCOUNT_FILE,
) -> Dict[str, Any]:
    """
    Two-phase concatenation pipeline.

    Phase 1: Parallel VDS creation on Coiled (returns lightweight metadata).
    Phase 2: Local xr.concat + single to_icechunk write (no append, no OOM).
    """
    import coiled
    from dask.distributed import Client, as_completed

    logger.info("=" * 70)
    logger.info("CMORPH Two-Phase Concatenation")
    logger.info("=" * 70)
    logger.info(f"  Range: {start_year}-{start_month:02d} to {end_year}-{end_month:02d}")
    logger.info(f"  Output: gs://{gcs_bucket}/{gcs_prefix}")
    logger.info(f"  Workers: {n_workers}")
    overall_start = time.time()

    results = {
        'start_time': datetime.now().isoformat(),
        'range': f"{start_year}-{end_year}",
        'output': f"gs://{gcs_bucket}/{gcs_prefix}",
        'n_workers': n_workers,
    }

    # ── File discovery ──
    files_by_month = get_all_files_by_month(
        start_year, start_month, end_year, end_month, max_months
    )
    months = list(files_by_month.keys())
    total_files = sum(len(v) for v in files_by_month.values())
    results['n_months'] = len(months)
    results['n_files'] = total_files

    if not months:
        logger.error("No files found!")
        results['status'] = 'error'
        return results

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 1: Parallel VDS creation on Coiled
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1: Creating monthly virtual datasets on Coiled")
    logger.info("=" * 70)

    phase1_start = time.time()

    logger.info(f"Starting Coiled cluster with {n_workers} workers...")
    cluster = coiled.Cluster(
        name=f"cmorph-2phase-{int(time.time()) % 10000}",
        n_workers=n_workers,
        worker_vm_types="n2-standard-2",
        package_sync=True,
        region="us-east1",
        workspace="e4drr",
        idle_timeout="30 minutes",
    )
    client = Client(cluster)
    client.wait_for_workers(n_workers=n_workers, timeout=300)
    logger.info(f"Cluster ready: {client.dashboard_link}")

    # Submit all months in parallel
    task_args = [
        (month_key, files_by_month[month_key], S3_BUCKET, S3_REGION)
        for month_key in months
    ]

    logger.info(f"Submitting {len(task_args)} monthly tasks to {n_workers} workers...")
    futures = client.map(create_month_vds, task_args)

    # Collect results as they complete
    monthly_vds = {}  # month_key -> vds (in order)
    failed_months = []
    completed = 0

    for future in as_completed(futures):
        completed += 1
        try:
            result = future.result()
            mk = result['month_key']

            if result['status'] == 'success':
                monthly_vds[mk] = result['vds']
                logger.info(
                    f"  [{completed}/{len(months)}] {mk}: "
                    f"{result['n_timesteps']} timesteps in {result['elapsed_sec']:.1f}s"
                )
            else:
                failed_months.append(mk)
                logger.error(f"  [{completed}/{len(months)}] {mk}: FAILED - {result['error']}")

        except Exception as e:
            completed_key = f"unknown_{completed}"
            failed_months.append(completed_key)
            logger.error(f"  [{completed}/{len(months)}] Future error: {e}")

    # Shutdown cluster — Phase 1 done
    logger.info("Shutting down Coiled cluster...")
    client.close()
    cluster.close()

    phase1_time = time.time() - phase1_start
    logger.info(f"\nPhase 1 complete: {len(monthly_vds)}/{len(months)} months in {phase1_time/60:.1f} min")
    logger.info(f"  Failed: {len(failed_months)}")

    results['phase1'] = {
        'successful_months': len(monthly_vds),
        'failed_months': failed_months,
        'time_sec': phase1_time,
        'time_min': phase1_time / 60,
    }

    if not monthly_vds:
        logger.error("No virtual datasets created! Aborting.")
        results['status'] = 'error'
        return results

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 2: Local concat + single Icechunk write
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: Local concat + single Icechunk write")
    logger.info("=" * 70)

    import xarray as xr
    import icechunk

    phase2_start = time.time()

    # Sort by month key to ensure chronological order
    sorted_keys = sorted(monthly_vds.keys())
    sorted_vds = [monthly_vds[k] for k in sorted_keys]

    logger.info(f"Concatenating {len(sorted_vds)} monthly virtual datasets...")
    concat_start = time.time()
    full_vds = xr.concat(sorted_vds, dim="time")
    concat_time = time.time() - concat_start
    logger.info(f"  Concat done in {concat_time:.1f}s")
    logger.info(f"  Full dataset dims: {dict(full_vds.sizes)}")
    logger.info(f"  Time range: {sorted_keys[0]} to {sorted_keys[-1]}")

    # Write to Icechunk — SINGLE write, no append_dim
    logger.info(f"\nWriting to Icechunk: gs://{gcs_bucket}/{gcs_prefix}")

    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(
            S3_BUCKET,
            store=icechunk.s3_store(region=S3_REGION, anonymous=True),
        )
    )

    storage = icechunk.gcs_storage(
        bucket=gcs_bucket,
        prefix=gcs_prefix,
        service_account_file=service_account,
    )

    try:
        repo = icechunk.Repository.open(storage, config=config)
        logger.info("  Opened existing repository")
    except Exception:
        repo = icechunk.Repository.create(storage, config=config)
        logger.info("  Created new repository")

    session = repo.writable_session("main")

    write_start = time.time()
    logger.info("  Writing full dataset (single to_icechunk call, no append)...")
    full_vds.virtualize.to_icechunk(session.store, group='cmorph')
    write_time = time.time() - write_start
    logger.info(f"  Write done in {write_time:.1f}s")

    commit_start = time.time()
    snapshot_id = session.commit(
        message=f"Full {sorted_keys[0]} to {sorted_keys[-1]}: "
                f"{len(sorted_vds)} months, {total_files} files"
    )
    commit_time = time.time() - commit_start
    logger.info(f"  Committed in {commit_time:.1f}s: {snapshot_id}")

    phase2_time = time.time() - phase2_start

    results['phase2'] = {
        'n_months_concat': len(sorted_vds),
        'dims': dict(full_vds.sizes),
        'concat_time_sec': concat_time,
        'write_time_sec': write_time,
        'commit_time_sec': commit_time,
        'total_time_sec': phase2_time,
        'total_time_min': phase2_time / 60,
        'snapshot_id': str(snapshot_id),
    }

    # ── Verify ──
    logger.info("\n" + "=" * 70)
    logger.info("VERIFICATION")
    logger.info("=" * 70)

    s3_creds = icechunk.containers_credentials({
        S3_BUCKET: icechunk.s3_credentials(anonymous=True)
    })
    repo_verify = icechunk.Repository.open(
        storage, config=config,
        authorize_virtual_chunk_access=s3_creds
    )
    verify_session = repo_verify.readonly_session(branch="main")
    ds = xr.open_zarr(verify_session.store, group='cmorph', consolidated=False)

    logger.info(f"  Dims: {dict(ds.sizes)}")
    logger.info(f"  Time: {ds.time.values[0]} -> {ds.time.values[-1]}")

    # Quick time-slice test
    t0 = time.time()
    mid = ds.sizes['time'] // 2
    sample = ds['cmorph'].isel(time=slice(mid, mid + 10), lat=slice(0, 5), lon=slice(0, 5)).load()
    logger.info(f"  Time slice test: shape={sample.shape}, time={time.time()-t0:.2f}s - PASS")

    # ── Summary ──
    total_time = time.time() - overall_start
    results['total_time_sec'] = total_time
    results['total_time_min'] = total_time / 60
    results['status'] = 'success'
    results['end_time'] = datetime.now().isoformat()

    logger.info("\n" + "=" * 70)
    logger.info("COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Phase 1 (Coiled VDS creation): {phase1_time/60:.1f} min")
    logger.info(f"  Phase 2 (concat + write):      {phase2_time/60:.1f} min")
    logger.info(f"  Total:                         {total_time/60:.1f} min")
    logger.info(f"  Output: gs://{gcs_bucket}/{gcs_prefix}")
    logger.info(f"  Snapshot: {snapshot_id}")

    results_file = Path(f"cmorph_twophase_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"  Results: {results_file}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='CMORPH Two-Phase Concatenation (OOM-free)'
    )
    parser.add_argument('--start-year', type=int, default=1998)
    parser.add_argument('--end-year', type=int, default=2024)
    parser.add_argument('--start-month', type=int, default=1)
    parser.add_argument('--end-month', type=int, default=12)
    parser.add_argument('--n-workers', type=int, default=10)
    parser.add_argument('--gcs-bucket', type=str, default='cpc_awc')
    parser.add_argument('--gcs-prefix', type=str, default='cmorph_1998_2024_twophase')
    parser.add_argument('--service-account', type=str, default=SERVICE_ACCOUNT_FILE)
    parser.add_argument('--max-months', type=int, default=None,
                        help='Limit months for testing')

    args = parser.parse_args()

    run_two_phase(
        service_account=args.service_account,
        start_year=args.start_year,
        end_year=args.end_year,
        start_month=args.start_month,
        end_month=args.end_month,
        n_workers=args.n_workers,
        gcs_bucket=args.gcs_bucket,
        gcs_prefix=args.gcs_prefix,
        max_months=args.max_months,
    )


if __name__ == "__main__":
    main()
