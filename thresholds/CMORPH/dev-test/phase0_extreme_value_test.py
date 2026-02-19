#!/usr/bin/env python3
"""
Phase 0: Extreme Value Analysis Test Run
=========================================
Test the return period calculation pipeline with:
- 153 days of CMORPH data (current Icechunk store)
- 4 Coiled workers
- 20x20 pixel subset from East Africa (~400 pixels)
- Output to non-virtual Icechunk store (actual computed data)

This script validates the methodology before scaling to full dataset.

Pipeline Stages:
- Stage 1 (Coiled workers): Load data -> Rolling sums -> Extract pseudo-annual maxima
- Stage 2 (Local): Fit normal distribution -> Calculate return periods
- Stage 3 (Local): Build xarray Dataset with metadata
- Stage 4 (Local): Write to Icechunk (actual data, not virtual references)

Author: AI Assistant
Date: 2026-01-23
"""

import os
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    # Icechunk source (input)
    'gcs_bucket': 'cpc_awc',
    'gcs_prefix': 'cmorph_20260123',
    's3_bucket': 's3://noaa-cdr-precip-cmorph-pds/',
    'service_account_file': '/home/roller/Documents/08-2023/working_notes_jupyter/ignore_nka_gitrepos/pam_team/deploy-itt/arco_fetch/CMORPH/coiled-data-e4drr_202505.json',

    # Icechunk output
    'output_gcs_bucket': 'cpc_awc',
    'output_gcs_prefix': 'cmorph_return_periods_test_phase0',

    # Spatial subset (East Africa test area)
    # Full East Africa: lat [23, -12], lon [21, 53]
    # Test subset: 20x20 pixels for initial testing (reduced from 50x50)
    'spatial': {
        'lat_start_idx': 200,  # Starting latitude index
        'lat_count': 20,       # Number of latitude pixels (reduced)
        'lon_start_idx': 300,  # Starting longitude index
        'lon_count': 20,       # Number of longitude pixels (reduced)
    },

    # Accumulation durations (in 30-minute timesteps)
    'durations': {
        '30min': 1,
        '1hr': 2,
        '3hr': 6,
        '6hr': 12,
        '12hr': 24,
        '24hr': 48,
        '48hr': 96,
        '72hr': 144,
        '7day': 336
    },

    # Return periods to calculate (years)
    'return_periods': [2, 5, 10, 20, 50],

    # Pseudo-year configuration
    # With 153 days of data, we create pseudo-years
    # Split into 5 periods of ~30 days each
    'pseudo_year_days': 30,

    # Batch processing (small batches for testing)
    'batch_size_lat': 5,
    'batch_size_lon': 5,

    # Coiled cluster configuration
    'cluster': {
        'n_workers': 4,
        'worker_vm_types': 'n2-standard-4',
        'region': 'us-east1',
        'workspace': 'e4drr',
        'idle_timeout': '15 minutes',
    }
}


def compute_annual_maxima_batch(args) -> Dict[str, Any]:
    """
    Worker function to compute pseudo-annual maxima for a spatial batch.

    This runs on Coiled workers and:
    1. Loads time series in chunks (pseudo-year by pseudo-year) to avoid memory issues
    2. Calculates rolling sums for each accumulation period
    3. Extracts maximum per pseudo-year (30-day periods)

    Args:
        args: Tuple of (batch_id, lat_slice, lon_slice, creds_json, config)

    Returns:
        Dictionary with annual maxima results
    """
    import os
    import tempfile
    import time
    import numpy as np

    batch_id, lat_slice, lon_slice, creds_json, config = args

    # Extract config values
    gcs_bucket = config['gcs_bucket']
    gcs_prefix = config['gcs_prefix']
    s3_bucket = config['s3_bucket']
    durations = config['durations']
    pseudo_year_days = config['pseudo_year_days']

    timing = {'start': time.time()}
    creds_file = None

    try:
        # Write credentials to temp file
        creds_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        creds_file.write(creds_json)
        creds_file.close()

        # Import inside worker
        import icechunk
        import xarray as xr

        # Open Icechunk store
        timing['icechunk_start'] = time.time()
        storage = icechunk.gcs_storage(
            bucket=gcs_bucket,
            prefix=gcs_prefix,
            service_account_file=creds_file.name
        )

        s3_creds = icechunk.containers_credentials({
            s3_bucket: icechunk.s3_credentials(anonymous=True)
        })

        repo = icechunk.Repository.open(
            storage=storage,
            authorize_virtual_chunk_access=s3_creds
        )

        session = repo.readonly_session(branch="main")
        ds = xr.open_zarr(session.store, group='cmorph', consolidated=False)
        timing['icechunk_open'] = time.time() - timing['icechunk_start']

        # Get total time dimension and calculate pseudo-years
        n_time_total = ds.dims['time']
        timesteps_per_day = 48
        timesteps_per_pseudo_year = pseudo_year_days * timesteps_per_day
        n_pseudo_years = n_time_total // timesteps_per_pseudo_year

        # Get spatial subset dimensions
        lat_indices = list(range(lat_slice.start, lat_slice.stop))
        lon_indices = list(range(lon_slice.start, lon_slice.stop))
        n_lat = len(lat_indices)
        n_lon = len(lon_indices)

        # Initialize results - we'll accumulate maxima per pseudo-year
        annual_maxima = {dur_name: np.zeros((n_pseudo_years, n_lat, n_lon), dtype=np.float32)
                        for dur_name in durations.keys()}

        timing['load_start'] = time.time()
        total_data_loaded = 0

        # Process each pseudo-year separately to reduce memory
        for py in range(n_pseudo_years):
            start_time_idx = py * timesteps_per_pseudo_year
            end_time_idx = start_time_idx + timesteps_per_pseudo_year

            # Load this pseudo-year's data for the spatial batch
            subset = ds['cmorph'].isel(
                time=slice(start_time_idx, end_time_idx),
                lat=lat_slice,
                lon=lon_slice
            ).load()

            data = subset.values.astype(np.float32)
            total_data_loaded += data.nbytes

            # Store lat/lon values from first iteration
            if py == 0:
                lat_values = subset.lat.values.tolist()
                lon_values = subset.lon.values.tolist()

            # Process each duration
            for dur_name, window in durations.items():
                if window == 1:
                    rolled = data
                else:
                    # Rolling sum using cumsum
                    # Handle NaN by replacing with 0 for cumsum, then restore
                    data_clean = np.where(np.isnan(data), 0, data)
                    cumsum = np.cumsum(data_clean, axis=0)
                    rolled = np.zeros_like(data)
                    rolled[:window-1] = np.nan
                    rolled[window-1:] = cumsum[window-1:] - np.concatenate([
                        np.zeros((1, n_lat, n_lon)),
                        cumsum[:-window]
                    ], axis=0)[window-1:]

                # Get max for this pseudo-year
                with np.errstate(invalid='ignore'):
                    annual_maxima[dur_name][py] = np.nanmax(rolled, axis=0)

            # Free memory
            del subset, data

        timing['load_time'] = time.time() - timing['load_start']
        timing['compute_time'] = 0  # Compute is interleaved with load

        return {
            'status': 'success',
            'batch_id': batch_id,
            'lat_slice': (lat_slice.start, lat_slice.stop),
            'lon_slice': (lon_slice.start, lon_slice.stop),
            'lat_values': lat_values,
            'lon_values': lon_values,
            'n_pseudo_years': n_pseudo_years,
            'shape': (n_pseudo_years, n_lat, n_lon),
            'annual_maxima': {k: v.tolist() for k, v in annual_maxima.items()},
            'timing': {
                'icechunk_open': timing['icechunk_open'],
                'load_time': timing['load_time'],
                'compute_time': timing['compute_time'],
                'total': time.time() - timing['start']
            },
            'data_loaded_mb': total_data_loaded / (1024 * 1024),
        }

    except Exception as e:
        import traceback
        return {
            'status': 'error',
            'batch_id': batch_id,
            'lat_slice': (lat_slice.start, lat_slice.stop),
            'lon_slice': (lon_slice.start, lon_slice.stop),
            'error': str(e),
            'traceback': traceback.format_exc()
        }
    finally:
        if creds_file is not None:
            try:
                os.unlink(creds_file.name)
            except:
                pass


def fit_distribution_and_return_periods(
    annual_maxima: np.ndarray,
    return_periods: List[int],
    method: str = 'normal'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit distribution and compute return periods for a pixel.

    Args:
        annual_maxima: Array of shape (n_years,) with annual maximum values
        return_periods: List of return periods to calculate (e.g., [2, 5, 10, 20, 50])
        method: 'normal' or 'gumbel'

    Returns:
        Tuple of (return_period_values, location, scale)
    """
    from scipy import stats

    # Remove NaN values
    valid = annual_maxima[~np.isnan(annual_maxima)]

    if len(valid) < 3:
        # Insufficient data
        return (
            np.full(len(return_periods), np.nan),
            np.nan,
            np.nan
        )

    if method == 'normal':
        mu = np.mean(valid)
        sigma = np.std(valid, ddof=1)

        rp_values = np.zeros(len(return_periods))
        for i, T in enumerate(return_periods):
            p = 1 - 1/T
            z = stats.norm.ppf(p)
            rp_values[i] = mu + z * sigma

        return rp_values, mu, sigma

    elif method == 'gumbel':
        loc, scale = stats.gumbel_r.fit(valid)

        rp_values = np.zeros(len(return_periods))
        for i, T in enumerate(return_periods):
            y_T = -np.log(-np.log(1 - 1/T))
            rp_values[i] = loc + scale * y_T

        return rp_values, loc, scale

    else:
        raise ValueError(f"Unknown method: {method}")


def process_stage2_local(
    batch_results: List[Dict],
    config: Dict
) -> Dict[str, np.ndarray]:
    """
    Stage 2: Fit distributions and calculate return periods locally.

    Args:
        batch_results: List of results from Stage 1 workers
        config: Configuration dictionary

    Returns:
        Dictionary with output arrays
    """
    logger.info("Stage 2: Fitting distributions and calculating return periods...")

    durations = list(config['durations'].keys())
    return_periods = config['return_periods']
    n_durations = len(durations)
    n_return_periods = len(return_periods)

    # Get dimensions from first successful result
    successful = [r for r in batch_results if r['status'] == 'success']
    if not successful:
        raise ValueError("No successful batch results!")

    # Collect all unique lat/lon values
    all_lats = set()
    all_lons = set()
    for r in successful:
        all_lats.update(r['lat_values'])
        all_lons.update(r['lon_values'])

    all_lats = sorted(all_lats, reverse=True)  # Lat typically decreasing
    all_lons = sorted(all_lons)
    n_lat = len(all_lats)
    n_lon = len(all_lons)

    lat_to_idx = {lat: i for i, lat in enumerate(all_lats)}
    lon_to_idx = {lon: i for i, lon in enumerate(all_lons)}

    # Initialize output arrays
    return_period_precip = np.full((n_durations, n_return_periods, n_lat, n_lon), np.nan, dtype=np.float32)
    dist_location = np.full((n_durations, n_lat, n_lon), np.nan, dtype=np.float32)
    dist_scale = np.full((n_durations, n_lat, n_lon), np.nan, dtype=np.float32)

    # Process each batch result
    for batch in successful:
        batch_lats = batch['lat_values']
        batch_lons = batch['lon_values']
        annual_maxima = batch['annual_maxima']

        for d_idx, dur_name in enumerate(durations):
            am_data = np.array(annual_maxima[dur_name])  # Shape: (n_years, n_lat, n_lon)

            for i, lat in enumerate(batch_lats):
                for j, lon in enumerate(batch_lons):
                    lat_idx = lat_to_idx[lat]
                    lon_idx = lon_to_idx[lon]

                    # Get time series for this pixel
                    pixel_am = am_data[:, i, j]

                    # Fit distribution
                    rp_values, loc, scale = fit_distribution_and_return_periods(
                        pixel_am,
                        return_periods,
                        method='normal'
                    )

                    # Store results
                    return_period_precip[d_idx, :, lat_idx, lon_idx] = rp_values
                    dist_location[d_idx, lat_idx, lon_idx] = loc
                    dist_scale[d_idx, lat_idx, lon_idx] = scale

    return {
        'return_period_precip': return_period_precip,
        'dist_location': dist_location,
        'dist_scale': dist_scale,
        'lats': np.array(all_lats),
        'lons': np.array(all_lons),
        'durations': durations,
        'return_periods': return_periods
    }


def create_output_dataset(results: Dict, config: Dict):
    """
    Stage 3: Build xarray Dataset with metadata.

    Args:
        results: Dictionary from Stage 2
        config: Configuration dictionary

    Returns:
        xarray.Dataset
    """
    import xarray as xr

    logger.info("Stage 3: Creating output xarray Dataset...")

    ds = xr.Dataset(
        data_vars={
            'return_period_precip': (
                ['duration', 'return_period', 'lat', 'lon'],
                results['return_period_precip'],
                {
                    'long_name': 'Precipitation depth for return period',
                    'units': 'mm',
                    'description': 'Precipitation accumulation expected to be exceeded once per return period'
                }
            ),
            'dist_location': (
                ['duration', 'lat', 'lon'],
                results['dist_location'],
                {
                    'long_name': 'Distribution location parameter (mu)',
                    'units': 'mm',
                    'description': 'Mean of annual maxima distribution'
                }
            ),
            'dist_scale': (
                ['duration', 'lat', 'lon'],
                results['dist_scale'],
                {
                    'long_name': 'Distribution scale parameter (sigma)',
                    'units': 'mm',
                    'description': 'Standard deviation of annual maxima distribution'
                }
            ),
        },
        coords={
            'lat': ('lat', results['lats'], {'units': 'degrees_north', 'long_name': 'Latitude'}),
            'lon': ('lon', results['lons'], {'units': 'degrees_east', 'long_name': 'Longitude'}),
            'duration': ('duration', results['durations'], {'long_name': 'Accumulation duration'}),
            'return_period': ('return_period', results['return_periods'], {'units': 'years', 'long_name': 'Return period'}),
        },
        attrs={
            'title': 'CMORPH Extreme Value Analysis - Return Periods',
            'source': 'NOAA CDR CMORPH v1.0',
            'institution': 'Generated using Icechunk/VirtualiZarr pipeline',
            'history': f'Created {datetime.now().isoformat()}',
            'distribution_method': 'normal',
            'Conventions': 'CF-1.8',
            'pseudo_year_days': config['pseudo_year_days'],
            'note': 'Test run with pseudo-years (30-day periods) from 153 days of data'
        }
    )

    return ds


def write_to_icechunk(ds, config: Dict) -> str:
    """
    Stage 4: Write dataset to Icechunk store.

    Args:
        ds: xarray.Dataset to write
        config: Configuration dictionary

    Returns:
        Snapshot ID of the commit
    """
    import icechunk

    logger.info("Stage 4: Writing to Icechunk store...")

    # Create storage config
    storage = icechunk.gcs_storage(
        bucket=config['output_gcs_bucket'],
        prefix=config['output_gcs_prefix'],
        service_account_file=config['service_account_file'],
    )

    # Create or open repository
    try:
        repo = icechunk.Repository.open(storage)
        logger.info("Opened existing repository")
    except Exception:
        repo = icechunk.Repository.create(storage)
        logger.info("Created new repository")

    # Get writable session
    session = repo.writable_session("main")

    # Write dataset (actual data, not virtual)
    ds.to_zarr(session.store, mode='w')

    # Commit
    snapshot_id = session.commit(
        message=f"Phase 0 test: Return period analysis with {len(ds.lat)} x {len(ds.lon)} pixels"
    )

    logger.info(f"Committed to Icechunk: {snapshot_id}")
    return str(snapshot_id)


def main():
    """Main function to run Phase 0 extreme value analysis test."""
    import coiled
    from dask.distributed import Client

    logger.info("=" * 70)
    logger.info("PHASE 0: Extreme Value Analysis Test Run")
    logger.info("=" * 70)
    logger.info(f"Start time: {datetime.now().isoformat()}")

    # Load credentials
    with open(CONFIG['service_account_file'], 'r') as f:
        creds_content = f.read()
    logger.info("Credentials loaded")

    # Log configuration
    spatial = CONFIG['spatial']
    batch_lat = CONFIG['batch_size_lat']
    batch_lon = CONFIG['batch_size_lon']
    n_pixels = spatial['lat_count'] * spatial['lon_count']
    n_batches_lat = (spatial['lat_count'] + batch_lat - 1) // batch_lat
    n_batches_lon = (spatial['lon_count'] + batch_lon - 1) // batch_lon
    n_batches = n_batches_lat * n_batches_lon

    logger.info(f"\nProcessing configuration:")
    logger.info(f"  Spatial extent: {spatial['lat_count']} lat x {spatial['lon_count']} lon = {n_pixels} pixels")
    logger.info(f"  Batch size: {batch_lat} x {batch_lon} = {batch_lat * batch_lon} pixels/batch")
    logger.info(f"  Total batches: {n_batches}")
    logger.info(f"  Durations: {len(CONFIG['durations'])}")
    logger.info(f"  Return periods: {CONFIG['return_periods']}")

    # Create Coiled cluster
    logger.info("\n" + "=" * 70)
    logger.info("Starting Coiled Cluster...")
    logger.info("=" * 70)

    cluster_start = time.time()
    cluster = coiled.Cluster(
        name=f"phase0-eva-{datetime.now().strftime('%m%d-%H%M')}",
        **CONFIG['cluster']
    )
    client = Client(cluster)

    # Wait for workers to be ready
    logger.info("Waiting for workers to be ready...")
    client.wait_for_workers(n_workers=CONFIG['cluster']['n_workers'], timeout=300)
    cluster_time = time.time() - cluster_start

    logger.info(f"Cluster ready in {cluster_time:.1f}s")
    logger.info(f"Dashboard: {client.dashboard_link}")

    try:
        # Create batch arguments
        logger.info("\n" + "=" * 70)
        logger.info("Stage 1: Computing Annual Maxima on Coiled")
        logger.info("=" * 70)

        stage1_start = time.time()

        batches = []
        batch_id = 0
        for lat_start in range(0, spatial['lat_count'], batch_lat):
            for lon_start in range(0, spatial['lon_count'], batch_lon):
                lat_end = min(lat_start + batch_lat, spatial['lat_count'])
                lon_end = min(lon_start + batch_lon, spatial['lon_count'])

                lat_slice = slice(
                    spatial['lat_start_idx'] + lat_start,
                    spatial['lat_start_idx'] + lat_end
                )
                lon_slice = slice(
                    spatial['lon_start_idx'] + lon_start,
                    spatial['lon_start_idx'] + lon_end
                )

                batches.append((batch_id, lat_slice, lon_slice, creds_content, CONFIG))
                batch_id += 1

        logger.info(f"Submitting {len(batches)} batches to {CONFIG['cluster']['n_workers']} workers...")

        # Submit tasks
        futures = client.map(compute_annual_maxima_batch, batches)

        # Gather results
        batch_results = client.gather(futures)
        stage1_time = time.time() - stage1_start

        # Count successes
        successful = [r for r in batch_results if r['status'] == 'success']
        failed = [r for r in batch_results if r['status'] == 'error']

        logger.info(f"Stage 1 complete: {len(successful)}/{len(batches)} batches successful")
        logger.info(f"Stage 1 time: {stage1_time:.1f}s")

        if failed:
            for f in failed[:3]:
                logger.error(f"Batch {f['batch_id']} failed: {f['error']}")

        if not successful:
            raise RuntimeError("All batches failed!")

        # Stage 2: Fit distributions locally
        logger.info("\n" + "=" * 70)
        logger.info("Stage 2: Fitting Distributions (Local)")
        logger.info("=" * 70)

        stage2_start = time.time()
        results = process_stage2_local(successful, CONFIG)
        stage2_time = time.time() - stage2_start
        logger.info(f"Stage 2 time: {stage2_time:.1f}s")

        # Stage 3: Create dataset
        logger.info("\n" + "=" * 70)
        logger.info("Stage 3: Creating Output Dataset")
        logger.info("=" * 70)

        stage3_start = time.time()
        ds = create_output_dataset(results, CONFIG)
        stage3_time = time.time() - stage3_start
        logger.info(f"Stage 3 time: {stage3_time:.1f}s")
        logger.info(f"Dataset dimensions: {dict(ds.dims)}")

        # Stage 4: Write to Icechunk
        logger.info("\n" + "=" * 70)
        logger.info("Stage 4: Writing to Icechunk")
        logger.info("=" * 70)

        stage4_start = time.time()
        snapshot_id = write_to_icechunk(ds, CONFIG)
        stage4_time = time.time() - stage4_start
        logger.info(f"Stage 4 time: {stage4_time:.1f}s")

        # Summary
        total_time = time.time() - cluster_start

        logger.info("\n" + "=" * 70)
        logger.info("PHASE 0 COMPLETE - SUMMARY")
        logger.info("=" * 70)
        logger.info(f"\nTiming:")
        logger.info(f"  Cluster startup: {cluster_time:.1f}s")
        logger.info(f"  Stage 1 (annual maxima): {stage1_time:.1f}s")
        logger.info(f"  Stage 2 (distribution fit): {stage2_time:.1f}s")
        logger.info(f"  Stage 3 (create dataset): {stage3_time:.1f}s")
        logger.info(f"  Stage 4 (write output): {stage4_time:.1f}s")
        logger.info(f"  Total: {total_time:.1f}s ({total_time/60:.1f} min)")

        logger.info(f"\nOutput:")
        logger.info(f"  Location: gs://{CONFIG['output_gcs_bucket']}/{CONFIG['output_gcs_prefix']}")
        logger.info(f"  Snapshot: {snapshot_id}")
        logger.info(f"  Dimensions: {dict(ds.dims)}")

        logger.info(f"\nNext steps:")
        logger.info(f"  1. Run verify_phase0_output.py to validate results")
        logger.info(f"  2. Check return period values are physically reasonable")
        logger.info(f"  3. Scale up to full East Africa region")

    finally:
        logger.info("\nCleaning up Coiled cluster...")
        client.close()
        cluster.close()
        logger.info("Done!")


if __name__ == "__main__":
    main()
