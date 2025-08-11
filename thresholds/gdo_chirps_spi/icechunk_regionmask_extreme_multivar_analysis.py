#!/usr/bin/env python3
"""
Enhanced Multi-Variable Icechunk-based Regional Extreme Value Analysis v20250811
================================================================================

MULTI-VARIABLE PARALLEL PROCESSING VERSION

This enhanced script extends the v20250809 version to support:
1. Multiple SPI variables (spi3, spi6, spi9, spi12, spi24, spi48)
2. Parallel processing with 3 workers for improved performance
3. Configurable variable selection
4. Enhanced performance monitoring and logging

Key improvements over v20250809:
- Multi-variable support for comprehensive drought analysis
- Increased worker count for better parallel performance
- Variable-specific data loading and processing
- Enhanced error handling for different SPI types
- Improved result organization by variable

Based on the successful approach from v20250809 with worker-side Icechunk connections.

Usage:
    python icechunk_regionmask_extreme_analysis_enhanced_v20250811.py
    
    Or specify variables:
    python icechunk_regionmask_extreme_analysis_enhanced_v20250811.py --variables spi3,spi6,spi9
"""

import icechunk
import xarray as xr
import numpy as np
import geopandas as gpd
import regionmask
import warnings
import time
import xclim
from xclim.indices import stats
from xclim.indices.generic import select_resample_op
from google.oauth2 import service_account
import coiled
import dask
from dask.distributed import Client, get_worker
from dask.diagnostics import ProgressBar
import logging
import json
import os
from datetime import datetime
import pickle
from pathlib import Path
import traceback
import sys
import argparse

warnings.filterwarnings('ignore')

# Configuration
BASE_PREFIX = "t2spi1_east_africa_icechunk"
BUCKET_NAME = "cdi_arco"
SERVICE_ACCOUNT_FILE = "coiled-data-e4drr_202505.json"
GEOJSON_FILE = "icpac_adm1v3.geojson"
BUFFER_SIZE = 0.25

# Multi-variable configuration
DEFAULT_SPI_VARIABLES = ["spi3", "spi6", "spi9", "spi12", "spi24", "spi48"]
DEFAULT_N_WORKERS = 3

# Logging and session configuration
LOG_DIR = Path("logs")
SESSION_DIR = Path("sessions")
CLUSTER_CACHE_FILE = "cluster_cache.pkl"


def parse_arguments():
    """Parse command line arguments for variable selection and worker configuration"""
    parser = argparse.ArgumentParser(description='Multi-variable SPI extreme value analysis')
    
    parser.add_argument('--variables', 
                       type=str, 
                       default=','.join(DEFAULT_SPI_VARIABLES),
                       help=f'Comma-separated list of SPI variables to process (default: {",".join(DEFAULT_SPI_VARIABLES)})')
    
    parser.add_argument('--workers',
                       type=int,
                       default=DEFAULT_N_WORKERS,
                       help=f'Number of Dask workers to use (default: {DEFAULT_N_WORKERS})')
    
    parser.add_argument('--return-periods',
                       type=str,
                       default='2,5,10,25,50,100',
                       help='Comma-separated list of return periods (default: 2,5,10,25,50,100)')
    
    args = parser.parse_args()
    
    # Parse variables
    variables = [var.strip() for var in args.variables.split(',')]
    
    # Validate variables
    valid_variables = []
    for var in variables:
        if var in DEFAULT_SPI_VARIABLES:
            valid_variables.append(var)
        else:
            print(f"Warning: Skipping invalid variable '{var}'. Valid options: {DEFAULT_SPI_VARIABLES}")
    
    if not valid_variables:
        print(f"Error: No valid variables specified. Using default: {DEFAULT_SPI_VARIABLES}")
        valid_variables = DEFAULT_SPI_VARIABLES
    
    # Parse return periods
    try:
        return_periods = [int(x.strip()) for x in args.return_periods.split(',')]
    except ValueError:
        print("Error: Invalid return periods format. Using default: [2, 5, 10, 25, 50, 100]")
        return_periods = [2, 5, 10, 25, 50, 100]
    
    return valid_variables, args.workers, return_periods


def setup_logging(log_level=logging.INFO, variables=None):
    """Setup comprehensive logging system with file and console output"""
    # Create log directory if it doesn't exist
    LOG_DIR.mkdir(exist_ok=True)

    # Create unique session identifier
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    var_suffix = "_".join(variables) if variables else "multi"
    log_file = LOG_DIR / f"icechunk_analysis_v20250811_{var_suffix}_{session_id}.log"

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Clear any existing handlers
    logger.handlers.clear()

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(funcName)-25s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    console_formatter = logging.Formatter('%(levelname)-8s | %(message)s')

    # File handler for detailed logging
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)

    # Console handler for user-friendly output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logging.info(f"Logging initialized - Session ID: {session_id}")
    logging.info(f"Log file: {log_file}")
    logging.info(f"Processing variables: {variables}")

    return session_id, log_file


def save_cluster_info(cluster, client, session_id):
    """Save cluster connection information for reuse"""
    SESSION_DIR.mkdir(exist_ok=True)

    cluster_info = {
        'session_id': session_id,
        'cluster_name': cluster.name,
        'scheduler_address': client.scheduler.address,
        'dashboard_link': client.dashboard_link,
        'created_at': datetime.now().isoformat(),
        'status': 'active'
    }

    cache_file = SESSION_DIR / CLUSTER_CACHE_FILE

    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(cluster_info, f)
        logging.info(f"Cluster info saved to {cache_file}")
    except Exception as e:
        logging.warning(f"Failed to save cluster info: {e}")


def load_cluster_info():
    """Load existing cluster connection information"""
    cache_file = SESSION_DIR / CLUSTER_CACHE_FILE

    if not cache_file.exists():
        return None

    try:
        with open(cache_file, 'rb') as f:
            cluster_info = pickle.load(f)

        # Check if cluster info is recent (within 4 hours)
        created_at = datetime.fromisoformat(cluster_info['created_at'])
        age_hours = (datetime.now() - created_at).total_seconds() / 3600

        if age_hours > 4:
            logging.info(f"Cluster cache expired ({age_hours:.1f} hours old)")
            return None

        logging.info(
            f"Found cached cluster info: {cluster_info['cluster_name']}")
        return cluster_info

    except Exception as e:
        logging.warning(f"Failed to load cluster info: {e}")
        return None


def test_cluster_connection(client):
    """Test if cluster connection is still active"""
    try:
        worker_info = client.scheduler_info().get('workers', {})
        if len(worker_info) > 0:
            logging.info(
                f"Cluster connection active with {len(worker_info)} workers")
            return True
        else:
            logging.warning(
                "Cluster connection exists but no workers available")
            return False
    except Exception as e:
        logging.warning(f"Cluster connection test failed: {e}")
        return False


def setup_coiled_cluster(software_env="v5-geosfm-rm-x",
                         n_workers=DEFAULT_N_WORKERS,
                         reuse_existing=True,
                         session_id=None):
    """Setup Coiled Dask cluster with configurable worker count and error recovery"""

    # Try to reuse existing cluster first
    if reuse_existing:
        cluster_info = load_cluster_info()
        if cluster_info:
            try:
                logging.info(
                    f"Attempting to reconnect to existing cluster: {cluster_info['cluster_name']}"
                )

                # Try to get existing cluster
                cluster = coiled.Cluster(cluster_info['cluster_name'])
                client = cluster.get_client()

                # Test connection
                if test_cluster_connection(client):
                    logging.info(
                        f"✅ Reusing existing cluster: {client.dashboard_link}")
                    return client, cluster
                else:
                    logging.info(
                        "Existing cluster not responsive, creating new one...")
                    client.close()
                    cluster.close()

            except Exception as e:
                logging.warning(f"Failed to reuse existing cluster: {e}")

    logging.info(f"Creating new Coiled cluster with {n_workers} workers...")

    cluster_name = f"spi-multi-analysis-v20250811-{datetime.now().strftime('%m%d-%H%M')}"

    try:
        cluster = coiled.Cluster(
            name=cluster_name,
            software=software_env,
            n_workers=n_workers,
            scheduler_vm_types=["n2-standard-4"],
            worker_vm_types="n2-standard-4",
            region="us-east1",
            arm=False,
            compute_purchase_option="spot",
            workspace='geosfm')

        client = Client(cluster)

        # Verify cluster is ready
        worker_info = client.scheduler_info().get('workers', {})
        actual_workers = len(worker_info)

        logging.info(f"✅ New Coiled cluster ready: {client.dashboard_link}")
        logging.info(f"   Cluster name: {cluster_name}")
        logging.info(f"   Workers: {actual_workers}/{n_workers}")
        logging.info(f"   VM type: n2-standard-4")
        logging.info(f"   Region: us-east1")

        # Save cluster info for reuse
        if session_id:
            save_cluster_info(cluster, client, session_id)

        return client, cluster

    except Exception as e:
        logging.error(f"❌ Failed to setup Coiled cluster: {e}")
        raise


def upload_credentials_to_workers(client, service_account_file):
    """Upload service account credentials to all workers with verification"""
    logging.info("=" * 70)
    logging.info("UPLOADING CREDENTIALS TO WORKERS")
    logging.info("=" * 70)

    if not Path(service_account_file).exists():
        raise FileNotFoundError(f"Service account file not found: {service_account_file}")

    try:
        # Upload credentials file to all workers
        logging.info(f"Uploading {service_account_file} to all workers...")
        client.upload_file(service_account_file)
        
        # Wait for upload to complete
        time.sleep(10)
        
        # Verify upload on workers
        def verify_credentials_on_worker(creds_filename):
            """Verify that credentials file exists on worker"""
            try:
                worker = get_worker()
                local_dir = worker.local_directory
                creds_path = Path(local_dir) / creds_filename
                
                return {
                    'worker_id': worker.address,
                    'local_dir': local_dir,
                    'creds_path': str(creds_path),
                    'file_exists': creds_path.exists(),
                    'status': 'success' if creds_path.exists() else 'missing_file'
                }
            except Exception as e:
                return {
                    'worker_id': 'unknown',
                    'error': str(e),
                    'status': 'error'
                }

        # Test credential access on all workers
        futures = client.map(verify_credentials_on_worker, [service_account_file] * len(client.scheduler_info()['workers']))
        verification_results = client.gather(futures)

        # Check results
        successful_workers = [r for r in verification_results if r['status'] == 'success']
        failed_workers = [r for r in verification_results if r['status'] != 'success']

        logging.info(f"✅ Credentials verified on {len(successful_workers)} workers")
        
        if failed_workers:
            logging.warning(f"❌ Credentials failed on {len(failed_workers)} workers")
            for failed in failed_workers:
                logging.warning(f"   Worker {failed.get('worker_id', 'unknown')}: {failed.get('error', 'missing file')}")
        
        if len(successful_workers) == 0:
            raise RuntimeError("No workers have access to credentials")

        return True

    except Exception as e:
        logging.error(f"❌ Failed to upload credentials: {e}")
        raise


def load_administrative_regions():
    """Load administrative regions and create regionmask with enhanced logging"""
    logging.info("=" * 70)
    logging.info("LOADING ADMINISTRATIVE REGIONS")
    logging.info("=" * 70)

    start_time = time.time()

    try:
        # Load GeoJSON file
        logging.debug(f"Loading GeoJSON file: {GEOJSON_FILE}")
        gdf = gpd.read_file(GEOJSON_FILE)
        gdf['region_idx'] = np.arange(len(gdf))

        load_time = time.time() - start_time

        logging.info(
            f"✅ Loaded {len(gdf)} administrative regions in {load_time:.2f} seconds"
        )
        logging.info(f"   Columns: {list(gdf.columns)}")

        # Check for the correct name column
        name_col = 'GID_1' if 'GID_1' in gdf.columns else 'shapeName'
        id_col = 'region_idx' if 'region_idx' in gdf.columns else 'shapeID'

        logging.info(f"   Using name column: {name_col}")
        logging.info(f"   Sample regions: {gdf[name_col].head().tolist()}")

        # Fix invalid geometries that can cause overlap detection issues
        invalid_count = (~gdf.geometry.is_valid).sum()
        if invalid_count > 0:
            logging.warning(f"Fixing {invalid_count} invalid geometries...")
            start_fix = time.time()
            gdf.geometry = gdf.geometry.buffer(0)
            fix_time = time.time() - start_fix
            logging.info(
                f"   Geometry fixes completed in {fix_time:.2f} seconds")

        # Create regionmask using the appropriate columns
        logging.debug("Creating regionmask...")
        regions = regionmask.from_geopandas(gdf,
                                            names=name_col,
                                            abbrevs=id_col,
                                            name="icpac_regions")

        logging.info(f"✅ Created regionmask with {len(regions)} regions")

        return gdf, regions

    except Exception as e:
        logging.error(f"❌ Failed to load regions: {e}")
        logging.debug(traceback.format_exc())
        raise


def get_region_metadata(gdf, regions):
    """Extract region metadata for worker processing"""
    logging.info("=" * 70)
    logging.info("EXTRACTING REGION METADATA")
    logging.info("=" * 70)

    region_metadata = []
    
    for i, region in enumerate(regions):
        try:
            # Get region name from different possible columns
            if 'shapeName' in gdf.columns:
                region_name = gdf.iloc[i]['shapeName']
            elif 'GID_1' in gdf.columns:
                region_name = gdf.iloc[i]['GID_1']
            else:
                region_name = f"Region_{i}"

            # Get region bounds
            bounds = gdf.iloc[i].geometry.bounds
            
            metadata = {
                'region_id': i,
                'region_name': region_name,
                'bounds': bounds,  # (minx, miny, maxx, maxy)
                'geometry': gdf.iloc[i].geometry.__geo_interface__  # Serialize geometry
            }
            region_metadata.append(metadata)
            
        except Exception as e:
            logging.warning(f"Failed to extract metadata for region {i}: {e}")
            continue

    logging.info(f"✅ Extracted metadata for {len(region_metadata)} regions")
    return region_metadata


def validate_spi_variable_availability(bucket, prefix, spi_type):
    """Check if a specific SPI variable is available in the Icechunk repository"""
    try:
        storage = icechunk.gcs_storage(
            bucket=bucket,
            prefix=prefix,
            service_account_file=SERVICE_ACCOUNT_FILE)
        
        repo = icechunk.Repository.open(storage)
        session = repo.readonly_session("main")
        
        group_name = f"{spi_type}_data"
        dataset = xr.open_zarr(session.store, group=group_name)
        
        # Check for expected variable name
        expected_var = f"spc{spi_type[3:]:0>2}"  # e.g., spi3 -> spc03
        
        available_vars = list(dataset.data_vars)
        spi_vars = [var for var in available_vars if 'spi' in var.lower() or 'spc' in var.lower()]
        
        if expected_var in available_vars or spi_vars:
            logging.info(f"✅ {spi_type.upper()} data available - vars: {spi_vars}")
            return True, spi_vars[0] if spi_vars else expected_var
        else:
            logging.warning(f"❌ {spi_type.upper()} data not found - available vars: {available_vars}")
            return False, None
            
    except Exception as e:
        logging.warning(f"❌ Failed to validate {spi_type.upper()} availability: {e}")
        return False, None


def process_region_with_worker_icechunk(region_metadata, bucket, prefix, group_name, 
                                       creds_filename, return_periods, spi_var_name, spi_type):
    """
    Process a single region with worker-side Icechunk connection for specific SPI variable
    
    This function runs on individual workers and:
    1. Establishes its own Icechunk connection using uploaded credentials
    2. Loads only the required data subset for the specific SPI variable
    3. Applies regionmask filtering
    4. Performs extreme value analysis
    5. Returns results to main process
    """
    try:
        import icechunk
        import xarray as xr
        import numpy as np
        import regionmask
        from shapely.geometry import shape
        from xclim.indices import stats
        from xclim.indices.generic import select_resample_op
        from dask.distributed import get_worker
        from pathlib import Path

        # Get worker information and credentials path
        worker = get_worker()
        worker_id = worker.address
        local_dir = worker.local_directory
        creds_path = Path(local_dir) / creds_filename

        region_id = region_metadata['region_id']
        region_name = region_metadata['region_name']
        
        # Verify credentials exist
        if not creds_path.exists():
            return {
                'spi_type': spi_type,
                'region_id': region_id,
                'region_name': region_name,
                'return_periods': return_periods,
                'return_levels': [np.nan] * len(return_periods),
                'status': f'error: credentials not found at {creds_path}',
                'worker_id': worker_id,
                'n_years': 0
            }

        # Initialize Icechunk connection on worker
        storage = icechunk.gcs_storage(
            bucket=bucket,
            prefix=prefix,
            service_account_file=str(creds_path)
        )
        
        repo = icechunk.Repository.open(storage)
        session = repo.readonly_session("main")
        
        # Load dataset on worker
        dataset = xr.open_zarr(session.store, group=group_name, consolidated=False)
        
        # Get SPI data - try specified variable first, then search
        if spi_var_name not in dataset.data_vars:
            available_vars = list(dataset.data_vars)
            for var in available_vars:
                if 'spi' in var.lower() or 'spc' in var.lower():
                    spi_var_name = var
                    break
        
        if spi_var_name not in dataset.data_vars:
            return {
                'spi_type': spi_type,
                'region_id': region_id,
                'region_name': region_name,
                'return_periods': return_periods,
                'return_levels': [np.nan] * len(return_periods),
                'status': f'error: SPI variable not found. Available: {list(dataset.data_vars)}',
                'worker_id': worker_id,
                'n_years': 0
            }
        
        spi_data = dataset[spi_var_name]
        if 'band' in spi_data.dims:
            spi_data = spi_data.squeeze('band')

        # Create region geometry and mask on worker
        region_geom = shape(region_metadata['geometry'])
        
        # Get coordinate arrays
        lons = spi_data.lon.values
        lats = spi_data.lat.values
        
        # Create mask for this specific region
        region_gdf = gpd.GeoDataFrame([{'geometry': region_geom, 'region_id': region_id}])
        region_mask = regionmask.mask_geopandas(region_gdf.geometry, lons, lats)
        
        # Extract annual extremes on worker
        spi_data.attrs['units'] = '1'
        annual_minima = select_resample_op(
            spi_data,
            op='min',
            freq='YS'  # Annual frequency starting in January
        )
        
        # Apply regional mask
        region_data = annual_minima.where(region_mask == 0)  # regionmask uses 0 for first region
        
        # Compute regional time series
        region_values = region_data.values
        
        if isinstance(region_values, np.ndarray):
            if region_values.ndim > 1:
                # Take spatial mean over the region, ignoring NaNs
                region_ts = np.nanmean(region_values, axis=tuple(range(1, region_values.ndim)))
            else:
                region_ts = region_values
        else:
            region_ts = np.array(region_values).flatten()

        # Remove NaN values
        valid_data = region_ts[~np.isnan(region_ts)]

        if len(valid_data) < 10:  # Need sufficient data points
            return {
                'spi_type': spi_type,
                'region_id': region_id,
                'region_name': region_name,
                'return_periods': return_periods,
                'return_levels': [np.nan] * len(return_periods),
                'status': 'insufficient_data',
                'worker_id': worker_id,
                'n_years': len(valid_data)
            }

        # For drought analysis, convert minima to maxima (negative values)
        drought_data = -1 * valid_data

        # Convert to xarray for xclim compatibility
        drought_xr = xr.DataArray(drought_data, dims=['time'])

        # Calculate return levels using xclim's fa() function
        fa_result = stats.fa(drought_xr,
                             t=return_periods,
                             dist='genextreme',
                             mode='max')

        # Convert back to SPI scale (drought levels)
        drought_return_levels = -1 * fa_result

        return {
            'spi_type': spi_type,
            'region_id': region_id,
            'region_name': region_name,
            'return_periods': return_periods,
            'return_levels': drought_return_levels.values.tolist(),
            'status': 'success',
            'worker_id': worker_id,
            'n_years': len(valid_data)
        }

    except Exception as e:
        import traceback
        return {
            'spi_type': spi_type,
            'region_id': region_metadata.get('region_id', -1),
            'region_name': region_metadata.get('region_name', 'unknown'),
            'return_periods': return_periods,
            'return_levels': [np.nan] * len(return_periods),
            'status': f'error: {str(e)}',
            'worker_id': getattr(get_worker(), 'address', 'unknown') if 'get_worker' in locals() else 'unknown',
            'n_years': 0,
            'traceback': traceback.format_exc()
        }


def calculate_return_periods_multi_variable(region_metadata, client, spi_variables, return_periods):
    """Calculate return periods for multiple SPI variables using worker-side connections"""
    logging.info("=" * 70)
    logging.info("CALCULATING RETURN PERIODS FOR MULTIPLE SPI VARIABLES")
    logging.info("=" * 70)

    logging.info(f"SPI variables to process: {spi_variables}")
    logging.info(f"Return periods: {return_periods}")
    logging.info(f"Processing {len(region_metadata)} regions")

    all_results = {}
    bucket = BUCKET_NAME
    creds_filename = SERVICE_ACCOUNT_FILE

    # Process each SPI variable
    for spi_type in spi_variables:
        logging.info(f"\n{'='*50}")
        logging.info(f"PROCESSING {spi_type.upper()}")
        logging.info(f"{'='*50}")
        
        start_time = time.time()
        
        # Configuration for this SPI variable
        prefix = f"{BASE_PREFIX}_{spi_type}"
        group_name = f"{spi_type}_data"
        
        # Validate variable availability
        is_available, spi_var_name = validate_spi_variable_availability(bucket, prefix, spi_type)
        
        if not is_available:
            logging.warning(f"❌ Skipping {spi_type.upper()} - data not available")
            all_results[spi_type] = {
                'status': 'skipped',
                'reason': 'data_not_available',
                'results': []
            }
            continue
        
        try:
            # Submit tasks to workers
            logging.info(f"Submitting {len(region_metadata)} tasks for {spi_type.upper()}...")
            
            futures = []
            for region_meta in region_metadata:
                future = client.submit(
                    process_region_with_worker_icechunk,
                    region_meta,
                    bucket,
                    prefix,
                    group_name,
                    creds_filename,
                    return_periods,
                    spi_var_name,
                    spi_type
                )
                futures.append(future)

            # Collect results with progress tracking
            results = []
            completed = 0

            for i, future in enumerate(futures):
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1

                    if completed % 25 == 0 or completed == len(futures):
                        progress = (completed / len(futures)) * 100
                        logging.info(
                            f"   {spi_type.upper()} Progress: {completed}/{len(futures)} regions ({progress:.1f}%)"
                        )

                except Exception as e:
                    logging.error(f"Task {i} failed for {spi_type.upper()}: {e}")
                    results.append({
                        'spi_type': spi_type,
                        'region_id': i,
                        'region_name': f'failed_region_{i}',
                        'return_periods': return_periods,
                        'return_levels': [np.nan] * len(return_periods),
                        'status': f'error: {str(e)}',
                        'worker_id': 'unknown',
                        'n_years': 0
                    })

        except Exception as e:
            logging.error(f"Failed to process {spi_type.upper()}: {e}")
            all_results[spi_type] = {
                'status': 'error',
                'reason': str(e),
                'results': []
            }
            continue

        computation_time = time.time() - start_time

        # Process results for this variable
        successful_regions = [r for r in results if r['status'] == 'success']
        failed_regions = [r for r in results if r['status'] != 'success']

        logging.info(
            f"✅ {spi_type.upper()} completed in {computation_time:.2f} seconds"
        )
        logging.info(f"   Successful regions: {len(successful_regions)}")
        logging.info(f"   Failed regions: {len(failed_regions)}")
        logging.info(
            f"   Processing rate: {len(region_metadata)/computation_time:.2f} regions/second"
        )

        # Display sample results
        if successful_regions:
            sample = successful_regions[0]
            logging.info(f"\n   Sample results for {sample['region_name']} ({spi_type.upper()}):")
            for T, level in zip(sample['return_periods'], sample['return_levels']):
                logging.info(f"      {T:3d}-year drought: SPI = {level:.3f}")

        # Store results for this variable
        all_results[spi_type] = {
            'status': 'completed',
            'processing_time': computation_time,
            'successful_regions': len(successful_regions),
            'failed_regions': len(failed_regions),
            'results': results
        }

    # Summary across all variables
    logging.info("\n" + "=" * 70)
    logging.info("MULTI-VARIABLE PROCESSING SUMMARY")
    logging.info("=" * 70)
    
    for spi_type, var_results in all_results.items():
        status = var_results['status']
        if status == 'completed':
            logging.info(f"✅ {spi_type.upper()}: {var_results['successful_regions']} regions processed")
        elif status == 'skipped':
            logging.info(f"⚠️  {spi_type.upper()}: Skipped - {var_results['reason']}")
        else:
            logging.info(f"❌ {spi_type.upper()}: Failed - {var_results['reason']}")

    return all_results


def cleanup_cluster(client, cluster, force=False):
    """Gracefully cleanup Dask cluster with proper disentanglement"""
    logging.info("Starting cluster cleanup...")

    try:
        if client:
            # Cancel any running tasks
            try:
                client.cancel(client.futures, force=force)
                logging.info("Cancelled running tasks")
            except Exception as e:
                logging.warning(f"Error cancelling tasks: {e}")

            # Close client connection
            try:
                client.close(timeout=10)
                logging.info("Client connection closed")
            except Exception as e:
                logging.warning(f"Error closing client: {e}")

        if cluster:
            # Close cluster
            try:
                cluster.close()
                logging.info("Cluster closed successfully")
            except Exception as e:
                logging.warning(f"Error closing cluster: {e}")

        # Clear cluster cache
        cache_file = SESSION_DIR / CLUSTER_CACHE_FILE
        if cache_file.exists():
            try:
                cache_file.unlink()
                logging.info("Cluster cache cleared")
            except Exception as e:
                logging.warning(f"Error clearing cache: {e}")

    except Exception as e:
        logging.error(f"Error during cluster cleanup: {e}")
        if force:
            logging.warning("Forcing cleanup despite errors")
        else:
            raise


def save_results(all_results, session_id, spi_variables, n_workers, output_file=None):
    """Save multi-variable results to JSON file with enhanced metadata"""
    if output_file is None:
        var_suffix = "_".join(spi_variables)
        output_file = f"multi_spi_extreme_analysis_results_v20250811_{var_suffix}_{session_id}.json"

    logging.info("=" * 70)
    logging.info("SAVING MULTI-VARIABLE RESULTS")
    logging.info("=" * 70)

    try:
        # Calculate overall statistics
        total_regions_processed = 0
        total_successful = 0
        total_failed = 0
        
        variable_summary = {}
        
        for spi_type, var_results in all_results.items():
            if var_results['status'] == 'completed':
                total_successful += var_results['successful_regions']
                total_failed += var_results['failed_regions']
                total_regions_processed += len(var_results['results'])
                
                variable_summary[spi_type] = {
                    'status': 'completed',
                    'processing_time': var_results['processing_time'],
                    'regions_processed': len(var_results['results']),
                    'successful_regions': var_results['successful_regions'],
                    'failed_regions': var_results['failed_regions']
                }
            else:
                variable_summary[spi_type] = {
                    'status': var_results['status'],
                    'reason': var_results.get('reason', 'unknown')
                }

        # Add metadata to results
        metadata = {
            'session_id': session_id,
            'created_at': datetime.now().isoformat(),
            'script_version': 'multi_variable_v20250811',
            'configuration': {
                'base_prefix': BASE_PREFIX,
                'spi_variables': spi_variables,
                'n_workers': n_workers,
                'bucket_name': BUCKET_NAME,
                'geojson_file': GEOJSON_FILE
            },
            'summary': {
                'variables_requested': spi_variables,
                'variables_processed': list(variable_summary.keys()),
                'total_regions_processed': total_regions_processed,
                'total_successful_regions': total_successful,
                'total_failed_regions': total_failed
            },
            'variable_summary': variable_summary
        }

        output_data = {
            'metadata': metadata, 
            'results_by_variable': all_results
        }

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)

        logging.info(f"✅ Multi-variable results saved to {output_file}")
        logging.info(f"   Variables processed: {len(variable_summary)}")
        logging.info(f"   Total regions processed: {total_regions_processed}")
        logging.info(f"   Total successful: {total_successful}")
        logging.info(f"   Total failed: {total_failed}")

        return output_file

    except Exception as e:
        logging.error(f"❌ Failed to save results: {e}")
        logging.debug(traceback.format_exc())


def main():
    """Main execution function with multi-variable support"""
    
    # Parse command line arguments
    spi_variables, n_workers, return_periods = parse_arguments()
    
    # Initialize logging system
    session_id, log_file = setup_logging(variables=spi_variables)

    logging.info("=" * 70)
    logging.info("MULTI-VARIABLE ICECHUNK REGIONMASK EXTREME VALUE ANALYSIS v20250811")
    logging.info("=" * 70)
    logging.info(f"Session ID: {session_id}")
    logging.info(f"SPI Variables: {spi_variables}")
    logging.info(f"Workers: {n_workers}")
    logging.info(f"Return periods: {return_periods}")

    cluster = None
    client = None
    success = False

    try:
        # Step 1: Setup Coiled Dask cluster with specified worker count
        logging.info(f"STEP 1: Setting up Dask cluster with {n_workers} workers")
        client, cluster = setup_coiled_cluster(n_workers=n_workers, session_id=session_id)

        # Step 2: Upload credentials to workers
        logging.info("STEP 2: Uploading credentials to workers")
        upload_credentials_to_workers(client, SERVICE_ACCOUNT_FILE)

        # Step 3: Load administrative regions
        logging.info("STEP 3: Loading administrative regions")
        gdf, regions = load_administrative_regions()

        # Step 4: Extract region metadata for workers
        logging.info("STEP 4: Extracting region metadata")
        region_metadata = get_region_metadata(gdf, regions)

        # Step 5: Calculate return periods for multiple variables
        logging.info("STEP 5: Calculating return periods for multiple SPI variables")
        all_results = calculate_return_periods_multi_variable(
            region_metadata,
            client,
            spi_variables,
            return_periods)

        # Step 6: Save results
        logging.info("STEP 6: Saving multi-variable results")
        output_file = save_results(all_results, session_id, spi_variables, n_workers)

        # Summary
        logging.info("=" * 70)
        logging.info("MULTI-VARIABLE ANALYSIS COMPLETED SUCCESSFULLY")
        logging.info("=" * 70)

        # Count successful processing across all variables
        total_successful = 0
        completed_vars = 0
        for spi_type, var_results in all_results.items():
            if var_results['status'] == 'completed':
                total_successful += var_results['successful_regions']
                completed_vars += 1

        logging.info(f"✅ Processed {completed_vars} SPI variables successfully")
        logging.info(f"✅ Total regions processed: {total_successful}")
        logging.info(f"✅ Results saved to: {output_file}")
        logging.info(f"✅ Log file: {log_file}")
        logging.info(f"✅ Session ID: {session_id}")

        success = True

    except KeyboardInterrupt:
        logging.warning("Analysis interrupted by user")
        success = False

    except Exception as e:
        logging.error(f"❌ Multi-variable analysis failed: {e}")
        logging.debug(traceback.format_exc())
        success = False

    finally:
        # Cleanup cluster
        logging.info("Performing cleanup...")
        try:
            cleanup_cluster(client, cluster, force=not success)
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")

        logging.info(f"Multi-variable analysis session {session_id} completed")

        if success:
            logging.info("✅ Session completed successfully")
        else:
            logging.error("❌ Session completed with errors")

    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)