#!/usr/bin/env python3
"""
Sequential Multi-Variable Icechunk-based Regional Extreme Value Analysis v20250812
==================================================================================

SEQUENTIAL PROCESSING VERSION

This script processes multiple SPI variables sequentially using separate clusters,
based on the stable v20250809 approach that works reliably with single variables.

Key features:
1. Sequential processing of SPI variables (spi3, spi6, spi9, spi12, spi24, spi48)
2. Fresh cluster for each variable to avoid memory issues
3. Individual output files for each variable
4. Stable n2-standard-4 VM configuration
5. Command-line variable selection

Usage:
    # Process single variable
    python icechunk_regionmask_extreme_analysis_enhanced_v20250812.py --variable spi6
    
    # Process multiple variables sequentially
    python icechunk_regionmask_extreme_analysis_enhanced_v20250812.py --variables spi3,spi6,spi9

Based on the proven approach from v20250809 with individual cluster management.
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

# Available SPI variables
AVAILABLE_SPI_VARIABLES = ["spi3", "spi6", "spi9", "spi12", "spi24", "spi48"]

# Logging and session configuration
LOG_DIR = Path("logs")
SESSION_DIR = Path("sessions")


def parse_arguments():
    """Parse command line arguments for variable selection"""
    parser = argparse.ArgumentParser(description='Sequential SPI extreme value analysis')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--variable', 
                      type=str,
                      help=f'Single SPI variable to process. Options: {", ".join(AVAILABLE_SPI_VARIABLES)}')
    
    group.add_argument('--variables',
                      type=str,
                      help=f'Comma-separated list of SPI variables to process sequentially. Options: {", ".join(AVAILABLE_SPI_VARIABLES)}')
    
    parser.add_argument('--return-periods',
                       type=str,
                       default='2,5,10,25,50,100',
                       help='Comma-separated list of return periods (default: 2,5,10,25,50,100)')
    
    args = parser.parse_args()
    
    # Parse variables
    if args.variable:
        variables = [args.variable.strip()]
    else:
        variables = [var.strip() for var in args.variables.split(',')]
    
    # Validate variables
    valid_variables = []
    for var in variables:
        if var in AVAILABLE_SPI_VARIABLES:
            valid_variables.append(var)
        else:
            print(f"Warning: Skipping invalid variable '{var}'. Valid options: {AVAILABLE_SPI_VARIABLES}")
    
    if not valid_variables:
        print(f"Error: No valid variables specified. Valid options: {AVAILABLE_SPI_VARIABLES}")
        sys.exit(1)
    
    # Parse return periods
    try:
        return_periods = [int(x.strip()) for x in args.return_periods.split(',')]
    except ValueError:
        print("Error: Invalid return periods format. Using default: [2, 5, 10, 25, 50, 100]")
        return_periods = [2, 5, 10, 25, 50, 100]
    
    return valid_variables, return_periods


def setup_logging(log_level=logging.INFO, variable=None):
    """Setup comprehensive logging system with file and console output"""
    # Create log directory if it doesn't exist
    LOG_DIR.mkdir(exist_ok=True)

    # Create unique session identifier
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    var_suffix = variable if variable else "multi"
    log_file = LOG_DIR / f"icechunk_analysis_v20250812_{var_suffix}_{session_id}.log"

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

    return session_id, log_file


def setup_coiled_cluster_for_variable(spi_variable, software_env="v5-geosfm-rm-x", n_workers=1):
    """Setup fresh Coiled Dask cluster for a specific SPI variable"""
    
    logging.info(f"Creating fresh cluster for {spi_variable.upper()}")
    
    cluster_name = f"spi-{spi_variable}-analysis-v20250812-{datetime.now().strftime('%m%d-%H%M')}"

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

        logging.info(f"✅ New Coiled cluster ready for {spi_variable.upper()}: {client.dashboard_link}")
        logging.info(f"   Cluster name: {cluster_name}")
        logging.info(f"   Workers: {actual_workers}/{n_workers}")
        logging.info(f"   VM type: n2-standard-4")
        logging.info(f"   Region: us-east1")

        return client, cluster

    except Exception as e:
        logging.error(f"❌ Failed to setup Coiled cluster for {spi_variable.upper()}: {e}")
        raise


def upload_credentials_to_workers(client, service_account_file):
    """Upload service account credentials to all workers with verification"""
    logging.info("=" * 50)
    logging.info("UPLOADING CREDENTIALS TO WORKERS")
    logging.info("=" * 50)

    if not Path(service_account_file).exists():
        raise FileNotFoundError(f"Service account file not found: {service_account_file}")

    try:
        # Upload credentials file to all workers
        logging.info(f"Uploading {service_account_file} to all workers...")
        client.upload_file(service_account_file)
        
        # Wait longer for upload to complete (especially for larger datasets)
        time.sleep(20)
        
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
    logging.info("=" * 50)
    logging.info("LOADING ADMINISTRATIVE REGIONS")
    logging.info("=" * 50)

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
    logging.info("=" * 50)
    logging.info("EXTRACTING REGION METADATA")
    logging.info("=" * 50)

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


def process_region_with_worker_icechunk(region_metadata, bucket, prefix, group_name, 
                                       creds_filename, return_periods, spi_var_name):
    """
    Process a single region with worker-side Icechunk connection
    
    This function runs on individual workers and:
    1. Establishes its own Icechunk connection using uploaded credentials
    2. Loads only the required data subset
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
        
        # Get SPI data
        if spi_var_name not in dataset.data_vars:
            available_vars = list(dataset.data_vars)
            for var in available_vars:
                if 'spi' in var.lower() or 'spc' in var.lower():
                    spi_var_name = var
                    break
        
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
            'region_id': region_metadata.get('region_id', -1),
            'region_name': region_metadata.get('region_name', 'unknown'),
            'return_periods': return_periods,
            'return_levels': [np.nan] * len(return_periods),
            'status': f'error: {str(e)}',
            'worker_id': getattr(get_worker(), 'address', 'unknown') if 'get_worker' in locals() else 'unknown',
            'n_years': 0,
            'traceback': traceback.format_exc()
        }


def calculate_return_periods_for_variable(region_metadata, client, spi_variable, return_periods):
    """Calculate return periods for a specific SPI variable using worker-side connections"""
    logging.info(f"=" * 60)
    logging.info(f"CALCULATING RETURN PERIODS FOR {spi_variable.upper()}")
    logging.info(f"=" * 60)

    logging.info(f"Return periods to calculate: {return_periods}")
    logging.info(f"Processing {len(region_metadata)} regions")

    # Configuration for worker tasks
    bucket = BUCKET_NAME
    prefix = f"{BASE_PREFIX}_{spi_variable}"
    group_name = f"{spi_variable}_data"
    creds_filename = SERVICE_ACCOUNT_FILE
    spi_var_name = f"spc{spi_variable[3:]:0>2}"  # Convert spi9 -> spc09

    start_time = time.time()

    try:
        # Submit tasks to workers
        logging.info(f"Submitting {len(region_metadata)} tasks to workers...")
        
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
                spi_var_name
            )
            futures.append(future)

        # Collect results with progress tracking and longer timeouts
        results = []
        completed = 0

        for i, future in enumerate(futures):
            try:
                # Increase timeout for larger datasets
                result = future.result(timeout=300)  # 5 minutes per region
                results.append(result)
                completed += 1

                if completed % 10 == 0 or completed == len(futures):
                    progress = (completed / len(futures)) * 100
                    logging.info(
                        f"Progress: {completed}/{len(futures)} regions ({progress:.1f}%)"
                    )

            except Exception as e:
                logging.error(f"Task {i} failed: {e}")
                results.append({
                    'region_id': i,
                    'region_name': f'failed_region_{i}',
                    'return_periods': return_periods,
                    'return_levels': [np.nan] * len(return_periods),
                    'status': f'error: {str(e)}',
                    'worker_id': 'unknown',
                    'n_years': 0
                })

    except Exception as e:
        logging.error(f"Failed to process regions for {spi_variable.upper()}: {e}")
        raise

    computation_time = time.time() - start_time

    # Process results
    successful_regions = [r for r in results if r['status'] == 'success']
    failed_regions = [r for r in results if r['status'] != 'success']

    logging.info(
        f"✅ {spi_variable.upper()} calculation completed in {computation_time:.2f} seconds"
    )
    logging.info(f"   Successful regions: {len(successful_regions)}")
    logging.info(f"   Failed regions: {len(failed_regions)}")
    logging.info(
        f"   Processing rate: {len(region_metadata)/computation_time:.2f} regions/second"
    )

    # Log failure details if any
    if failed_regions:
        for failed in failed_regions[:3]:  # Show first 3 failures
            logging.warning(
                f"   Failed region {failed['region_id']}: {failed['status']}")

    # Display sample results
    if successful_regions:
        sample = successful_regions[0]
        logging.info(f"\nSample results for {sample['region_name']} ({spi_variable.upper()}):")
        for T, level in zip(sample['return_periods'], sample['return_levels']):
            logging.info(f"   {T:3d}-year drought: SPI = {level:.3f}")

    return results


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

    except Exception as e:
        logging.error(f"Error during cluster cleanup: {e}")
        if force:
            logging.warning("Forcing cleanup despite errors")
        else:
            raise


def save_results(results, session_id, spi_variable, output_file=None):
    """Save results to JSON file with enhanced metadata"""
    if output_file is None:
        output_file = f"{spi_variable}_extreme_analysis_results_v20250812_{session_id}.json"

    logging.info("=" * 50)
    logging.info(f"SAVING {spi_variable.upper()} RESULTS")
    logging.info("=" * 50)

    try:
        # Add metadata to results
        metadata = {
            'session_id': session_id,
            'created_at': datetime.now().isoformat(),
            'script_version': 'sequential_v20250812',
            'spi_variable': spi_variable,
            'configuration': {
                'base_prefix': BASE_PREFIX,
                'bucket_name': BUCKET_NAME,
                'geojson_file': GEOJSON_FILE
            },
            'summary': {
                'total_regions': len(results),
                'successful_regions': len([r for r in results if r['status'] == 'success']),
                'failed_regions': len([r for r in results if r['status'] != 'success'])
            }
        }

        output_data = {'metadata': metadata, 'results': results}

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)

        logging.info(f"✅ {spi_variable.upper()} results saved to {output_file}")
        logging.info(f"   Total regions processed: {len(results)}")
        logging.info(f"   Successful: {metadata['summary']['successful_regions']}")
        logging.info(f"   Failed: {metadata['summary']['failed_regions']}")
        
        return output_file

    except Exception as e:
        logging.error(f"❌ Failed to save {spi_variable.upper()} results: {e}")
        logging.debug(traceback.format_exc())
        return None


def process_single_variable(spi_variable, region_metadata, return_periods, session_id, max_retries=2):
    """Process a single SPI variable with its own cluster and retry capability"""
    
    logging.info(f"\n{'='*80}")
    logging.info(f"PROCESSING {spi_variable.upper()}")
    logging.info(f"{'='*80}")
    
    for attempt in range(max_retries + 1):
        if attempt > 0:
            logging.info(f"🔄 Retry attempt {attempt}/{max_retries} for {spi_variable.upper()}")
            time.sleep(30)  # Wait before retry
        
        cluster = None
        client = None
        success = False
        
        try:
            # Step 1: Setup fresh cluster for this variable
            logging.info(f"STEP 1: Setting up Dask cluster for {spi_variable.upper()}")
            client, cluster = setup_coiled_cluster_for_variable(spi_variable)

            # Step 2: Upload credentials to workers
            logging.info(f"STEP 2: Uploading credentials to workers")
            upload_credentials_to_workers(client, SERVICE_ACCOUNT_FILE)

            # Step 3: Calculate return periods for this variable
            logging.info(f"STEP 3: Calculating return periods for {spi_variable.upper()}")
            results = calculate_return_periods_for_variable(
                region_metadata, client, spi_variable, return_periods)

            # Step 4: Save results for this variable
            logging.info(f"STEP 4: Saving {spi_variable.upper()} results")
            output_file = save_results(results, session_id, spi_variable)
            
            successful_regions = [r for r in results if r['status'] == 'success']
            logging.info(f"✅ {spi_variable.upper()} completed: {len(successful_regions)} regions processed")
            
            success = True
            return output_file, len(successful_regions)

        except Exception as e:
            logging.error(f"❌ {spi_variable.upper()} processing failed (attempt {attempt + 1}): {e}")
            logging.debug(traceback.format_exc())
            
            # Don't retry on final attempt
            if attempt == max_retries:
                return None, 0

        finally:
            # Cleanup cluster for this variable
            logging.info(f"Cleaning up cluster for {spi_variable.upper()}...")
            try:
                cleanup_cluster(client, cluster, force=not success)
                time.sleep(10)  # Wait after cleanup
            except Exception as e:
                logging.error(f"Error during {spi_variable.upper()} cleanup: {e}")
    
    return None, 0


def main():
    """Main execution function with sequential variable processing"""

    # Parse command line arguments
    spi_variables, return_periods = parse_arguments()
    
    # Initialize logging system
    session_id, log_file = setup_logging(variable="_".join(spi_variables))

    logging.info("=" * 80)
    logging.info("SEQUENTIAL MULTI-VARIABLE ICECHUNK REGIONMASK EXTREME VALUE ANALYSIS v20250812")
    logging.info("=" * 80)
    logging.info(f"Session ID: {session_id}")
    logging.info(f"SPI Variables: {spi_variables}")
    logging.info(f"Return periods: {return_periods}")
    logging.info(f"Processing mode: Sequential (separate cluster per variable)")

    try:
        # Check service account file exists
        if not Path(SERVICE_ACCOUNT_FILE).exists():
            logging.error(f"❌ Service account file not found: {SERVICE_ACCOUNT_FILE}")
            logging.error("   Make sure the credentials file is in the current directory")
            return False

        # Load administrative regions once (shared across all variables)
        logging.info("PRELIMINARY: Loading administrative regions")
        gdf, regions = load_administrative_regions()
        
        logging.info("PRELIMINARY: Extracting region metadata")
        region_metadata = get_region_metadata(gdf, regions)

        # Process each variable sequentially with its own cluster
        completed_files = []
        total_successful_regions = 0
        
        for i, spi_variable in enumerate(spi_variables, 1):
            logging.info(f"\n🔄 Processing variable {i}/{len(spi_variables)}: {spi_variable.upper()}")
            
            output_file, successful_count = process_single_variable(
                spi_variable, region_metadata, return_periods, session_id)
                
            if output_file:
                completed_files.append(output_file)
                total_successful_regions += successful_count
                logging.info(f"✅ {spi_variable.upper()} completed successfully")
            else:
                logging.error(f"❌ {spi_variable.upper()} failed")

        # Final summary
        logging.info("=" * 80)
        logging.info("SEQUENTIAL PROCESSING COMPLETED")
        logging.info("=" * 80)
        
        logging.info(f"✅ Variables processed: {len(completed_files)}/{len(spi_variables)}")
        logging.info(f"✅ Total regions processed: {total_successful_regions}")
        logging.info(f"✅ Output files created:")
        for i, filename in enumerate(completed_files, 1):
            logging.info(f"   {i}. {filename}")
        logging.info(f"✅ Log file: {log_file}")
        logging.info(f"✅ Session ID: {session_id}")

        return len(completed_files) > 0

    except KeyboardInterrupt:
        logging.warning("Analysis interrupted by user")
        return False

    except Exception as e:
        logging.error(f"❌ Sequential analysis failed: {e}")
        logging.debug(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)