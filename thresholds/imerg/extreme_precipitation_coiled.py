#!/usr/bin/env python3
"""
Extreme Precipitation Analysis using Coiled and IMERG Data v20250814
===================================================================

This script analyzes extreme precipitation events using IMERG data from Planetary Computer
with distributed processing via Coiled. It supports multiple accumulation periods
and follows the pattern established in icechunk_regionmask_extreme_analysis.py.

Key features:
1. Multiple accumulation periods: 1h, 3h, 6h, 12h, 24h, 48h, 7days (168h)
2. xarray groupby-based rolling accumulation from 30-minute IMERG data
3. Regional extreme value analysis using regionmask
4. Return period analysis using xclim.indices.stats
5. Distributed processing with Coiled Dask clusters
6. Command-line configuration for accumulation periods

Usage:
    # Process single accumulation period
    python extreme_precipitation_coiled.py --accumulation 1
    
    # Process multiple accumulation periods
    python extreme_precipitation_coiled.py --accumulations 1,3,6,24
    
    # Specify custom return periods
    python extreme_precipitation_coiled.py --accumulation 24 --return-periods 2,10,50,100

Based on the approach from icechunk_regionmask_extreme_analysis.py adapted for IMERG precipitation.
"""

import xarray as xr
import numpy as np
import geopandas as gpd
import regionmask
import warnings
import time
import xclim
from xclim.indices import stats
import coiled
import dask
from dask.distributed import Client, get_worker
import logging
import json
import os
from datetime import datetime
import traceback
import sys
import argparse
from pathlib import Path
import pystac_client
import fsspec
import planetary_computer
from typing import List, Dict

warnings.filterwarnings('ignore')

# Configuration
GEOJSON_FILE = "icpac_adm1v3.geojson"
BUFFER_SIZE = 0.25

# Available accumulation periods (hours)
AVAILABLE_ACCUMULATIONS = [1, 3, 6, 12, 24, 48, 168]  # 168h = 7 days

# Logging and session configuration
LOG_DIR = Path("logs")
SESSION_DIR = Path("sessions")


def parse_arguments():
    """Parse command line arguments for accumulation period selection"""
    parser = argparse.ArgumentParser(description='Extreme precipitation analysis with IMERG data')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--accumulation', 
                      type=int,
                      help=f'Single accumulation period in hours. Options: {AVAILABLE_ACCUMULATIONS}')
    
    group.add_argument('--accumulations',
                      type=str,
                      help=f'Comma-separated list of accumulation periods in hours. Options: {AVAILABLE_ACCUMULATIONS}')
    
    parser.add_argument('--return-periods',
                       type=str,
                       default='2,5,10,25,50,100',
                       help='Comma-separated list of return periods (default: 2,5,10,25,50,100)')
    
    args = parser.parse_args()
    
    # Parse accumulation periods
    if args.accumulation:
        accumulations = [args.accumulation]
    else:
        accumulations = [int(acc.strip()) for acc in args.accumulations.split(',')]
    
    # Validate accumulation periods
    valid_accumulations = []
    for acc in accumulations:
        if acc in AVAILABLE_ACCUMULATIONS:
            valid_accumulations.append(acc)
        else:
            print(f"Warning: Skipping invalid accumulation '{acc}h'. Valid options: {AVAILABLE_ACCUMULATIONS}")
    
    if not valid_accumulations:
        print(f"Error: No valid accumulation periods specified. Valid options: {AVAILABLE_ACCUMULATIONS}")
        sys.exit(1)
    
    # Parse return periods
    try:
        return_periods = [int(x.strip()) for x in args.return_periods.split(',')]
    except ValueError:
        print("Error: Invalid return periods format. Using default: [2, 5, 10, 25, 50, 100]")
        return_periods = [2, 5, 10, 25, 50, 100]
    
    return valid_accumulations, return_periods


def setup_logging(log_level=logging.INFO, accumulation=None):
    """Setup comprehensive logging system with file and console output"""
    # Create log directory if it doesn't exist
    LOG_DIR.mkdir(exist_ok=True)

    # Create unique session identifier
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    acc_suffix = f"{accumulation}h" if accumulation else "multi"
    log_file = LOG_DIR / f"extreme_precipitation_v20250814_{acc_suffix}_{session_id}.log"

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


def setup_coiled_cluster_for_accumulation(accumulation_hours, software_env="v7-geosfm-rm-x", n_workers=3):
    """Setup fresh Coiled Dask cluster for a specific accumulation period"""
    
    logging.info(f"Creating fresh cluster for {accumulation_hours}h accumulation")
    
    cluster_name = f"precip-{accumulation_hours}h-analysis-v20250814-{datetime.now().strftime('%m%d-%H%M')}"

    try:
        cluster = coiled.Cluster(
            name=cluster_name,
            software=software_env,
            n_workers=n_workers,
            scheduler_vm_types=["n2-standard-4"],
            worker_vm_types="n2-standard-8",
            region="us-east1",
            arm=False,
            compute_purchase_option="spot",
            workspace='geosfm')

        client = Client(cluster)

        # Verify cluster is ready
        worker_info = client.scheduler_info().get('workers', {})
        actual_workers = len(worker_info)

        logging.info(f"✅ New Coiled cluster ready for {accumulation_hours}h: {client.dashboard_link}")
        logging.info(f"   Cluster name: {cluster_name}")
        logging.info(f"   Workers: {actual_workers}/{n_workers}")
        logging.info(f"   VM type: n2-standard-4")
        logging.info(f"   Region: us-east1")

        return client, cluster

    except Exception as e:
        logging.error(f"❌ Failed to setup Coiled cluster for {accumulation_hours}h: {e}")
        raise


def load_imerg_dataset_worker():
    """Load IMERG data from Planetary Computer on worker - subset to East Africa region"""
    
    try:
        # Load IMERG data
        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )
        asset = catalog.get_collection("gpm-imerg-hhr").assets["zarr-abfs"]
        fs = fsspec.get_mapper(asset.href, **asset.extra_fields["xarray:storage_options"])
        ds = xr.open_zarr(fs, **asset.extra_fields["xarray:open_kwargs"])
        
        # Define East Africa bounds (based on administrative regions)
        east_africa_bounds = {
            'lon_min': 21.84,
            'lon_max': 51.42,
            'lat_min': -11.75,
            'lat_max': 23.15
        }
        
        # Add buffer to ensure we capture all boundary regions
        buffer = BUFFER_SIZE
        lon_min = east_africa_bounds['lon_min'] - buffer
        lon_max = east_africa_bounds['lon_max'] + buffer
        lat_min = east_africa_bounds['lat_min'] - buffer
        lat_max = east_africa_bounds['lat_max'] + buffer
        
        # Subset the dataset to East Africa region
        lat_values = ds.lat.values
        if lat_values[0] < lat_values[-1]:  # ascending order
            ds_subset = ds.sel(
                lon=slice(lon_min, lon_max),
                lat=slice(lat_min, lat_max)
            )
        else:  # descending order
            ds_subset = ds.sel(
                lon=slice(lon_min, lon_max),
                lat=slice(lat_max, lat_min)
            )
        
        return ds_subset
        
    except Exception as e:
        raise RuntimeError(f"Failed to load IMERG dataset: {e}")


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

        # Fix invalid geometries
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


def process_region_extreme_precipitation_worker(region_metadata, accumulation_hours, return_periods):
    """
    Process extreme precipitation for a single region on Dask worker
    
    This function runs on individual workers and:
    1. Loads IMERG data subset to East Africa
    2. Calculates rolling accumulation 
    3. Applies regionmask filtering
    4. Performs extreme value analysis
    5. Returns results to main process
    """
    try:
        import xarray as xr
        import numpy as np
        import regionmask
        import geopandas as gpd
        from shapely.geometry import shape
        from xclim.indices import stats
        from dask.distributed import get_worker
        import warnings
        warnings.filterwarnings('ignore')

        # Get worker information
        worker = get_worker()
        worker_id = worker.address

        region_id = region_metadata['region_id']
        region_name = region_metadata['region_name']
        
        # Load IMERG dataset on worker
        dataset = load_imerg_dataset_worker()
        
        # Calculate number of 30-minute timesteps for the accumulation period
        timesteps_per_accumulation = accumulation_hours * 2  # 2 timesteps per hour
        
        # Get precipitation variable
        precip_vars = ['precipitationCal', 'precipitation']
        precip_var = None
        for var in precip_vars:
            if var in dataset.data_vars:
                precip_var = var
                break
        
        if precip_var is None:
            raise ValueError(f"No precipitation variable found. Available: {list(dataset.data_vars)}")
        
        # Calculate rolling accumulation
        precip_data = dataset[precip_var]
        accumulated = precip_data.rolling(time=timesteps_per_accumulation, center=False).sum()
        
        # Extract annual maxima using groupby
        annual_maxima = accumulated.groupby('time.year').max('time')
        
        # Create region geometry and mask on worker
        region_geom = shape(region_metadata['geometry'])
        
        # Get coordinate arrays
        lons = dataset.lon.values
        lats = dataset.lat.values
        
        # Create mask for this specific region
        region_gdf = gpd.GeoDataFrame([{'geometry': region_geom, 'region_id': region_id}])
        region_mask = regionmask.mask_geopandas(region_gdf.geometry, lons, lats)
        
        # Apply regional mask to get region-specific data
        region_data = annual_maxima.where(region_mask == 0)  # regionmask uses 0 for first region
        
        # Calculate regional mean (spatial average over the region)
        region_annual_maxima = region_data.mean(dim=['lon', 'lat'], skipna=True)
        
        # Extract values and remove NaNs
        valid_data = region_annual_maxima.values
        valid_data = valid_data[~np.isnan(valid_data)]

        if len(valid_data) < 10:  # Need sufficient data points
            return {
                'region_id': region_id,
                'region_name': region_name,
                'accumulation_hours': accumulation_hours,
                'return_periods': return_periods,
                'return_levels': [np.nan] * len(return_periods),
                'status': 'insufficient_data',
                'worker_id': worker_id,
                'n_years': len(valid_data)
            }

        # Convert to xarray DataArray for xclim compatibility
        precip_xr = xr.DataArray(valid_data, dims=['time'])
        precip_xr.attrs['units'] = 'mm'

        # Calculate return levels using xclim's fa() function for precipitation extremes
        fa_result = stats.fa(precip_xr,
                           t=return_periods,
                           dist='genextreme',  # Generalized extreme value distribution
                           mode='max')  # For precipitation maxima

        return {
            'region_id': region_id,
            'region_name': region_name,
            'accumulation_hours': accumulation_hours,
            'return_periods': return_periods,
            'return_levels': fa_result.values.tolist(),
            'status': 'success',
            'worker_id': worker_id,
            'n_years': len(valid_data),
            'max_observed': float(np.max(valid_data)),
            'mean_annual_max': float(np.mean(valid_data))
        }

    except Exception as e:
        import traceback
        return {
            'region_id': region_metadata.get('region_id', -1),
            'region_name': region_metadata.get('region_name', 'unknown'),
            'accumulation_hours': accumulation_hours,
            'return_periods': return_periods,
            'return_levels': [np.nan] * len(return_periods),
            'status': f'error: {str(e)}',
            'worker_id': getattr(get_worker(), 'address', 'unknown') if 'get_worker' in locals() else 'unknown',
            'n_years': 0,
            'traceback': traceback.format_exc()
        }


def calculate_return_periods_for_accumulation(region_metadata, client, accumulation_hours, return_periods):
    """Calculate return periods for a specific accumulation period using distributed processing"""
    logging.info(f"=" * 60)
    logging.info(f"CALCULATING RETURN PERIODS FOR {accumulation_hours}H ACCUMULATION")
    logging.info(f"=" * 60)

    logging.info(f"Return periods to calculate: {return_periods}")
    logging.info(f"Processing {len(region_metadata)} regions")

    start_time = time.time()

    try:
        # Submit tasks to workers
        logging.info(f"Submitting {len(region_metadata)} tasks to workers...")
        
        futures = []
        for region_meta in region_metadata:
            future = client.submit(
                process_region_extreme_precipitation_worker,
                region_meta,
                accumulation_hours,
                return_periods
            )
            futures.append(future)

        # Collect results with progress tracking
        results = []
        completed = 0

        for i, future in enumerate(futures):
            try:
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
                    'accumulation_hours': accumulation_hours,
                    'return_periods': return_periods,
                    'return_levels': [np.nan] * len(return_periods),
                    'status': f'error: {str(e)}',
                    'worker_id': 'unknown',
                    'n_years': 0
                })

    except Exception as e:
        logging.error(f"Failed to process regions for {accumulation_hours}h: {e}")
        raise

    computation_time = time.time() - start_time

    # Process results
    successful_regions = [r for r in results if r['status'] == 'success']
    failed_regions = [r for r in results if r['status'] != 'success']

    logging.info(
        f"✅ {accumulation_hours}h calculation completed in {computation_time:.2f} seconds"
    )
    logging.info(f"   Successful regions: {len(successful_regions)}")
    logging.info(f"   Failed regions: {len(failed_regions)}")
    logging.info(
        f"   Processing rate: {len(region_metadata)/computation_time:.2f} regions/second"
    )

    # Display sample results
    if successful_regions:
        sample = successful_regions[0]
        logging.info(f"\nSample results for {sample['region_name']} ({accumulation_hours}h accumulation):")
        for T, level in zip(sample['return_periods'], sample['return_levels']):
            logging.info(f"   {T:3d}-year return level: {level:.2f} mm")

    return results


def cleanup_cluster(client, cluster, force=False):
    """Gracefully cleanup Dask cluster"""
    logging.info("Starting cluster cleanup...")

    try:
        if client:
            try:
                client.cancel(client.futures, force=force)
                logging.info("Cancelled running tasks")
            except Exception as e:
                logging.warning(f"Error cancelling tasks: {e}")

            try:
                client.close(timeout=10)
                logging.info("Client connection closed")
            except Exception as e:
                logging.warning(f"Error closing client: {e}")

        if cluster:
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


def save_results(results, session_id, accumulation_hours, output_file=None):
    """Save results to JSON file with enhanced metadata"""
    if output_file is None:
        output_file = f"extreme_precipitation_{accumulation_hours}h_results_v20250814_{session_id}.json"

    logging.info("=" * 50)
    logging.info(f"SAVING {accumulation_hours}H RESULTS")
    logging.info("=" * 50)

    try:
        # Add metadata to results
        metadata = {
            'session_id': session_id,
            'created_at': datetime.now().isoformat(),
            'script_version': 'extreme_precipitation_v20250814',
            'accumulation_hours': accumulation_hours,
            'configuration': {
                'geojson_file': GEOJSON_FILE,
                'buffer_size': BUFFER_SIZE
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

        logging.info(f"✅ {accumulation_hours}h results saved to {output_file}")
        logging.info(f"   Total regions processed: {len(results)}")
        logging.info(f"   Successful: {metadata['summary']['successful_regions']}")
        logging.info(f"   Failed: {metadata['summary']['failed_regions']}")
        
        return output_file

    except Exception as e:
        logging.error(f"❌ Failed to save {accumulation_hours}h results: {e}")
        logging.debug(traceback.format_exc())
        return None


def process_single_accumulation(accumulation_hours, region_metadata, return_periods, session_id, max_retries=2):
    """Process a single accumulation period with its own cluster and retry capability"""
    
    logging.info(f"\n{'='*80}")
    logging.info(f"PROCESSING {accumulation_hours}H ACCUMULATION")
    logging.info(f"{'='*80}")
    
    for attempt in range(max_retries + 1):
        if attempt > 0:
            logging.info(f"🔄 Retry attempt {attempt}/{max_retries} for {accumulation_hours}h")
            time.sleep(30)  # Wait before retry
        
        cluster = None
        client = None
        success = False
        
        try:
            # Step 1: Setup fresh cluster for this accumulation period
            logging.info(f"STEP 1: Setting up Dask cluster for {accumulation_hours}h")
            client, cluster = setup_coiled_cluster_for_accumulation(accumulation_hours)

            # Step 2: Calculate return periods for this accumulation period
            logging.info(f"STEP 2: Calculating return periods for {accumulation_hours}h")
            results = calculate_return_periods_for_accumulation(
                region_metadata, client, accumulation_hours, return_periods)

            # Step 3: Save results for this accumulation period
            logging.info(f"STEP 3: Saving {accumulation_hours}h results")
            output_file = save_results(results, session_id, accumulation_hours)
            
            successful_regions = [r for r in results if r['status'] == 'success']
            logging.info(f"✅ {accumulation_hours}h completed: {len(successful_regions)} regions processed")
            
            success = True
            return output_file, len(successful_regions)

        except Exception as e:
            logging.error(f"❌ {accumulation_hours}h processing failed (attempt {attempt + 1}): {e}")
            logging.debug(traceback.format_exc())
            
            # Don't retry on final attempt
            if attempt == max_retries:
                return None, 0

        finally:
            # Cleanup cluster for this accumulation period
            logging.info(f"Cleaning up cluster for {accumulation_hours}h...")
            try:
                cleanup_cluster(client, cluster, force=not success)
                time.sleep(10)  # Wait after cleanup
            except Exception as e:
                logging.error(f"Error during {accumulation_hours}h cleanup: {e}")
    
    return None, 0


def main():
    """Main execution function with sequential accumulation period processing"""

    # Parse command line arguments
    accumulations, return_periods = parse_arguments()
    
    # Initialize logging system
    session_id, log_file = setup_logging(accumulation="_".join(map(str, accumulations)))

    logging.info("=" * 80)
    logging.info("EXTREME PRECIPITATION ANALYSIS WITH COILED v20250814")
    logging.info("=" * 80)
    logging.info(f"Session ID: {session_id}")
    logging.info(f"Accumulation periods: {accumulations}h")
    logging.info(f"Return periods: {return_periods}")
    logging.info(f"Processing mode: Sequential (separate cluster per accumulation)")

    try:
        # Load administrative regions once (shared across all accumulations)
        logging.info("PRELIMINARY: Loading administrative regions")
        gdf, regions = load_administrative_regions()
        
        logging.info("PRELIMINARY: Extracting region metadata")
        region_metadata = get_region_metadata(gdf, regions)

        # Process each accumulation period sequentially with its own cluster
        completed_files = []
        total_successful_regions = 0
        
        for i, accumulation_hours in enumerate(accumulations, 1):
            logging.info(f"\n🔄 Processing accumulation {i}/{len(accumulations)}: {accumulation_hours}h")
            
            output_file, successful_count = process_single_accumulation(
                accumulation_hours, region_metadata, return_periods, session_id)
                
            if output_file:
                completed_files.append(output_file)
                total_successful_regions += successful_count
                logging.info(f"✅ {accumulation_hours}h completed successfully")
            else:
                logging.error(f"❌ {accumulation_hours}h failed")

        # Final summary
        logging.info("=" * 80)
        logging.info("SEQUENTIAL PROCESSING COMPLETED")
        logging.info("=" * 80)
        
        logging.info(f"✅ Accumulations processed: {len(completed_files)}/{len(accumulations)}")
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
