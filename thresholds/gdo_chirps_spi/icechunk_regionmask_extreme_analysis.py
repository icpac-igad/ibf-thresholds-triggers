#!/usr/bin/env python3
"""
Enhanced Icechunk-based Regional Extreme Value Analysis with Logging and Cluster Management
========================================================================================

This enhanced script integrates:
1. Comprehensive logging system with file output
2. Dask cluster reuse and failure recovery
3. Icechunk data loading for SPI9 datasets
4. Regionmask functionality for administrative regions
5. Coiled Dask cluster management with persistence
6. Extreme value analysis using xclim for each region

Features:
- Session-based logging with unique log files
- Cluster connection caching and reuse
- Enhanced error handling and recovery
- Progress tracking and performance metrics
- Graceful cluster cleanup and disentanglement

Usage:
    python icechunk_regionmask_extreme_analysis_enhanced.py
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
from dask.distributed import Client
from dask.diagnostics import ProgressBar
import logging
import json
import os
from datetime import datetime
import pickle
from pathlib import Path
import traceback
import sys

warnings.filterwarnings('ignore')

# Configuration
BASE_PREFIX = "t2spi1_east_africa_icechunk"
SPI_TYPE = "spi9"
BUCKET_NAME = "cdi_arco"
SERVICE_ACCOUNT_FILE = "coiled-data-e4drr_202505.json"
GEOJSON_FILE = "icpac_adm1v3.geojson"
BUFFER_SIZE = 0.25

# Logging and session configuration
LOG_DIR = Path("logs")
SESSION_DIR = Path("sessions")
CLUSTER_CACHE_FILE = "cluster_cache.pkl"


def setup_logging(log_level=logging.INFO):
    """Setup comprehensive logging system with file and console output"""
    # Create log directory if it doesn't exist
    LOG_DIR.mkdir(exist_ok=True)
    
    # Create unique session identifier
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"icechunk_analysis_{session_id}.log"
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(funcName)-25s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = logging.Formatter(
        '%(levelname)-8s | %(message)s'
    )
    
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
            
        logging.info(f"Found cached cluster info: {cluster_info['cluster_name']}")
        return cluster_info
        
    except Exception as e:
        logging.warning(f"Failed to load cluster info: {e}")
        return None


def test_cluster_connection(client):
    """Test if cluster connection is still active"""
    try:
        worker_info = client.scheduler_info().get('workers', {})
        if len(worker_info) > 0:
            logging.info(f"Cluster connection active with {len(worker_info)} workers")
            return True
        else:
            logging.warning("Cluster connection exists but no workers available")
            return False
    except Exception as e:
        logging.warning(f"Cluster connection test failed: {e}")
        return False


def setup_coiled_cluster(software_env="v3-geosfm-rm-x", n_workers=3, reuse_existing=True, session_id=None):
    """Setup Coiled Dask cluster with reuse capability and error recovery"""
    
    # Try to reuse existing cluster first
    if reuse_existing:
        cluster_info = load_cluster_info()
        if cluster_info:
            try:
                logging.info(f"Attempting to reconnect to existing cluster: {cluster_info['cluster_name']}")
                
                # Try to get existing cluster
                cluster = coiled.Cluster(cluster_info['cluster_name'])
                client = cluster.get_client()
                
                # Test connection
                if test_cluster_connection(client):
                    logging.info(f"✅ Reusing existing cluster: {client.dashboard_link}")
                    return client, cluster
                else:
                    logging.info("Existing cluster not responsive, creating new one...")
                    client.close()
                    cluster.close()
                    
            except Exception as e:
                logging.warning(f"Failed to reuse existing cluster: {e}")
    
    logging.info(f"Creating new Coiled cluster with {n_workers} workers...")
    
    cluster_name = f"spi-extreme-analysis-{datetime.now().strftime('%m%d-%H%M')}"
    
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
            workspace='geosfm',
            worker_options={
                "security": {
                    "key_path": SERVICE_ACCOUNT_FILE
                }
            }
        )

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


def load_icechunk_spi_data():
    """Load SPI9 data from Icechunk repository with enhanced logging"""
    logging.info("=" * 70)
    logging.info("LOADING SPI9 DATA FROM ICECHUNK")
    logging.info("=" * 70)

    # Construct repository prefix for SPI9
    repo_prefix = f"{BASE_PREFIX}_{SPI_TYPE}"

    logging.info(f"Repository prefix: {repo_prefix}")
    logging.info(f"Bucket: {BUCKET_NAME}")

    start_time = time.time()
    
    try:
        # Setup storage connection
        logging.debug("Setting up GCS storage connection...")
        storage = icechunk.gcs_storage(
            bucket=BUCKET_NAME,
            prefix=repo_prefix,
            service_account_file=SERVICE_ACCOUNT_FILE)

        # Open repository
        logging.debug("Opening Icechunk repository...")
        repo = icechunk.Repository.open(storage)
        session = repo.readonly_session("main")

        # Load data from specific Zarr group
        group_name = f"{SPI_TYPE}_data"  # e.g., "spi9_data"
        logging.debug(f"Loading data from group: {group_name}")
        dataset = xr.open_zarr(session.store, group=group_name)

        load_time = time.time() - start_time

        logging.info(f"✅ Successfully loaded {SPI_TYPE.upper()} dataset in {load_time:.2f} seconds")
        logging.info(f"   Shape: {dict(dataset.sizes)}")
        logging.info(f"   Variables: {list(dataset.data_vars)}")
        logging.info(f"   Time range: {dataset.time.min().values} to {dataset.time.max().values}")
        
        # Log memory usage if available
        try:
            memory_mb = dataset.nbytes / (1024 * 1024)
            logging.info(f"   Dataset size: {memory_mb:.1f} MB")
        except:
            pass

        return dataset

    except Exception as e:
        logging.error(f"❌ Failed to load Icechunk data: {e}")
        logging.debug(traceback.format_exc())
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

        logging.info(f"✅ Loaded {len(gdf)} administrative regions in {load_time:.2f} seconds")
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
            logging.info(f"   Geometry fixes completed in {fix_time:.2f} seconds")

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


def create_region_mask(gdf, dataset):
    """Create region mask for the dataset using regionmask.mask_geopandas with enhanced logging"""
    logging.info("=" * 70)
    logging.info("CREATING REGION MASK")
    logging.info("=" * 70)

    start_time = time.time()

    try:
        # Extract coordinates
        lons = dataset.lon.values
        lats = dataset.lat.values

        logging.info(f"   Dataset coordinates: {len(lons)} lons × {len(lats)} lats")
        logging.info(f"   Coordinate ranges: lon [{lons.min():.2f}, {lons.max():.2f}], lat [{lats.min():.2f}, {lats.max():.2f}]")

        # Create mask using regionmask.mask_geopandas
        logging.debug("Attempting to create region mask...")
        try:
            mask = regionmask.mask_geopandas(gdf.geometry, lons, lats)
        except ValueError as e:
            if "overlapping regions" in str(e):
                logging.warning("Detected overlapping regions, using overlap=False...")
                mask = regionmask.mask_geopandas(gdf.geometry,
                                                 lons,
                                                 lats,
                                                 overlap=False)
            else:
                raise

        mask_time = time.time() - start_time

        logging.info(f"✅ Created region mask in {mask_time:.2f} seconds")
        logging.info(f"   Mask shape: {mask.shape}")
        
        unique_regions = np.unique(mask.values[~np.isnan(mask.values)])
        logging.info(f"   Unique regions in mask: {len(unique_regions)}")
        logging.debug(f"   Region IDs: {unique_regions}")

        return mask

    except Exception as e:
        logging.error(f"❌ Failed to create region mask: {e}")
        logging.debug(traceback.format_exc())
        raise


def extract_annual_extremes(spi_data):
    """Extract annual extremes using xclim methods with enhanced logging"""
    logging.info("=" * 70)
    logging.info("EXTRACTING ANNUAL EXTREMES")
    logging.info("=" * 70)

    start_time = time.time()

    try:
        # Use xclim's select_resample_op for annual minima (drought analysis)
        logging.info("Extracting annual minima for drought analysis...")
        
        # Add units attribute for xclim compatibility (SPI is dimensionless)
        spi_data.attrs['units'] = '1'

        annual_minima = select_resample_op(
            spi_data,
            op='min',
            freq='YS',  # Annual frequency starting in January
        )

        extraction_time = time.time() - start_time
        
        logging.info(f"✅ Annual extremes extracted in {extraction_time:.2f} seconds")
        logging.info(f"   Shape: {dict(annual_minima.sizes)}")
        logging.info(f"   Time range: {annual_minima.time.min().values} to {annual_minima.time.max().values}")
        
        # Log statistical summary
        try:
            min_val = float(annual_minima.min().values)
            max_val = float(annual_minima.max().values)
            mean_val = float(annual_minima.mean().values)
            logging.info(f"   Value range: [{min_val:.3f}, {max_val:.3f}], mean: {mean_val:.3f}")
        except:
            pass

        return annual_minima

    except Exception as e:
        logging.error(f"❌ Failed to extract annual extremes: {e}")
        logging.debug(traceback.format_exc())
        raise


def process_region_for_return_periods(annual_minima, region_mask, region_id, region_name, return_periods):
    """Process a single region for return period calculation - runs on worker with logging"""
    import logging
    
    try:
        # Extract regional data on the worker
        region_data = annual_minima.where(region_mask == region_id)
        
        # Compute values on the worker (avoiding serialization issues)
        region_values = region_data.values  # Get numpy array directly
        
        # Work with numpy array directly
        if isinstance(region_values, np.ndarray):
            # Remove spatial dimensions and work with time series
            if region_values.ndim > 1:
                # Take spatial mean over the region, ignoring NaNs
                region_ts = np.nanmean(region_values, axis=tuple(range(1, region_values.ndim)))
            else:
                region_ts = region_values
        else:
            # Fallback
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
                'n_years': len(valid_data)
            }

        # For drought analysis, convert minima to maxima (negative values)
        drought_data = -1 * valid_data

        # Convert to xarray for xclim compatibility
        import xarray as xr
        from xclim.indices import stats
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
            'n_years': len(valid_data)
        }

    except Exception as e:
        return {
            'region_id': region_id,
            'region_name': region_name,
            'return_periods': return_periods,
            'return_levels': [np.nan] * len(return_periods),
            'status': f'error: {str(e)}',
            'n_years': 0
        }


def calculate_return_periods_per_region(annual_minima, region_mask, gdf, client, 
                                        return_periods=[2, 5, 10, 25, 50, 100]):
    """Calculate return periods for each administrative region using distributed computing with enhanced logging"""
    logging.info("=" * 70)
    logging.info("CALCULATING RETURN PERIODS PER REGION")
    logging.info("=" * 70)

    logging.info(f"Return periods to calculate: {return_periods}")
    logging.info(f"Using Dask client: {client}")
    
    # Ensure data is chunked for distributed processing
    logging.debug(f"Data dimensions: {annual_minima.dims}")
    logging.debug(f"Current chunks: {annual_minima.chunks}")
    
    # Determine coordinate names and rechunk appropriately
    if 'latitude' in annual_minima.dims:
        chunk_dict = {'time': -1, 'latitude': 50, 'longitude': 50}
    elif 'lat' in annual_minima.dims:
        chunk_dict = {'time': -1, 'lat': 50, 'lon': 50}
    else:
        chunk_dict = {'time': -1}
    
    logging.info(f"Rechunking with: {chunk_dict}")
    annual_minima = annual_minima.chunk(chunk_dict)
    logging.debug(f"New chunks: {annual_minima.chunks}")
    
    # Check cluster status
    try:
        worker_info = client.scheduler_info().get('workers', {})
        logging.info(f"Cluster status: Connected with {len(worker_info)} workers")
        
        if len(worker_info) == 0:
            logging.warning("No workers detected. Tasks may run on scheduler.")
            
    except Exception as e:
        logging.error(f"Cluster connection issue: {e}")
        raise

    # Get unique region IDs from the mask
    unique_regions = np.unique(region_mask.values[~np.isnan(region_mask.values)])
    logging.info(f"Processing {len(unique_regions)} regions...")

    start_time = time.time()
    
    # Prepare tasks for distributed computation
    tasks = []
    
    for region_id in unique_regions:
        region_id = int(region_id)

        # Get region name from GeoDataFrame
        try:
            if 'shapeName' in gdf.columns:
                region_name = gdf.iloc[region_id]['shapeName']
            elif 'GID_1' in gdf.columns:
                region_name = gdf.iloc[region_id]['GID_1']
            else:
                region_name = f"Region_{region_id}"
        except:
            region_name = f"Region_{region_id}"

        # Submit the region processing as a delayed task
        task = dask.delayed(process_region_for_return_periods)(
            annual_minima, region_mask, region_id, region_name, return_periods
        )
        tasks.append(task)

    # Compute all tasks in parallel using the client
    logging.info(f"Computing {len(tasks)} return period tasks in parallel...")
    
    try:
        # Use client.compute to force execution on the cluster
        futures = client.compute(tasks, sync=False)
        
        # Wait for completion with progress tracking
        results = []
        completed = 0
        
        for i, future in enumerate(futures):
            try:
                result = future.result()
                results.append(result)
                completed += 1
                
                if completed % 10 == 0 or completed == len(futures):
                    progress = (completed / len(futures)) * 100
                    logging.info(f"Progress: {completed}/{len(futures)} regions ({progress:.1f}%)")
                    
            except Exception as e:
                logging.error(f"Task {i} failed: {e}")
                results.append({
                    'region_id': -1,
                    'region_name': 'failed',
                    'return_periods': return_periods,
                    'return_levels': [np.nan] * len(return_periods),
                    'status': f'error: {str(e)}',
                    'n_years': 0
                })

    except Exception as e:
        logging.error(f"Failed to compute tasks: {e}")
        raise

    computation_time = time.time() - start_time

    # Process results
    successful_regions = [r for r in results if r['status'] == 'success']
    failed_regions = [r for r in results if r['status'] != 'success']

    logging.info(f"✅ Return period calculation completed in {computation_time:.2f} seconds")
    logging.info(f"   Successful regions: {len(successful_regions)}")
    logging.info(f"   Failed regions: {len(failed_regions)}")
    logging.info(f"   Processing rate: {len(unique_regions)/computation_time:.2f} regions/second")

    # Log failure details if any
    if failed_regions:
        for failed in failed_regions[:5]:  # Show first 5 failures
            logging.warning(f"   Failed region {failed['region_id']}: {failed['status']}")

    # Display sample results
    if successful_regions:
        sample = successful_regions[0]
        logging.info(f"\nSample results for {sample['region_name']}:")
        for T, level in zip(sample['return_periods'], sample['return_levels']):
            logging.info(f"   {T:3d}-year drought: SPI = {level:.3f}")

    return results


def save_results(results, session_id, output_file=None):
    """Save results to JSON file with enhanced metadata"""
    if output_file is None:
        output_file = f"extreme_value_analysis_results_{session_id}.json"

    logging.info("=" * 70)
    logging.info("SAVING RESULTS")
    logging.info("=" * 70)

    try:
        # Add metadata to results
        metadata = {
            'session_id': session_id,
            'created_at': datetime.now().isoformat(),
            'script_version': 'enhanced_v1.0',
            'configuration': {
                'base_prefix': BASE_PREFIX,
                'spi_type': SPI_TYPE,
                'bucket_name': BUCKET_NAME,
                'geojson_file': GEOJSON_FILE
            },
            'summary': {
                'total_regions': len(results),
                'successful_regions': len([r for r in results if r['status'] == 'success']),
                'failed_regions': len([r for r in results if r['status'] != 'success'])
            }
        }
        
        output_data = {
            'metadata': metadata,
            'results': results
        }

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)

        logging.info(f"✅ Results saved to {output_file}")
        logging.info(f"   Total regions processed: {len(results)}")
        logging.info(f"   Successful: {metadata['summary']['successful_regions']}")
        logging.info(f"   Failed: {metadata['summary']['failed_regions']}")

    except Exception as e:
        logging.error(f"❌ Failed to save results: {e}")
        logging.debug(traceback.format_exc())


def main():
    """Main execution function with enhanced logging and error recovery"""
    
    # Initialize logging system
    session_id, log_file = setup_logging()
    
    logging.info("=" * 70)
    logging.info("ICECHUNK REGIONMASK EXTREME VALUE ANALYSIS - ENHANCED VERSION")
    logging.info("=" * 70)
    logging.info(f"Session ID: {session_id}")

    cluster = None
    client = None
    success = False

    try:
        # Step 1: Setup Coiled Dask cluster
        logging.info("STEP 1: Setting up Dask cluster")
        client, cluster = setup_coiled_cluster(session_id=session_id)

        # Step 2: Load SPI9 data from Icechunk
        logging.info("STEP 2: Loading SPI data from Icechunk")
        dataset = load_icechunk_spi_data()

        # Get the SPI variable (adjust variable name as needed)
        spi_var = None
        for var in dataset.data_vars:
            if 'spi' in var.lower() or 'spc' in var.lower():
                spi_var = var
                break

        if spi_var is None:
            raise ValueError(f"No SPI variable found. Available variables: {list(dataset.data_vars)}")

        spi_data = dataset[spi_var]
        if 'band' in spi_data.dims:
            spi_data = spi_data.squeeze('band')

        logging.info(f"Using SPI variable: {spi_var}")

        # Step 3: Load administrative regions
        logging.info("STEP 3: Loading administrative regions")
        gdf, regions = load_administrative_regions()

        # Step 4: Create region mask
        logging.info("STEP 4: Creating region mask")
        region_mask = create_region_mask(gdf, dataset)

        # Step 5: Extract annual extremes
        logging.info("STEP 5: Extracting annual extremes")
        annual_minima = extract_annual_extremes(spi_data)

        # Step 6: Calculate return periods per region
        logging.info("STEP 6: Calculating return periods per region")
        results = calculate_return_periods_per_region(
            annual_minima,
            region_mask,
            gdf,
            client,
            return_periods=[2, 5, 10, 25, 50, 100]
        )

        # Step 7: Save results
        logging.info("STEP 7: Saving results")
        save_results(results, session_id)

        # Summary
        logging.info("=" * 70)
        logging.info("ANALYSIS COMPLETED SUCCESSFULLY")
        logging.info("=" * 70)

        successful_regions = [r for r in results if r['status'] == 'success']
        logging.info(f"✅ Processed {len(successful_regions)} regions successfully")
        logging.info(f"✅ Results saved for extreme value analysis")
        logging.info(f"✅ Log file: {log_file}")
        logging.info(f"✅ Session ID: {session_id}")
        
        success = True

    except KeyboardInterrupt:
        logging.warning("Analysis interrupted by user")
        success = False
        
    except Exception as e:
        logging.error(f"❌ Analysis failed: {e}")
        logging.debug(traceback.format_exc())
        success = False

    finally:
        # Cleanup cluster
        logging.info("Performing cleanup...")
        try:
            cleanup_cluster(client, cluster, force=not success)
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")
        
        logging.info(f"Analysis session {session_id} completed")
        
        if success:
            logging.info("✅ Session completed successfully")
        else:
            logging.error("❌ Session completed with errors")

    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)