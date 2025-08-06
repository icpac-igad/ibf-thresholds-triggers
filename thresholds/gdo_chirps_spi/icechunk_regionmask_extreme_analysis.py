#!/usr/bin/env python3
"""
Icechunk-based Regional Extreme Value Analysis using RegionMask
================================================================

This script integrates:
1. Icechunk data loading for SPI9 datasets
2. Regionmask functionality for administrative regions
3. Coiled Dask cluster for distributed computing
4. Extreme value analysis using xclim for each region

Usage:
    python icechunk_regionmask_extreme_analysis.py
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

warnings.filterwarnings('ignore')

# Configuration
BASE_PREFIX = "t2spi1_east_africa_icechunk"
SPI_TYPE = "spi9"
BUCKET_NAME = "cdi_arco"
SERVICE_ACCOUNT_FILE = "coiled-data-e4drr_202505.json"
GEOJSON_FILE = "icpac_adm1v3.geojson"
BUFFER_SIZE = 0.25


def setup_coiled_cluster(software_env="v3-geosfm-rm-x", n_workers=3):
    """Setup Coiled Dask cluster with specified configuration"""
    print(f"Setting up Coiled cluster with {n_workers} workers...")

    try:
        cluster = coiled.Cluster(
            name="spi-extreme-analysis",  
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
                    "key_path": SERVICE_ACCOUNT_FILE  # Pass credentials to workers for Icechunk
                }
            }
        )

        client = Client(cluster)
        
        print(f"✅ Coiled cluster ready: {client.dashboard_link}")
        print(f"   Workers: {n_workers}")
        print(f"   VM type: n2-standard-4")
        print(f"   Region: us-east1")

        return client, cluster

    except Exception as e:
        print(f"❌ Failed to setup Coiled cluster: {e}")
        raise


def load_icechunk_spi_data():
    """Load SPI9 data from Icechunk repository"""
    print(f"\n{'='*70}")
    print("LOADING SPI9 DATA FROM ICECHUNK")
    print(f"{'='*70}")

    # Construct repository prefix for SPI9
    repo_prefix = f"{BASE_PREFIX}_{SPI_TYPE}"

    print(f"Repository prefix: {repo_prefix}")
    print(f"Bucket: {BUCKET_NAME}")

    try:
        # Setup storage connection
        storage = icechunk.gcs_storage(
            bucket=BUCKET_NAME,
            prefix=repo_prefix,
            service_account_file=SERVICE_ACCOUNT_FILE)

        # Open repository
        repo = icechunk.Repository.open(storage)
        session = repo.readonly_session("main")

        # Load data from specific Zarr group
        group_name = f"{SPI_TYPE}_data"  # e.g., "spi9_data"
        dataset = xr.open_zarr(session.store, group=group_name)

        print(f"✅ Successfully loaded {SPI_TYPE.upper()} dataset")
        print(f"   Shape: {dict(dataset.sizes)}")
        print(f"   Variables: {list(dataset.data_vars)}")
        print(
            f"   Time range: {dataset.time.min().values} to {dataset.time.max().values}"
        )

        return dataset

    except Exception as e:
        print(f"❌ Failed to load Icechunk data: {e}")
        raise


def load_administrative_regions():
    """Load administrative regions and create regionmask"""
    print(f"\n{'='*70}")
    print("LOADING ADMINISTRATIVE REGIONS")
    print(f"{'='*70}")

    try:
        # Load GeoJSON file
        gdf = gpd.read_file(GEOJSON_FILE)
        gdf['region_idx'] = np.arange(len(gdf))

        print(f"✅ Loaded {len(gdf)} administrative regions")
        print(f"   Columns: {list(gdf.columns)}")

        # Check for the correct name column
        name_col = 'GID_1' if 'GID_1' in gdf.columns else 'shapeName'
        id_col = 'region_idx' if 'region_idx' in gdf.columns else 'shapeID'

        print(f"   Sample regions: {gdf[name_col].head().tolist()}")

        # Fix invalid geometries that can cause overlap detection issues
        invalid_count = (~gdf.geometry.is_valid).sum()
        if invalid_count > 0:
            print(f"   Fixing {invalid_count} invalid geometries...")
            gdf.geometry = gdf.geometry.buffer(0)

        # Create regionmask using the appropriate columns
        regions = regionmask.from_geopandas(gdf,
                                            names=name_col,
                                            abbrevs=id_col,
                                            name="icpac_regions")

        print(f"✅ Created regionmask with {len(regions)} regions")

        return gdf, regions

    except Exception as e:
        print(f"❌ Failed to load regions: {e}")
        raise


def create_region_mask(gdf, dataset):
    """Create region mask for the dataset using regionmask.mask_geopandas"""
    print(f"\n{'='*70}")
    print("CREATING REGION MASK")
    print(f"{'='*70}")

    try:
        # Extract coordinates
        lons = dataset.lon.values
        lats = dataset.lat.values

        print(f"   Dataset coordinates: {len(lons)} lons × {len(lats)} lats")
        print(
            f"   Coordinate ranges: lon [{lons.min():.2f}, {lons.max():.2f}], lat [{lats.min():.2f}, {lats.max():.2f}]"
        )

        # Create mask using regionmask.mask_geopandas
        # Try without overlap parameter first, fall back to overlap=False if needed
        try:
            mask = regionmask.mask_geopandas(gdf.geometry, lons, lats)
        except ValueError as e:
            if "overlapping regions" in str(e):
                print(
                    "   Detected overlapping regions, using overlap=False...")
                mask = regionmask.mask_geopandas(gdf.geometry,
                                                 lons,
                                                 lats,
                                                 overlap=False)
            else:
                raise

        print(f"✅ Created region mask")
        print(f"   Mask shape: {mask.shape}")
        print(
            f"   Unique regions in mask: {len(np.unique(mask.values[~np.isnan(mask.values)]))}"
        )

        return mask

    except Exception as e:
        print(f"❌ Failed to create region mask: {e}")
        raise


def extract_annual_extremes(spi_data):
    """Extract annual extremes using xclim methods"""
    print(f"\n{'='*70}")
    print("EXTRACTING ANNUAL EXTREMES")
    print(f"{'='*70}")

    try:
        # Use xclim's select_resample_op for annual minima (drought analysis)
        print("Extracting annual minima for drought analysis...")
        start_time = time.time()

        # Add units attribute for xclim compatibility (SPI is dimensionless)
        spi_data.attrs['units'] = '1'

        annual_minima = select_resample_op(
            spi_data,
            op='min',
            freq='YS',  # Annual frequency starting in January
        )

        extraction_time = time.time() - start_time
        print(f"✅ Annual extremes extracted in {extraction_time:.1f} seconds")
        print(f"   Shape: {dict(annual_minima.sizes)}")
        print(
            f"   Time range: {annual_minima.time.min().values} to {annual_minima.time.max().values}"
        )

        return annual_minima

    except Exception as e:
        print(f"❌ Failed to extract annual extremes: {e}")
        raise


def process_region_for_return_periods(annual_minima, region_mask, region_id, region_name, return_periods):
    """Process a single region for return period calculation - runs on worker"""
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


def calculate_return_periods_per_region(annual_minima,
                                        region_mask,
                                        gdf,
                                        client,
                                        return_periods=[2, 5, 10, 25, 50,
                                                        100]):
    """Calculate return periods for each administrative region using distributed computing"""
    print(f"\n{'='*70}")
    print("CALCULATING RETURN PERIODS PER REGION")
    print(f"{'='*70}")

    print(f"Return periods to calculate: {return_periods}")
    print(f"Using Dask client: {client}")
    
    # Ensure data is chunked for distributed processing
    # Check actual coordinate names and rechunk appropriately
    print(f"Data dimensions: {annual_minima.dims}")
    print(f"Current chunks: {annual_minima.chunks}")
    
    # Determine coordinate names (could be lat/lon or latitude/longitude)
    if 'latitude' in annual_minima.dims:
        chunk_dict = {'time': -1, 'latitude': 50, 'longitude': 50}
    elif 'lat' in annual_minima.dims:
        chunk_dict = {'time': -1, 'lat': 50, 'lon': 50}
    else:
        # Fallback - just chunk time dimension
        chunk_dict = {'time': -1}
    
    print(f"Rechunking with: {chunk_dict}")
    annual_minima = annual_minima.chunk(chunk_dict)
    print(f"New chunks: {annual_minima.chunks}")
    
    # Don't persist Icechunk data directly - avoid serialization issues
    # Instead, we'll work with the data locally and distribute the computation
    print("Skipping persist() to avoid Icechunk serialization issues")
    
    # Check if client is still connected
    try:
        worker_info = client.scheduler_info().get('workers', {})
        print(f"Client status: Connected with {len(worker_info)} workers")
    except Exception as e:
        print(f"❌ Client connection issue: {e}")
        print("Trying to reconnect...")
        try:
            # Try to get a fresh client
            cluster_name = "spi-extreme-analysis"
            import coiled
            cluster = coiled.Cluster(cluster_name)
            client = cluster.get_client()
            worker_info = client.scheduler_info().get('workers', {})
            print(f"✅ Reconnected with {len(worker_info)} workers")
        except Exception as reconnect_error:
            print(f"❌ Failed to reconnect: {reconnect_error}")
            raise

    def return_period_calculation(region_values, region_id, region_name):
        """Calculate return periods for a single region"""
        try:
            # Work with numpy array directly
            if isinstance(region_values, np.ndarray):
                # Remove spatial dimensions and work with time series
                if region_values.ndim > 1:
                    # Take spatial mean over the region, ignoring NaNs
                    region_ts = np.nanmean(region_values, axis=tuple(range(1, region_values.ndim)))
                else:
                    region_ts = region_values
            else:
                # Handle xarray case (fallback)
                if len(region_values.shape) > 1:
                    region_ts = region_values.mean(skipna=True).values
                else:
                    region_ts = region_values.values

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

    # Prepare tasks for distributed computation
    tasks = []
    results = []

    # Get unique region IDs from the mask
    unique_regions = np.unique(
        region_mask.values[~np.isnan(region_mask.values)])

    print(f"Processing {len(unique_regions)} regions...")

    start_time = time.time()

    for region_id in unique_regions:
        region_id = int(region_id)

        # Get region name from GeoDataFrame
        try:
            region_name = gdf.iloc[region_id]['shapeName']
            shape_id = gdf.iloc[region_id]['shapeID']
        except:
            region_name = f"Region_{region_id}"
            shape_id = f"ID_{region_id}"

        # Extract regional data - compute locally to avoid serialization issues
        region_data = annual_minima.where(region_mask == region_id)
        
        # Submit the region extraction and computation as a delayed task
        # This avoids local computation and sends everything to workers
        task = dask.delayed(process_region_for_return_periods)(
            annual_minima, region_mask, region_id, region_name, return_periods
        )
        tasks.append(task)

    # Compute all tasks in parallel using the client - like your working example
    print(f"Computing {len(tasks)} return period tasks in parallel...")
    
    # Check if we have workers available
    worker_info = client.scheduler_info().get('workers', {})
    print(f"Submitting to {len(worker_info)} workers...")
    
    if len(worker_info) == 0:
        print("Warning: No workers detected. Tasks may run on scheduler.")
    
    # Use client.compute to force execution on the cluster - same pattern as your working model
    futures = client.compute(tasks, sync=False)  # Get futures without blocking
    
    # Wait for completion like your working example
    results = []
    for future in futures:
        try:
            result = future.result()  # Wait for each task to complete
            results.append(result)
        except Exception as e:
            print(f"Task failed: {e}")
            results.append({
                'region_id': -1,
                'region_name': 'failed',
                'return_periods': return_periods,
                'return_levels': [np.nan] * len(return_periods),
                'status': f'error: {str(e)}',
                'n_years': 0
            })

    computation_time = time.time() - start_time

    # Process results
    successful_regions = [r for r in results if r['status'] == 'success']
    failed_regions = [r for r in results if r['status'] != 'success']

    print(
        f"✅ Return period calculation completed in {computation_time:.1f} seconds"
    )
    print(f"   Successful regions: {len(successful_regions)}")
    print(f"   Failed regions: {len(failed_regions)}")
    print(
        f"   Average processing rate: {len(unique_regions)/(computation_time):.1f} regions/second"
    )

    # Display sample results
    if successful_regions:
        print(f"\nSample results for {successful_regions[0]['region_name']}:")
        sample = successful_regions[0]
        for i, (T, level) in enumerate(
                zip(sample['return_periods'], sample['return_levels'])):
            print(f"   {T:3d}-year drought: SPI = {level:.3f}")

    return results


def save_results(results, output_file="extreme_value_analysis_results.json"):
    """Save results to JSON file"""
    import json

    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}")

    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"✅ Results saved to {output_file}")
        print(f"   Total regions processed: {len(results)}")

    except Exception as e:
        print(f"❌ Failed to save results: {e}")


def main():
    """Main execution function"""
    print("=" * 70)
    print("ICECHUNK REGIONMASK EXTREME VALUE ANALYSIS")
    print("=" * 70)

    cluster = None
    client = None

    try:
        # Step 1: Setup Coiled Dask cluster
        client, cluster = setup_coiled_cluster()

        # Step 2: Load SPI9 data from Icechunk
        dataset = load_icechunk_spi_data()

        # Get the SPI variable (adjust variable name as needed)
        spi_var = None
        for var in dataset.data_vars:
            if 'spi' in var.lower() or 'spc' in var.lower():
                spi_var = var
                break

        if spi_var is None:
            raise ValueError(
                f"No SPI variable found. Available variables: {list(dataset.data_vars)}"
            )

        spi_data = dataset[spi_var]
        if 'band' in spi_data.dims:
            spi_data = spi_data.squeeze('band')

        print(f"Using SPI variable: {spi_var}")

        # Step 3: Load administrative regions
        gdf, regions = load_administrative_regions()

        # Step 4: Create region mask
        region_mask = create_region_mask(gdf, dataset)

        # Step 5: Extract annual extremes
        annual_minima = extract_annual_extremes(spi_data)

        # Step 6: Calculate return periods per region
        results = calculate_return_periods_per_region(
            annual_minima,
            region_mask,
            gdf,
            client,
            return_periods=[2, 5, 10, 25, 50, 100])

        # Step 7: Save results
        save_results(results)

        # Summary
        print(f"\n{'='*70}")
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print(f"{'='*70}")

        successful_regions = [r for r in results if r['status'] == 'success']
        print(f"✅ Processed {len(successful_regions)} regions successfully")
        print(f"✅ Results saved for extreme value analysis")
        print(f"✅ Ready for drought risk assessment!")

        return True

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup
        if client:
            client.close()
        if cluster:
            cluster.close()
            print("Coiled cluster closed")


# if __name__ == "__main__":
#     success = main()
#     exit(0 if success else 1)

# if __name__ == "__main__":
#     success = main()
#     exit(0 if success else 1)

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
