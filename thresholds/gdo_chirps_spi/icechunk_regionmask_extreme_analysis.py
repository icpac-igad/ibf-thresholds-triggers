#!/usr/bin/env python3
"""
Quick Test: Icechunk Regionmask Extreme Analysis
===============================================

This is a quick test version using the working Icechunk + Dask approach.
Based on test_icechunk_credentials_simple.py successful pattern.

Usage:
    python icechunk_regionmask_extreme_analysis_quick.py
"""

import icechunk
import xarray as xr
import numpy as np
import geopandas as gpd
import regionmask
import warnings
import time
import coiled
from dask.distributed import Client
from pathlib import Path

warnings.filterwarnings('ignore')

# Configuration - same as working test
BASE_PREFIX = "t2spi1_east_africa_icechunk"
SPI_TYPE = "spi9"
BUCKET_NAME = "cdi_arco"
SERVICE_ACCOUNT_FILE = "coiled-data-e4drr_202505.json"
GEOJSON_FILE = "icpac_adm1v3.geojson"

def load_icechunk_dask_dataset():
    """Load Icechunk data using the working client-side approach"""
    print(f"\n{'='*60}")
    print("LOADING ICECHUNK DATA (CLIENT-SIDE APPROACH)")
    print(f"{'='*60}")
    
    try:
        # Initialize Icechunk repository on client (not on worker!)
        repo_prefix = f"{BASE_PREFIX}_{SPI_TYPE}"
        
        print(f"Repository: {repo_prefix}")
        print(f"Bucket: {BUCKET_NAME}")
        
        storage = icechunk.gcs_storage(
            bucket=BUCKET_NAME,
            prefix=repo_prefix,
            service_account_file=SERVICE_ACCOUNT_FILE  # Use local file path
        )
        
        repo = icechunk.Repository.open(storage)
        session = repo.readonly_session("main")
        group_name = f"{SPI_TYPE}_data"
        
        # Open dataset using session.store as zarr - this creates dask-backed arrays
        dataset = xr.open_zarr(
            session.store, 
            group=group_name,
            consolidated=False,
            chunks={'time': 50, 'lat': 200, 'lon': 200}  # Reasonable chunk sizes for analysis
        )
        
        print(f"✅ Successfully loaded dataset")
        print(f"   Shape: {dict(dataset.sizes)}")
        print(f"   Variables: {list(dataset.data_vars)}")
        print(f"   Chunks: {dataset.chunks}")
        
        return dataset
        
    except Exception as e:
        print(f"❌ Failed to load Icechunk data: {e}")
        raise

def load_regions_quick():
    """Load a subset of regions for quick testing"""
    print(f"\n{'='*60}")
    print("LOADING REGIONS (QUICK TEST)")
    print(f"{'='*60}")
    
    try:
        # Check if regions file exists
        if not Path(GEOJSON_FILE).exists():
            print(f"❌ Regions file not found: {GEOJSON_FILE}")
            print("   Creating dummy regions for testing...")
            
            # Create dummy regions for testing
            import geopandas as gpd
            from shapely.geometry import Polygon
            
            # Create 3 simple test regions
            regions_data = []
            for i in range(3):
                lon_min, lon_max = 35 + i*2, 37 + i*2
                lat_min, lat_max = -2 + i*0.5, 0 + i*0.5
                
                poly = Polygon([
                    (lon_min, lat_min), (lon_max, lat_min), 
                    (lon_max, lat_max), (lon_min, lat_max)
                ])
                
                regions_data.append({
                    'geometry': poly,
                    'region_id': i,
                    'region_name': f'TestRegion_{i}',
                    'GID_1': f'TEST_{i}',
                    'shapeName': f'TestRegion_{i}',
                    'shapeID': i
                })
            
            gdf = gpd.GeoDataFrame(regions_data, crs='EPSG:4326')
            
        else:
            # Load actual regions but take only first 5 for quick test
            gdf = gpd.read_file(GEOJSON_FILE)
            print(f"   Loaded {len(gdf)} regions, using first 5 for quick test")
            gdf = gdf.head(5).copy()
            gdf['region_id'] = range(len(gdf))
        
        print(f"✅ Using {len(gdf)} regions for testing")
        
        return gdf
        
    except Exception as e:
        print(f"❌ Failed to load regions: {e}")
        raise

def create_region_mask_quick(gdf, dataset):
    """Create region mask using regionmask"""
    print(f"\n{'='*60}")
    print("CREATING REGION MASK")
    print(f"{'='*60}")
    
    try:
        # Extract coordinates
        lons = dataset.lon.values
        lats = dataset.lat.values
        
        print(f"   Dataset grid: {len(lons)} × {len(lats)}")
        print(f"   Lon range: [{lons.min():.2f}, {lons.max():.2f}]")
        print(f"   Lat range: [{lats.min():.2f}, {lats.max():.2f}]")
        
        # Create mask
        mask = regionmask.mask_geopandas(gdf.geometry, lons, lats)
        
        unique_regions = len(np.unique(mask.values[~np.isnan(mask.values)]))
        print(f"✅ Region mask created")
        print(f"   Mask shape: {mask.shape}")
        print(f"   Regions in data: {unique_regions}")
        
        return mask
        
    except Exception as e:
        print(f"❌ Failed to create region mask: {e}")
        raise

def compute_regional_stats_on_worker(spi_data, region_mask, region_id, region_name):
    """Worker function to compute regional statistics - runs on Dask worker"""
    try:
        import xarray as xr
        import numpy as np
        
        # Extract data for this region
        region_data = spi_data.where(region_mask == region_id)
        
        # Compute basic statistics
        regional_mean = region_data.mean(dim=['lat', 'lon'], skipna=True)
        regional_min = region_data.min(dim=['lat', 'lon'], skipna=True) 
        regional_max = region_data.max(dim=['lat', 'lon'], skipna=True)
        
        # Get time series as numpy arrays for further processing
        mean_ts = regional_mean.compute().values
        min_ts = regional_min.compute().values
        max_ts = regional_max.compute().values
        
        # Remove NaN values and compute basic stats
        valid_mean = mean_ts[~np.isnan(mean_ts)]
        valid_min = min_ts[~np.isnan(min_ts)]
        
        result = {
            'region_id': region_id,
            'region_name': region_name,
            'n_valid_timesteps': len(valid_mean),
            'mean_spi': float(np.mean(valid_mean)) if len(valid_mean) > 0 else np.nan,
            'min_spi': float(np.min(valid_min)) if len(valid_min) > 0 else np.nan,
            'max_spi': float(np.max(max_ts[~np.isnan(max_ts)])) if len(max_ts[~np.isnan(max_ts)]) > 0 else np.nan,
            'drought_events': int(np.sum(valid_min < -1.5)) if len(valid_min) > 0 else 0,  # Severe drought events
            'status': 'success'
        }
        
        return result
        
    except Exception as e:
        return {
            'region_id': region_id,
            'region_name': region_name,
            'status': 'failed',
            'error': str(e)
        }

def test_distributed_regionmask_analysis(client, dataset, gdf, region_mask):
    """Test distributed regionmask analysis using working pattern"""
    print(f"\n{'='*60}")
    print("TESTING DISTRIBUTED REGIONMASK ANALYSIS")
    print(f"{'='*60}")
    
    # Get SPI variable
    spi_var = None
    for var in dataset.data_vars:
        if 'spi' in var.lower() or 'spc' in var.lower():
            spi_var = var
            break
    
    if spi_var is None:
        raise ValueError(f"No SPI variable found in {list(dataset.data_vars)}")
    
    spi_data = dataset[spi_var]
    if 'band' in spi_data.dims:
        spi_data = spi_data.squeeze('band')
    
    print(f"Using SPI variable: {spi_var}")
    print(f"SPI data shape: {spi_data.shape}")
    
    # Get unique regions from mask
    unique_regions = np.unique(region_mask.values[~np.isnan(region_mask.values)])
    print(f"Processing {len(unique_regions)} regions...")
    
    # Submit tasks to workers - same pattern as successful test
    futures = []
    for region_id in unique_regions:
        region_id = int(region_id)
        
        # Get region name
        try:
            if region_id < len(gdf):
                region_name = gdf.iloc[region_id].get('shapeName', f'Region_{region_id}')
            else:
                region_name = f'Region_{region_id}'
        except:
            region_name = f'Region_{region_id}'
        
        # Submit computation to worker
        future = client.submit(
            compute_regional_stats_on_worker,
            spi_data,
            region_mask, 
            region_id,
            region_name
        )
        futures.append(future)
    
    # Collect results - same pattern as successful test
    print("Collecting results from workers...")
    results = []
    for future in futures:
        try:
            result = future.result(timeout=120)  # 2 minute timeout per region
            results.append(result)
        except Exception as e:
            print(f"Task failed: {e}")
            results.append({
                'region_id': -1,
                'region_name': 'failed',
                'status': 'failed',
                'error': str(e)
            })
    
    # Process results
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    
    print(f"✅ Distributed analysis completed")
    print(f"   Successful: {len(successful)} regions")
    print(f"   Failed: {len(failed)} regions")
    
    # Show sample results
    if successful:
        print(f"\nSample results:")
        for result in successful[:3]:  # Show first 3
            print(f"   {result['region_name']}:")
            print(f"     Mean SPI: {result['mean_spi']:.3f}")
            print(f"     Min SPI: {result['min_spi']:.3f}")
            print(f"     Drought events: {result['drought_events']}")
    
    return results

def main():
    """Main execution function"""
    print("=" * 60)
    print("ICECHUNK REGIONMASK QUICK TEST")
    print("=" * 60)
    
    cluster = None
    client = None
    
    try:
        # Step 1: Setup small cluster for testing
        print("1. Setting up Coiled cluster...")
        cluster = coiled.Cluster(
            name="icechunk-regionmask-test",
            software="v3-geosfm-rm-x",
            n_workers=2,  # Small cluster for testing
            scheduler_vm_types=["n2-standard-2"],
            worker_vm_types="n2-standard-2",
            region="us-east1",
            arm=False,
            compute_purchase_option="spot",
            workspace='geosfm'
        )
        
        client = cluster.get_client()
        print(f"✅ Cluster ready: {client.dashboard_link}")
        
        # Wait for workers
        print("2. Waiting for workers...")
        time.sleep(20)
        workers = client.scheduler_info().get('workers', {})
        print(f"   Workers available: {len(workers)}")
        
        # Step 2: Load Icechunk data using working approach
        print("3. Loading Icechunk data...")
        dataset = load_icechunk_dask_dataset()
        
        # Step 3: Load regions (quick test version)
        print("4. Loading regions...")
        gdf = load_regions_quick()
        
        # Step 4: Create region mask
        print("5. Creating region mask...")
        region_mask = create_region_mask_quick(gdf, dataset)
        
        # Step 5: Test distributed regionmask analysis
        print("6. Testing distributed analysis...")
        results = test_distributed_regionmask_analysis(client, dataset, gdf, region_mask)
        
        # Summary
        print(f"\n{'='*60}")
        print("QUICK TEST COMPLETED")
        print(f"{'='*60}")
        
        successful = [r for r in results if r['status'] == 'success']
        print(f"✅ Processed {len(successful)} regions successfully")
        print(f"✅ Icechunk + Dask + Regionmask workflow is working!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        if client:
            client.close()
        if cluster:
            cluster.close()
            print("8. Cluster closed")

if __name__ == "__main__":
    success = main()
    print(f"\nFINAL RESULT: {'SUCCESS - Workflow is ready!' if success else 'FAILED - Need to debug'}")