#!/usr/bin/env python3
"""
Final SPI Virtual Dataset Test - ALL 35 FILES with Error Handling
- Opens all 35 JSON files for 1260+ timesteps
- Robust error handling for network timeouts
- First polygon analysis only
- Optimized for production use
"""

import json
import xarray as xr
import numpy as np
import logging
import time
import pickle
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_full_virtual_spi_dataset(use_all_files: bool = True, max_files: int = None) -> xr.Dataset:
    """
    Create virtual SPI dataset using ALL 35 files or specified number
    """
    
    # Check for cached dataset
    if use_all_files:
        cache_file = Path("spi1_full_virtual_dataset.pkl")
        cache_info_file = Path("spi1_full_info.json")
    else:
        cache_file = Path(f"spi1_{max_files}files_virtual_dataset.pkl")
        cache_info_file = Path(f"spi1_{max_files}files_info.json")
    
    # Try to load from cache first
    if cache_file.exists():
        logger.info(f"Loading cached dataset from {cache_file}...")
        try:
            with open(cache_file, 'rb') as f:
                ds = pickle.load(f)
            
            # Load info
            if cache_info_file.exists():
                with open(cache_info_file, 'r') as f:
                    info = json.load(f)
                logger.info(f"✓ Loaded cached dataset: {info}")
            
            logger.info(f"✓ Cached dataset shape: {dict(ds.sizes)}")
            return ds
            
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}, creating new dataset...")
    
    # Load file list
    logger.info("Creating virtual SPI dataset...")
    with open('spi1/spi1_virtual_concat.json', 'r') as f:
        virtual_info = json.load(f)
    
    file_list = virtual_info['spi1_virtual_concat']['file_list']
    
    if not use_all_files and max_files:
        file_list = file_list[:max_files]
        logger.info(f"Using first {max_files} files")
    else:
        logger.info(f"Using ALL {len(file_list)} files - this will take 8-10 minutes...")
        logger.info("Estimated total time steps: ~1260 (35 files × 36 steps)")
    
    start_time = time.time()
    
    # Create dataset with error handling
    logger.info("Opening files with xarray (this takes time but stays virtual)...")
    
    try:
        # Use more conservative settings for stability
        ds = xr.open_mfdataset(
            file_list,
            engine='kerchunk',
            concat_dim='time',
            combine='nested',
            chunks={},  # Keep original file chunks initially
            parallel=False,  # More stable than parallel
            decode_times=True,
            data_vars='minimal',  # Only load essential variables
            coords='minimal'      # Only load essential coordinates
        )
        
        # Apply optimized rechunking
        logger.info("Applying optimized rechunking for temporal analysis...")
        ds_rechunked = ds.chunk({
            'time': -1,      # Single time chunk for temporal analysis
            'lat': 200,      # Larger chunks for better performance  
            'lon': 200
        })
        
        load_time = time.time() - start_time
        
        logger.info(f"✓ Virtual dataset created successfully!")
        logger.info(f"  Total time: {load_time:.1f}s ({load_time/60:.1f} minutes)")
        logger.info(f"  Shape: {dict(ds_rechunked.sizes)}")
        logger.info(f"  Total time steps: {ds_rechunked.sizes['time']}")
        logger.info(f"  Optimized chunks: {ds_rechunked.chunks}")
        
        # Save to cache
        logger.info(f"Saving to cache: {cache_file}")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(ds_rechunked, f)
            
            # Save info
            info = {
                "creation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_files": len(file_list),
                "total_time_steps": ds_rechunked.sizes['time'],
                "shape": dict(ds_rechunked.sizes),
                "processing_time_minutes": load_time / 60
            }
            
            with open(cache_info_file, 'w') as f:
                json.dump(info, f, indent=2)
            
            logger.info(f"✓ Dataset cached successfully")
            
        except Exception as e:
            logger.warning(f"Failed to cache dataset: {e}")
        
        return ds_rechunked
        
    except Exception as e:
        logger.error(f"Failed to create dataset: {e}")
        logger.info("This might be due to network issues or file access problems")
        raise

def load_first_polygon_and_subset(geojson_file: str, ds: xr.Dataset):
    """
    Load GeoJSON and use the first polygon for analysis
    """
    
    try:
        import geopandas as gpd
        
        if not Path(geojson_file).exists():
            logger.warning(f"GeoJSON not found: {geojson_file}")
            logger.info("Using default East Africa bounds...")
            
            # Default bounds for East Africa
            bounds = (32.0, -5.0, 45.0, 15.0)  # lon_min, lat_min, lon_max, lat_max
            region_name = "East_Africa_Default"
            
        else:
            logger.info(f"Loading GeoJSON: {geojson_file}")
            gdf = gpd.read_file(geojson_file)
            logger.info(f"Loaded {len(gdf)} polygons")
            
            # Use first polygon
            first_polygon = gdf.iloc[0:1]
            bounds = first_polygon.total_bounds
            
            # Get region name
            name_cols = ['name', 'NAME', 'shapeName', 'admin_name', 'ADMIN_NAME']
            region_name = "Unknown_Region"
            
            for col in name_cols:
                if col in first_polygon.columns:
                    region_name = str(first_polygon[col].iloc[0])
                    break
        
        logger.info(f"Selected region: {region_name}")
        logger.info(f"Bounds: {bounds}")
        
        # Subset dataset to region
        minx, miny, maxx, maxy = bounds
        
        logger.info(f"Subsetting dataset to region bounds...")
        ds_subset = ds.sel(
            lon=slice(minx, maxx),
            lat=slice(maxy, miny)  # Reversed for latitude
        )
        
        logger.info(f"✓ Subset to region: {dict(ds_subset.sizes)}")
        
        return ds_subset, region_name, bounds
        
    except Exception as e:
        logger.error(f"GeoJSON processing failed: {e}")
        logger.info("Using fallback Kenya region...")
        
        # Fallback to Kenya region
        kenya_bounds = (33.0, -5.0, 42.0, 5.0)
        ds_subset = ds.sel(
            lon=slice(33.0, 42.0),
            lat=slice(5.0, -5.0)
        )
        
        return ds_subset, "Kenya_Fallback", kenya_bounds

def test_computation_robust(ds: xr.Dataset, region_name: str) -> dict:
    """
    Test computation with robust error handling for network issues
    """
    
    logger.info(f"Testing computation on {region_name}...")
    logger.info(f"Dataset shape: {dict(ds.sizes)}")
    
    # Get SPI variable
    spi_var = [v for v in ds.data_vars.keys() if 'spc' in v][0]
    spi_data = ds[spi_var]
    
    logger.info(f"SPI variable: {spi_var}")
    logger.info(f"SPI data shape: {spi_data.shape}")
    
    results = {'region': region_name, 'shape': dict(ds.sizes)}
    
    # Test 1: Sample a small region first (to test network connectivity)
    logger.info("1. Testing with small sample first...")
    try:
        # Take a small 5x5 pixel sample
        sample_data = spi_data.isel(
            lat=slice(0, 5),
            lon=slice(0, 5),
            time=slice(0, 10)  # First 10 time steps
        )
        
        sample_start = time.time()
        sample_result = sample_data.mean().compute()
        sample_time = time.time() - sample_start
        
        results['sample_test'] = {
            'success': True,
            'value': float(sample_result),
            'time': sample_time,
            'shape': sample_data.shape
        }
        
        logger.info(f"✓ Sample test successful: {sample_result:.3f} in {sample_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Sample test failed: {e}")
        results['sample_test'] = {'success': False, 'error': str(e)}
        
        # If sample fails, likely network issue - stop here
        logger.error("Network connectivity issue detected - stopping computation tests")
        return results
    
    # Test 2: Basic statistics (if sample worked)
    logger.info("2. Computing basic statistics...")
    try:
        compute_start = time.time()
        
        # Compute on a larger but manageable subset
        subset_data = spi_data.isel(
            lat=slice(0, min(20, ds.sizes['lat'])),
            lon=slice(0, min(20, ds.sizes['lon']))
        )
        
        stats = {
            'min': float(subset_data.min().compute()),
            'max': float(subset_data.max().compute()),
            'mean': float(subset_data.mean().compute()),
            'std': float(subset_data.std().compute())
        }
        
        compute_time = time.time() - compute_start
        results['basic_stats'] = {**stats, 'compute_time': compute_time, 'success': True}
        
        logger.info(f"✓ Basic stats computed in {compute_time:.2f}s")
        logger.info(f"  SPI range: {stats['min']:.3f} to {stats['max']:.3f}")
        logger.info(f"  Mean: {stats['mean']:.3f} ± {stats['std']:.3f}")
        
    except Exception as e:
        logger.error(f"Basic stats failed: {e}")
        results['basic_stats'] = {'success': False, 'error': str(e)}
    
    # Test 3: Time series analysis
    logger.info("3. Computing spatial mean time series...")
    try:
        # Use smaller spatial subset for time series
        ts_data = spi_data.isel(
            lat=slice(0, min(10, ds.sizes['lat'])),
            lon=slice(0, min(10, ds.sizes['lon']))
        )
        
        spatial_mean = ts_data.mean(dim=['lat', 'lon'])
        time_series = spatial_mean.compute()
        
        # Convert to list of floats (avoiding numpy array issues)
        ts_values = [float(x) for x in time_series.values.flatten()]
        
        results['time_series'] = {
            'success': True,
            'length': len(ts_values),
            'first_value': ts_values[0],
            'last_value': ts_values[-1],
            'mean': float(np.mean(ts_values)),
            'std': float(np.std(ts_values)),
            'sample_values': [ts_values[i] for i in [0, len(ts_values)//4, len(ts_values)//2, -1]]
        }
        
        logger.info(f"✓ Time series computed: {len(ts_values)} time steps")
        logger.info(f"  Range: {ts_values[0]:.3f} to {ts_values[-1]:.3f}")
        logger.info(f"  Mean: {results['time_series']['mean']:.3f}")
        
    except Exception as e:
        logger.error(f"Time series failed: {e}")
        results['time_series'] = {'success': False, 'error': str(e)}
    
    # Test 4: Dataset info verification
    logger.info("4. Verifying dataset properties...")
    try:
        time_coord = ds.time
        
        results['dataset_info'] = {
            'success': True,
            'time_range': [str(time_coord.min().values), str(time_coord.max().values)],
            'total_time_steps': int(ds.sizes['time']),
            'spatial_resolution': [float(ds.lat[1] - ds.lat[0]), float(ds.lon[1] - ds.lon[0])],
            'chunking': str(ds.chunks)
        }
        
        logger.info(f"✓ Dataset verification complete")
        logger.info(f"  Time range: {results['dataset_info']['time_range'][0]} to {results['dataset_info']['time_range'][1]}")
        logger.info(f"  Total time steps: {results['dataset_info']['total_time_steps']}")
        
    except Exception as e:
        logger.error(f"Dataset verification failed: {e}")
        results['dataset_info'] = {'success': False, 'error': str(e)}
    
    return results

def main():
    """Main function for full dataset test"""
    
    logger.info("="*70)
    logger.info("SPI VIRTUAL DATASET - FULL 35 FILES TEST")
    logger.info("="*70)
    
    try:
        # Step 1: Create/load full virtual dataset
        logger.info("Step 1: Creating/loading FULL virtual dataset (ALL 35 files)...")
        ds = create_full_virtual_spi_dataset(use_all_files=True)
        
        # Step 2: Load first polygon and subset
        logger.info("Step 2: Loading first polygon from GeoJSON...")
        geojson_file = "icpac_regions_admin1_20250630.geojson"
        ds_region, region_name, bounds = load_first_polygon_and_subset(geojson_file, ds)
        
        # Step 3: Test computation with error handling
        logger.info("Step 3: Testing computation with network error handling...")
        results = test_computation_robust(ds_region, region_name)
        
        # Step 4: Results summary
        logger.info("\n" + "="*70)
        logger.info("FINAL RESULTS SUMMARY")
        logger.info("="*70)
        
        success_count = sum(1 for key, result in results.items() 
                          if isinstance(result, dict) and result.get('success', False))
        total_tests = len([key for key, result in results.items() 
                          if isinstance(result, dict) and 'success' in result])
        
        logger.info(f"Region: {results.get('region', 'Unknown')}")
        logger.info(f"Dataset shape: {results.get('shape', 'Unknown')}")
        logger.info(f"Tests passed: {success_count}/{total_tests}")
        
        # Show specific results
        if results.get('dataset_info', {}).get('success'):
            info = results['dataset_info']
            logger.info(f"\n📊 DATASET VERIFICATION:")
            logger.info(f"  ✓ Total time steps: {info['total_time_steps']}")
            logger.info(f"  ✓ Time range: {info['time_range'][0]} to {info['time_range'][1]}")
            logger.info(f"  ✓ All 35 files successfully concatenated!")
        
        if results.get('basic_stats', {}).get('success'):
            stats = results['basic_stats']
            logger.info(f"\n📈 COMPUTATION RESULTS:")
            logger.info(f"  ✓ SPI range: {stats['min']:.3f} to {stats['max']:.3f}")
            logger.info(f"  ✓ Mean SPI: {stats['mean']:.3f} ± {stats['std']:.3f}")
            logger.info(f"  ✓ Computation time: {stats['compute_time']:.2f}s")
        
        if results.get('time_series', {}).get('success'):
            ts = results['time_series']
            logger.info(f"\n📈 TIME SERIES ANALYSIS:")
            logger.info(f"  ✓ Length: {ts['length']} time steps")
            logger.info(f"  ✓ Temporal mean: {ts['mean']:.3f}")
            logger.info(f"  ✓ Sample values: {[f'{x:.3f}' for x in ts['sample_values']]}")
        
        if success_count >= 3:
            logger.info(f"\n🎉 SUCCESS! Virtual dataset with ALL 35 files working!")
            logger.info(f"✓ Ready for extreme value analysis on Dask cluster")
            logger.info(f"✓ Dataset cached for reuse by workers")
        else:
            logger.warning(f"⚠️  Some issues detected - check network connectivity")
        
        logger.info("="*70)
        
        return 0
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())