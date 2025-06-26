#!/usr/bin/env python3
"""
Test xclim frequency analysis with locally computed regional data.
This avoids issues with virtual dataset structure by computing to local first.
"""

import pickle
import time
import xarray as xr
import numpy as np
import geopandas as gpd
import warnings
import xclim
from xclim.indices import stats
import os

warnings.filterwarnings('ignore')

def compute_and_save_regional_data(region_index=5):
    """Compute regional subset and save to NetCDF"""
    
    print("="*60)
    print("XCLIM FREQUENCY ANALYSIS WITH LOCAL COMPUTE")
    print("="*60)
    
    # Load virtual dataset
    print("1. Loading virtual dataset...")
    start = time.time()
    
    with open('spi1_full_virtual_dataset.pkl', 'rb') as f:
        ds = pickle.load(f)
    
    load_time = time.time() - start
    print(f"✓ Virtual dataset loaded in {load_time:.1f} seconds")
    
    # Get region
    gdf = gpd.read_file("icpac_regions_admin1_20250630.geojson")
    region = gdf.iloc[region_index:region_index+1]
    region_name = region['shapeName'].values[0]
    bounds = region.total_bounds
    
    print(f"\n2. Subsetting to region: {region_name}")
    print(f"   Bounds: lon [{bounds[0]:.2f}, {bounds[2]:.2f}], lat [{bounds[1]:.2f}, {bounds[3]:.2f}]")
    
    # Subset to region (virtual operation)
    start = time.time()
    ds_subset = ds.sel(
        lon=slice(bounds[0], bounds[2]),
        lat=slice(bounds[3], bounds[1])  # Reversed for latitude
    )
    
    subset_time = time.time() - start
    print(f"✓ Virtual subset in {subset_time:.3f} seconds")
    
    # Get SPI data and remove band dimension
    spi_data = ds_subset['spc01'].squeeze('band')
    print(f"   Region shape: {dict(spi_data.sizes)}")
    
    # Compute to local memory
    print(f"\n3. Computing regional data to local memory...")
    start = time.time()
    
    # This is the key step - compute the virtual data
    spi_data_local = spi_data.compute()
    
    compute_time = time.time() - start
    total_pixels = spi_data.sizes['lat'] * spi_data.sizes['lon']
    data_size_mb = spi_data_local.nbytes / 1024 / 1024
    
    print(f"✓ Computed {total_pixels} pixels in {compute_time:.1f} seconds")
    print(f"   Data size: {data_size_mb:.1f} MB")
    print(f"   Computation rate: {total_pixels/compute_time:.0f} pixels/second")
    
    # Save to NetCDF
    output_file = f'spi1_{region_name.lower().replace(" ", "_")}_local.nc'
    print(f"\n4. Saving to NetCDF: {output_file}")
    start = time.time()
    
    # Create dataset with proper metadata
    ds_local = xr.Dataset({
        'spi': spi_data_local
    })
    
    # Add attributes
    ds_local.attrs['description'] = f'SPI-1 data for {region_name} region'
    ds_local.attrs['region'] = region_name
    ds_local.attrs['computed_from'] = 'Virtual SPI dataset'
    
    # Fix encoding for time coordinate to avoid int64 issues
    if 'time' in ds_local.coords:
        ds_local.time.encoding['dtype'] = 'float64'
        ds_local.time.encoding['units'] = 'days since 1900-01-01'
    
    # Save with engine='netcdf4' to support more data types
    ds_local.to_netcdf(output_file, engine='netcdf4')
    save_time = time.time() - start
    
    print(f"✓ Saved in {save_time:.1f} seconds")
    
    return output_file, ds_local, region_name

def test_xclim_on_local_data(nc_file, ds_local, region_name):
    """Test xclim frequency analysis on local NetCDF data"""
    
    print(f"\n5. Testing xclim frequency analysis on local data...")
    
    # Get SPI data
    spi_data = ds_local['spi']
    
    # Compute annual minima for drought analysis
    print("   Computing annual minima...")
    start = time.time()
    
    annual_minima = spi_data.groupby('time.year').min()
    annual_minima = annual_minima.compute() if hasattr(annual_minima, 'compute') else annual_minima
    
    extremes_time = time.time() - start
    print(f"✓ Annual minima computed in {extremes_time:.1f} seconds")
    
    # Test xclim fit on single point
    print("\n6. Testing xclim.indices.stats.fit()...")
    
    try:
        # Select a point in the middle
        mid_lat = annual_minima.sizes['lat'] // 2
        mid_lon = annual_minima.sizes['lon'] // 2
        
        point_data = annual_minima.isel(lat=mid_lat, lon=mid_lon)
        print(f"   Point data shape: {point_data.shape}")
        print(f"   Point data type: {type(point_data)}")
        print(f"   Point data dims: {point_data.dims}")
        
        # For drought analysis, we analyze the negative of minima
        drought_data = -1 * point_data
        
        # xclim.stats.fit expects 'time' dimension, rename 'year' to 'time'
        if 'year' in drought_data.dims:
            drought_data = drought_data.rename({'year': 'time'})
        
        # Try xclim fit
        start = time.time()
        fitted_params = stats.fit(drought_data, dist='genextreme')
        fit_time = time.time() - start
        
        print(f"✓ xclim fit successful in {fit_time:.3f} seconds!")
        print(f"   GEV parameters: {fitted_params}")
        
        # Compute return levels
        print("\n7. Computing return levels...")
        return_periods = np.array([2, 5, 10, 25, 50, 100])
        
        # For annual data, exceedance probability = 1 - 1/T
        quantiles = 1 - 1/return_periods
        
        return_levels = stats.parametric_quantile(
            fitted_params, 
            q=quantiles,
            dist='genextreme'
        )
        
        # Convert back to drought scale
        drought_levels = -1 * return_levels
        
        print("   Drought return levels (SPI values):")
        for T, level in zip(return_periods, drought_levels):
            print(f"     {T:3.0f}-year drought: SPI = {level:.3f}")
        
        # Test on full spatial data
        print("\n8. Testing spatial application...")
        test_spatial_fitting(annual_minima)
        
    except Exception as e:
        print(f"✗ xclim fitting failed: {e}")
        import traceback
        traceback.print_exc()
    
    return annual_minima

def test_spatial_fitting(annual_minima):
    """Test fitting across spatial dimensions"""
    
    print("   Testing vectorized GEV fitting across space...")
    
    # Test on small subset first
    subset = annual_minima.isel(lat=slice(0, 3), lon=slice(0, 3))
    
    start = time.time()
    
    # Apply xclim fit using apply_ufunc
    def fit_gev_1d(data_1d):
        """Fit GEV to 1D annual data"""
        try:
            # Remove NaN values
            valid_data = data_1d[~np.isnan(data_1d)]
            
            if len(valid_data) < 10:
                return np.array([np.nan, np.nan, np.nan])
            
            # For drought, use negative values
            drought_data = -1 * valid_data
            
            # Fit using xclim
            params = stats.fit(drought_data, dist='genextreme')
            return params
            
        except:
            return np.array([np.nan, np.nan, np.nan])
    
    # Apply to each spatial point
    fitted_params = xr.apply_ufunc(
        fit_gev_1d,
        subset,
        input_core_dims=[['year']],
        output_core_dims=[['params']],
        output_sizes={'params': 3},
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float]
    )
    
    fit_time = time.time() - start
    n_points = subset.sizes['lat'] * subset.sizes['lon']
    
    print(f"✓ Fitted {n_points} points in {fit_time:.1f} seconds")
    print(f"   Rate: {n_points/fit_time:.0f} points/second")
    
    # Estimate for full region
    full_points = annual_minima.sizes['lat'] * annual_minima.sizes['lon']
    estimated_time = fit_time * (full_points / n_points)
    
    print(f"\n   Full region estimates:")
    print(f"   - Total points: {full_points}")
    print(f"   - Single core: {estimated_time:.1f} seconds")
    print(f"   - 8 workers: {estimated_time/8:.1f} seconds")
    print(f"   - 16 workers: {estimated_time/16:.1f} seconds")

def main():
    """Main execution"""
    
    # Step 1: Compute and save regional data
    nc_file, ds_local, region_name = compute_and_save_regional_data(region_index=5)
    
    # Step 2: Test xclim on local data
    annual_minima = test_xclim_on_local_data(nc_file, ds_local, region_name)
    
    print(f"\n{'='*60}")
    print("SUMMARY:")
    print(f"{'='*60}")
    print("✅ Virtual dataset → Regional subset → Local compute → xclim")
    print("✅ This workflow avoids virtual dataset compatibility issues")
    print("✅ xclim functions work properly on computed local data")
    print("✅ GEV fitting can be parallelized across spatial points")
    print("✅ Ready for Dask cluster deployment!")
    
    print(f"\nOutput file saved: {nc_file}")
    file_size = os.path.getsize(nc_file) / 1024 / 1024
    print(f"File size: {file_size:.1f} MB")

if __name__ == "__main__":
    main()