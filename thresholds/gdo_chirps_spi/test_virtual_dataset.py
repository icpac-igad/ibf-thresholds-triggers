#!/usr/bin/env python3
"""
Test creating virtual dataset from downloaded JSON files
"""

import xarray as xr
import json
import time
import numpy as np

def test_virtual_dataset():
    """Test loading virtual dataset using kerchunk"""
    print("Testing virtual dataset creation...")
    
    # Load the virtual concat info
    with open('spi1/spi1_virtual_concat.json', 'r') as f:
        virtual_info = json.load(f)
    
    file_list = virtual_info['spi1_virtual_concat']['file_list']
    
    print(f"Loading {len(file_list)} JSON files...")
    
    # Test loading dataset
    start_time = time.time()
    try:
        ds = xr.open_mfdataset(
            file_list,
            engine='kerchunk',
            concat_dim='time',
            combine='nested',
            chunks={'time': 72, 'lat': 400, 'lon': 400},
            parallel=False
        )
        
        load_time = time.time() - start_time
        print(f"Dataset loaded successfully in {load_time:.2f} seconds!")
        
        # Print dataset info
        print("\nDataset structure:")
        print(ds)
        
        print("\nVariables:")
        for var in ds.data_vars:
            print(f"  {var}: {ds[var].shape} {ds[var].dtype}")
        
        print("\nCoordinates:")
        for coord in ds.coords:
            if coord == 'time':
                print(f"  {coord}: {len(ds[coord])} timesteps from {ds[coord].values[0]} to {ds[coord].values[-1]}")
            else:
                print(f"  {coord}: {len(ds[coord])} values from {ds[coord].values[0]:.2f} to {ds[coord].values[-1]:.2f}")
        
        # Test a small subset (East Africa region)
        print("\nTesting subset for East Africa region...")
        east_africa_bounds = {
            'lat_min': -15.0,
            'lat_max': 20.0,
            'lon_min': 25.0,
            'lon_max': 55.0
        }
        
        start_subset = time.time()
        subset = ds.sel(
            lat=slice(east_africa_bounds['lat_max'], east_africa_bounds['lat_min']),
            lon=slice(east_africa_bounds['lon_min'], east_africa_bounds['lon_max'])
        )
        subset_time = time.time() - start_subset
        
        print(f"Subset created in {subset_time:.4f} seconds")
        print(f"Subset shape: {subset.dims}")
        
        # Test computing a small sample
        print("\nTesting computation on a tiny sample...")
        tiny_sample = subset.isel(lat=slice(0, 5), lon=slice(0, 5), time=slice(0, 3))
        
        start_compute = time.time()
        result = tiny_sample.compute()
        compute_time = time.time() - start_compute
        
        print(f"Computed sample in {compute_time:.2f} seconds")
        print(f"Sample shape: {result.dims}")
        
        # Get the main variable name (should be spc01)
        var_name = list(result.data_vars)[0]
        print(f"Variable name: {var_name}")
        print(f"Sample values range: {float(result[var_name].min().values):.2f} to {float(result[var_name].max().values):.2f}")
        
        return ds
        
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_virtual_dataset()