"""
08-kmj-district-prob-trigger.py - Script to regrid forecast probabilities to 1km resolution,
overlay with district shapefiles, and calculate district-level average probabilities.

This script takes SEAS51 SPI3 forecast probabilities, regrids them to 1km resolution,
overlays them with district boundary shapefiles, and calculates average probabilities
for each district. The results are saved as CSV files and visualized with maps.

Usage:
     python 08-kmj-district-stats.py --input_netcdf kmj_seas51_spi3_jja_eprob_2025_04.nc  
                                         --district_shapefile ../../data/Karamoja_Admin2.shp
                                         --output_dir ./output
Author: [Your Name]
Date: [Current Date]
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap, BoundaryNorm
from datetime import datetime
import logging
import xesmf as xe
import regionmask
import json
from pathlib import Path

# Add necessary paths to import modules from the project
sys.path.append('.')

from vthree_utils import BinCreateParams

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_netcdf_forecast(file_path):
    """
    Load forecast data from a NetCDF file.
    
    Args:
        file_path (str): Path to the NetCDF file
        
    Returns:
        xarray.Dataset: The loaded forecast dataset
    """
    try:
        logger.info(f"Loading forecast data from {file_path}")
        ds = xr.open_dataset(file_path)
        logger.info(f"Loaded forecast dataset with variables: {list(ds.data_vars)}")
        return ds
    except Exception as e:
        logger.error(f"Error loading forecast data: {e}")
        raise


def regrid_to_1km(ds, target_res=0.01, method='conservative'):
    """
    Regrid forecast data to approximately 1km resolution.
    
    Args:
        ds (xarray.Dataset): The forecast dataset
        target_res (float): Target resolution in degrees (0.01° ≈ 1km at equator)
        method (str): Regridding method ('conservative', 'bilinear', 'nearest_s2d', etc.)
        
    Returns:
        xarray.Dataset: Regridded dataset
    """
    try:
        logger.info(f"Regridding forecast data to {target_res}° resolution using {method} method")
        
        # Get original latitude and longitude bounds
        lat_min, lat_max = float(ds.lat.min()), float(ds.lat.max())
        lon_min, lon_max = float(ds.lon.min()), float(ds.lon.max())
        
        logger.info(f"Original bounds: lat [{lat_min}, {lat_max}], lon [{lon_min}, {lon_max}]")
        
        # Create target grid (1km resolution ~ 0.01 degrees)
        target_grid = xr.Dataset({
            "lat": (["lat"], np.arange(lat_min, lat_max + target_res, target_res)),
            "lon": (["lon"], np.arange(lon_min, lon_max + target_res, target_res))
        })
        
        logger.info(f"Target grid dimensions: lat {len(target_grid.lat)}, lon {len(target_grid.lon)}")
        
        # Create regridder
        regridder = xe.Regridder(ds, target_grid, method, periodic=False)
        
        # Apply regridding to each variable
        regridded_ds = xr.Dataset()
        for var_name in ds.data_vars:
            logger.info(f"Regridding variable: {var_name}")
            regridded_ds[var_name] = regridder(ds[var_name])
        
        # Copy coordinates and attributes
        for coord_name, coord in ds.coords.items():
            if coord_name not in ['lat', 'lon']:
                regridded_ds[coord_name] = coord
        
        regridded_ds.attrs = ds.attrs
        
        logger.info(f"Regridding completed. New dimensions: {dict(regridded_ds.dims)}")
        
        return regridded_ds
        
    except Exception as e:
        logger.error(f"Error during regridding: {e}")
        raise



def load_district_shapefile(shapefile_path):
    """
    Load district boundaries from a shapefile.
    
    Args:
        shapefile_path (str): Path to the district shapefile
        
    Returns:
        geopandas.GeoDataFrame: The loaded shapefile
    """
    try:
        logger.info(f"Loading district shapefile from {shapefile_path}")
        gdf = gpd.read_file(shapefile_path)
        
        # Check if there's a district name column
        district_name_col = None
        for col in ['name', 'district', 'NAME', 'DISTRICT', 'District']:
            if col in gdf.columns:
                district_name_col = col
                break
        
        # If no district name column found, create one with sequential IDs
        if district_name_col is None:
            logger.warning("No district name column found. Creating one with sequential IDs.")
            gdf['district_name'] = [f'District_{i}' for i in range(len(gdf))]
            district_name_col = 'district_name'
        
        logger.info(f"Loaded {len(gdf)} districts. Using '{district_name_col}' as district name column.")
        
        # Ensure the district names are used as index
        gdf = gdf.set_index(district_name_col)
        
        return gdf, district_name_col
        
    except Exception as e:
        logger.error(f"Error loading district shapefile: {e}")
        raise


def create_district_mask(districts_gdf, regridded_ds):
    """
    Create a district mask for the regridded data.
    
    Args:
        districts_gdf (geopandas.GeoDataFrame): District boundaries
        regridded_ds (xarray.Dataset): Regridded forecast dataset
        
    Returns:
        xarray.DataArray: District mask
    """
    try:
        logger.info("Creating district mask")
        
        # Extract coordinates
        lons = regridded_ds.lon.values
        lats = regridded_ds.lat.values
        
        # Create regionmask from districts GeoDataFrame
        district_mask = regionmask.mask_geopandas(
            districts_gdf.reset_index(), 
            lons, 
            lats,
        )
        
        # Add district names as attributes
        district_mask.attrs['district_names'] = list(districts_gdf.index)
        
        logger.info(f"Created district mask with {len(districts_gdf)} districts")
        
        return district_mask
        
    except Exception as e:
        logger.error(f"Error creating district mask: {e}")
        raise



def calculate_district_averages(regridded_ds, district_mask, districts_gdf, district_name_col, district_map=None):
    """
    Calculate average values for each district and remap district codes to names if provided.
    
    Args:
        regridded_ds (xarray.Dataset): Regridded dataset with variables
        district_mask (xarray.DataArray): Mask with district indices
        dd (geopandas.GeoDataFrame): District dataframe with names
        district_name_col (str): Column name containing district names
        district_map (dict, optional): Mapping from real district names to codes
        
    Returns:
        pandas.DataFrame: DataFrame with district averages
    """
    # Initialize dictionary to store district averages
    district_averages = {}
    
    # Process each variable in the dataset
    for var_name in regridded_ds.data_vars:
        district_averages[var_name] = {}
        
        # Loop through each district
        for district_idx in range(len(districts_gdf)):
            district_name = dd.iloc[district_idx][district_name_col]
            
            # Create mask for this specific district (where mask equals the district index)
            district_bool_mask = (district_mask == district_idx)
            
            # Skip if no pixels in this district
            if not district_bool_mask.any():
                print(f"No pixels found for district: {district_name}")
                continue
                
            # Expand mask to match dataset dimensions
            expanded_mask = district_bool_mask
            additional_dims = {}
            
            for dim in regridded_ds.dims:
                if dim not in ['lat', 'lon']:
                    additional_dims[dim] = regridded_ds[dim]
                    
            if additional_dims:
                expanded_mask = district_bool_mask.expand_dims(additional_dims)
            
            # Apply mask to get values only for this district
            masked_data = regridded_ds[var_name].where(expanded_mask)
            
            # Calculate district average
            district_avg = masked_data.mean(dim=['lat', 'lon']).values
            
            # Store in our results dictionary
            district_averages[var_name][district_name] = district_avg
    
    # Convert to DataFrame for easier handling
    results_df = pd.DataFrame()
    
    for var_name, districts in district_averages.items():
        var_df = pd.DataFrame(districts).T
        var_df.columns = [var_name]
        
        if results_df.empty:
            results_df = var_df
        else:
            results_df = pd.concat([results_df, var_df], axis=1)
    
    # Apply district mapping if provided
    if district_map:
        # Convert district codes to actual district names
        results_df['district_name'] = results_df.index.map(
            lambda x: next((k for k, v in district_map.items() if v == x), x)
        )
        
        # Set the district names as the index
        results_df = results_df.set_index('district_name')
        
        # Reorder rows based on the order in district_map
        ordered_districts = list(district_map.keys())
        
        # Only include districts that exist in our results
        valid_districts = [d for d in ordered_districts if d in results_df.index]
        
        if valid_districts:
            results_df = results_df.loc[valid_districts]
    
    return results_df


def main():
    """Main function to run the script
    
    Example usage:
    python 08-kmj-district-stats.py --input_netcdf kmj_seas51_spi3_jja_eprob_2025_04.nc \
                                         --district_shapefile ../../data/Karamoja_Admin2.shp \
                                         
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Calculate district-level drought risk from SEAS51 forecasts")
    parser.add_argument("--input_netcdf", required=True, help="Path to forecast emprical probablity netcdf file")
    parser.add_argument("--district_shapefile", required=True, help="Path to district shapefile")
       
    args = parser.parse_args()
    
    try:
        # Ensure output directory exists
        os.makedirs(args.output_dir, exist_ok=True)
        epds=load_netcdf_forecast(args.input_netcdf)
        regridded_ds = regrid_to_1km(epds, target_res=0.01)

        districts_gdf, district_name_col = load_district_shapefile(args.district_shapefile)

        district_mask = create_district_mask(districts_gdf, regridded_ds)
        dd_dict={'Karenga': 'District_7', 'Kaabong': 'District_6', 'Kotido': 'District_3', 'Abim': 'District_0', 'Napak': 'District_1', 'Moroto': 'District_4', 'Nabilatuk': 'District_2', 'Nakapiripirit': 'District_5', 'Amudat': 'District_8'}
        # Calculate district averages with mapping
        results_df = calculate_district_averages(
            regridded_ds,
            district_mask,
            districts_gdf, 
            district_name_col,
            district_map=dd_dict
        )
        results_df.to_csv(f"{os.path.splitext(args.input_netcdf)[0]}_district_averages.csv")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


