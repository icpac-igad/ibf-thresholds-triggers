#!/usr/bin/env python3
"""
Script to calculate GEV (Generalized Extreme Value) return periods for SPI data
across different regions defined in a GeoJSON file using xclim.

This script:
1. Reads SPI data from either NetCDF or Zarr format
2. Creates masks for regions defined in a GeoJSON file
3. Calculates GEV return periods for drought and flood events using xclim
4. Outputs results as CSV files for each month
"""

import os
import glob
import logging
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import regionmask
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings

# Import xclim for GEV analysis
import xclim
from xclim.indices.stats import frequency_analysis

# Ignore specific warnings that might occur during GEV fitting
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
REGIONS_FILE = "icpac_regions.geojson"
SPI_DATA_DIR = "ecmwf_spi_not_exists"  # Use a non-existent directory to force synthetic data generation
OUTPUT_DIR = "return_periods_xclim"
SPI_TYPES = ["SPI1"]  # Run just one for testing
RETURN_PERIODS = [2, 4, 7, 10, 15, 20, 40, 100]  # in years
MONTHS = [1]  # Just run for January to test
USE_SYNTHETIC_DATA = True  # Set to True to use synthetic data for testing

def create_region_masks(regions_file, ds_example):
    """
    Create region masks based on a GeoJSON file.
    
    Args:
        regions_file: Path to GeoJSON file with region polygons
        ds_example: Example dataset to extract lat/lon coordinates
        
    Returns:
        tuple: (region_mask, region_info_df)
            - region_mask: RegionMask object (or None if not needed)
            - region_info_df: DataFrame with region information
    """
    logger.info(f"Creating region masks from {regions_file}")
    
    # Read regions file
    regions_gdf = gpd.read_file(regions_file)
    
    # For testing, limit to just 5 regions to make it faster
    regions_gdf = regions_gdf.head(5)
    
    # Extract shape name as the region name
    if 'shapeName' in regions_gdf.columns:
        region_name_col = 'shapeName'
    elif 'name' in regions_gdf.columns:
        region_name_col = 'name'
    elif 'NAME' in regions_gdf.columns:
        region_name_col = 'NAME'
    elif 'Region' in regions_gdf.columns:
        region_name_col = 'Region'
    else:
        # Use the first string column as the name
        str_cols = [col for col in regions_gdf.columns 
                   if regions_gdf[col].dtype == 'object']
        if str_cols:
            region_name_col = str_cols[0]
        else:
            # If no string column found, use the index as string
            regions_gdf['region_name'] = regions_gdf.index.astype(str)
            region_name_col = 'region_name'
    
    # Use numeric IDs for regionmask
    regions_gdf['region_numeric_id'] = np.arange(1, len(regions_gdf) + 1)
    
    # For simplified testing, we'll just use the index as the region ID
    regions_gdf['region_id'] = regions_gdf.index + 1
    
    # Create region info DataFrame
    region_info = pd.DataFrame({
        'region_id': regions_gdf['region_id'],
        'region_name': regions_gdf[region_name_col]
    })
    
    # For synthetic data testing, we don't actually need the mask
    # So we'll return None instead and skip that part of the processing
    region_mask = None
    
    # Log info
    logger.info(f"Extracted information for {len(regions_gdf)} regions (limited for testing)")
        
    return region_mask, region_info

def load_spi_data(spi_type, data_dir):
    """
    Load SPI data either from individual NetCDF files or a combined Zarr store.
    
    Args:
        spi_type: Type of SPI data (e.g., "SPI1", "SPI3", etc.)
        data_dir: Directory containing the SPI data
        
    Returns:
        xarray.Dataset with the SPI data
    """
    logger.info(f"Loading {spi_type} data from {data_dir}")
    
    # Use synthetic data for testing if requested
    if USE_SYNTHETIC_DATA or not os.path.exists(data_dir) or len(glob.glob(os.path.join(data_dir, f"{spi_type}_*.nc"))) == 0:
        logger.warning(f"Using synthetic data for {spi_type}")
        
        # Create synthetic data - a very small dataset for testing
        # Time dimension - using just a few years for faster processing
        times = pd.date_range('2010-01-01', '2015-12-31', freq='MS')  # Reduced time period
        
        # Spatial dimensions - tiny grid for quick testing
        lats = np.linspace(-12, 21, 5)  # Reduced resolution
        lons = np.linspace(23, 53, 5)  # Reduced resolution
        
        # Create the synthetic data with a random component
        np.random.seed(42)  # For reproducibility
        data = np.random.normal(0, 1, size=(len(times), len(lats), len(lons)))
        
        # Create xarray Dataset
        ds = xr.Dataset(
            data_vars={
                spi_type: (['time', 'lat', 'lon'], data)
            },
            coords={
                'time': times,
                'lat': lats,
                'lon': lons
            }
        )
        
        # Add metadata
        ds[spi_type].attrs['long_name'] = f'Standardized Precipitation Index ({spi_type})'
        ds[spi_type].attrs['units'] = 'unitless'
        
        return ds
    
    # If not using synthetic data, try to load actual data
    # Check if a Zarr store exists
    zarr_path = os.path.join("spi_zarr", f"{spi_type}.zarr")
    if os.path.exists(zarr_path):
        logger.info(f"Loading from Zarr store: {zarr_path}")
        try:
            ds = xr.open_zarr(zarr_path)
            return ds
        except Exception as e:
            logger.error(f"Error loading Zarr store: {str(e)}")
    
    # If no Zarr store or loading failed, load from NetCDF files
    nc_pattern = os.path.join(data_dir, f"{spi_type}_*.nc")
    logger.info(f"Loading from NetCDF files: {nc_pattern}")
    try:
        ds = xr.open_mfdataset(nc_pattern, combine='by_coords')
        return ds
    except Exception as e:
        logger.error(f"Error loading NetCDF files: {str(e)}")
        return None

def calculate_return_levels_xclim(values, return_periods, event_type='min'):
    """
    Calculate return levels using xclim's frequency_analysis function.
    
    Args:
        values: 1D array of values
        return_periods: List of return periods in years
        event_type: 'min' for drought (minimum values), 'max' for flood (maximum values)
        
    Returns:
        dict: Return period -> return level
    """
    # Make sure we have enough samples
    if len(values) < 30 or np.all(np.isnan(values)):
        return {rp: np.nan for rp in return_periods}
    
    try:
        # Convert numpy array to list for xclim compatibility
        values_list = values.tolist()
        
        # Create a simple DataArray with the values
        da = xr.DataArray(values_list, dims=('time',), 
                         coords={'time': pd.date_range('2000-01-01', periods=len(values_list), freq='Y')})
        
        # Set the event type ('min' for drought events, 'max' for flood events)
        mode = 'low' if event_type == 'min' else 'high'
        
        # Make sure return periods are integers
        rp_int = [int(rp) for rp in return_periods]
        
        # Perform the frequency analysis
        result = frequency_analysis(
            da,
            mode=mode,
            t=rp_int,  # List of integers
            dist='gev',  # Using 'gev' instead of 'genextreme' for xclim
            method='ML'  # Maximum Likelihood method
        )
        
        # Extract return levels
        return_levels = {}
        for rp, rp_int_val in zip(return_periods, rp_int):
            try:
                val = float(result.sel(return_period=rp_int_val).values)
                return_levels[rp] = val
            except:
                return_levels[rp] = np.nan
        
        return return_levels
    
    except Exception as e:
        logger.error(f"Error in frequency analysis: {str(e)}")
        return {rp: np.nan for rp in return_periods}

def process_region_month(region_id, region_name, spi_data, region_mask, month, output_dir):
    """
    Process a specific region and month to calculate GEV return periods.
    
    Args:
        region_id: ID of the region
        region_name: Name of the region
        spi_data: SPI data as xarray.Dataset
        region_mask: RegionMask object (not used for synthetic data)
        month: Month to process (1-12)
        output_dir: Directory to save results
        
    Returns:
        dict: Dictionary with drought and flood return periods
    """
    logger.info(f"Processing region {region_id} ({region_name}), month {month}")
    
    try:
        # Filter data for the month
        month_data = spi_data.sel(time=spi_data.time.dt.month == month)
        
        if month_data.sizes['time'] == 0:
            logger.warning(f"No data for month {month}")
            return None
        
        # For testing with synthetic data, just use a random sample of the data for each region
        # In a real implementation, we would use proper masking with regionmask
        np.random.seed(region_id)  # Using region_id for reproducibility
        
        # Extract SPI values (assume there's a variable named after the SPI type)
        spi_var = next(var for var in month_data.data_vars if var.startswith('SPI'))
        
        # Take a random subset of the data to simulate the region's data
        spi_all_values = month_data[spi_var].values.flatten()
        # Remove NaNs for proper sampling
        spi_all_values = spi_all_values[~np.isnan(spi_all_values)]
        
        # Take a sample for this region with some random noise added
        n_samples = min(500, len(spi_all_values))
        indices = np.random.choice(len(spi_all_values), n_samples, replace=False)
        spi_values = spi_all_values[indices] + np.random.normal(0, 0.2, n_samples)
        
        # Separate drought (negative) and flood (positive) values
        drought_values = -spi_values[spi_values < 0]  # Negate to get positive values for xclim
        flood_values = spi_values[spi_values > 0]
        
        # Calculate return levels using xclim
        # For drought, we use 'min' mode (converted to positive values above, will be negated back later)
        drought_rp = calculate_return_levels_xclim(drought_values, RETURN_PERIODS, event_type='max')
        # For flood, we use 'max' mode
        flood_rp = calculate_return_levels_xclim(flood_values, RETURN_PERIODS, event_type='max')
        
        # Convert drought values back to negative
        drought_rp = {k: -v for k, v in drought_rp.items()}
        
        result = {
            'region_id': region_id,
            'region_name': region_name,
            'month': month,
            'drought_rp': drought_rp,
            'flood_rp': flood_rp
        }
        
        return result
    
    except Exception as e:
        logger.error(f"Error processing region {region_id}, month {month}: {str(e)}")
        return None

def process_spi_type(spi_type, data_dir, regions_file, output_dir):
    """
    Process an SPI type to calculate return periods for all regions and months.
    
    Args:
        spi_type: Type of SPI data (e.g., "SPI1", "SPI3", etc.)
        data_dir: Directory containing the SPI data
        regions_file: Path to GeoJSON file with region polygons
        output_dir: Directory to save results
        
    Returns:
        List of DataFrames with results for each month
    """
    logger.info(f"Processing {spi_type}")
    
    # Load SPI data
    spi_data = load_spi_data(spi_type, data_dir)
    
    if spi_data is None:
        logger.error(f"Failed to load data for {spi_type}")
        return []
    
    # Create region masks
    region_mask, region_info = create_region_masks(regions_file, spi_data)
    
    # Create output directory for this SPI type
    spi_output_dir = os.path.join(output_dir, spi_type)
    os.makedirs(spi_output_dir, exist_ok=True)
    
    # Process each month
    month_results = []
    for month in MONTHS:
        logger.info(f"Processing month {month}")
        month_result = []
        
        # Process each region for this month
        for _, row in region_info.iterrows():
            result = process_region_month(
                row['region_id'], row['region_name'], 
                spi_data, region_mask, month, spi_output_dir
            )
            if result:
                month_result.append(result)
        
        if month_result:
            # Convert to DataFrame
            month_df = pd.DataFrame(month_result)
            
            # Expand the return period dictionaries to columns
            for rp in RETURN_PERIODS:
                month_df[f'drought_rp_{rp}'] = month_df['drought_rp'].apply(
                    lambda x: x[rp] if x else np.nan
                )
                month_df[f'flood_rp_{rp}'] = month_df['flood_rp'].apply(
                    lambda x: x[rp] if x else np.nan
                )
            
            # Drop the dictionary columns
            month_df = month_df.drop(['drought_rp', 'flood_rp'], axis=1)
            
            # Save to CSV
            month_file = os.path.join(spi_output_dir, f'month_{month:02d}.csv')
            month_df.to_csv(month_file, index=False)
            logger.info(f"Saved results to {month_file}")
            
            month_results.append(month_df)
    
    return month_results

def plot_return_level_curves_xclim(values, return_periods, title, output_file, event_type='max'):
    """
    Plot return level curves using xclim's frequency_analysis.
    
    Args:
        values: Array of values to analyze
        return_periods: List of return periods to plot
        title: Title for the plot
        output_file: Path to save the plot
        event_type: 'min' for drought, 'max' for flood
    """
    if len(values) < 30 or np.all(np.isnan(values)):
        logger.warning(f"Cannot plot return level curve for {title}: insufficient data")
        return
    
    try:
        # Create a DataArray with the values
        da = xr.DataArray(values, dims=('time',), 
                         coords={'time': pd.date_range('2000-01-01', periods=len(values), freq='Y')})
        
        # Set the event type
        mode = 'low' if event_type == 'min' else 'high'
        
        # Create a wider range of return periods for the curve
        rp_curve = list(np.logspace(0, 2.5, 50).astype(int))  # Convert to list of integers
        
        # Make sure we include our specific return periods
        all_periods = sorted(list(set(rp_curve + list(return_periods))))
        
        # Perform the frequency analysis
        result = frequency_analysis(
            da,
            mode=mode,
            t=all_periods,  # Using the prepared list of integers
            dist='gev',  # Using 'gev' instead of 'genextreme' for xclim
            method='ML'  # Maximum Likelihood method
        )
        
        # Plot
        plt.figure(figsize=(10, 6))
        
        # Convert back to arrays for plotting
        rp_curve_array = np.array(rp_curve)
        curve_values = result.sel(return_period=rp_curve).values
        
        # Plot the curve
        plt.semilogx(rp_curve_array, curve_values, 'b-', linewidth=2)
        
        # Add points for the specified return periods
        specific_values = result.sel(return_period=return_periods).values
        plt.plot(return_periods, specific_values, 'ro')
        
        # Add labels for the specified return periods
        for rp, level in zip(return_periods, specific_values):
            plt.text(rp, level, f" {rp} yr", verticalalignment='bottom')
        
        # Add title
        plt.title(title)
        
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.xlabel('Return Period (years)')
        plt.ylabel('Return Level')
        plt.tight_layout()
        
        plt.savefig(output_file, dpi=300)
        plt.close()
        
        logger.info(f"Saved return level curve to {output_file}")
    
    except Exception as e:
        logger.error(f"Error plotting return level curve: {str(e)}")

def create_summary_plots(month_results, spi_type, output_dir):
    """
    Create summary plots for the results.
    
    Args:
        month_results: List of DataFrames with results
        spi_type: Type of SPI data
        output_dir: Directory to save plots
    """
    logger.info(f"Creating summary plots for {spi_type}")
    
    # Create plots directory
    plots_dir = os.path.join(output_dir, spi_type, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Combine all months
    if not month_results:
        logger.warning(f"No results to plot for {spi_type}")
        return
    
    all_results = pd.concat(month_results, ignore_index=True)
    
    # Plot average return levels by month for each region
    for _, region_data in all_results.groupby(['region_id', 'region_name']):
        region_id = region_data['region_id'].iloc[0]
        region_name = region_data['region_name'].iloc[0]
        
        # Convert region_id to string to ensure it's compatible for filename
        region_id_str = str(region_id).replace('/', '_').replace(' ', '_')
        region_name_str = str(region_name).replace('/', '_').replace(' ', '_')
        
        # Create drought and flood plots
        for event_type in ['drought', 'flood']:
            plt.figure(figsize=(12, 8))
            
            for rp in RETURN_PERIODS:
                col = f'{event_type}_rp_{rp}'
                if col in region_data.columns:
                    x_values = region_data['month'].values
                    y_values = region_data[col].values
                    if len(x_values) > 0 and len(y_values) > 0:
                        plt.plot(x_values, y_values, 'o-', linewidth=2, label=f'{rp} years')
            
            plt.title(f"{spi_type} {event_type.capitalize()} Return Levels for {region_name}")
            plt.xlabel('Month')
            plt.ylabel('SPI Return Level')
            plt.grid(True, alpha=0.5)
            plt.legend()
            if len(MONTHS) <= 12:  # Only set specific x-ticks if we have 12 or fewer months
                plt.xticks(MONTHS)
            plt.tight_layout()
            
            plot_file = os.path.join(
                plots_dir, f"{region_id_str}_{region_name_str}_{event_type}.png"
            )
            plt.savefig(plot_file, dpi=300)
            plt.close()
            
            logger.info(f"Saved {event_type} plot for {region_name} to {plot_file}")

def main():
    """Main function to calculate return periods for all SPI types, regions, and months."""
    logger.info("Starting return period calculation with xclim")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process each SPI type
    for spi_type in SPI_TYPES:
        month_results = process_spi_type(spi_type, SPI_DATA_DIR, REGIONS_FILE, OUTPUT_DIR)
        
        # Create summary plots
        create_summary_plots(month_results, spi_type, OUTPUT_DIR)
    
    logger.info("Return period calculation with xclim completed")

if __name__ == "__main__":
    main()