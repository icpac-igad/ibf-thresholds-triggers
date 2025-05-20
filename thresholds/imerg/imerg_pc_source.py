"""
East Africa Precipitation Return Period Analysis with GPM IMERG dataset
- Subsets GPM IMERG data to East Africa region
- Loads and processes admin boundary polygons
- Aggregates precipitation at multiple time intervals
- Performs GEV return period analysis for each region and aggregation level
"""

import os
import numpy as np
import xarray as xr
import geopandas as gpd
import regionmask
import pystac_client
import planetary_computer
import fsspec
import matplotlib.pyplot as plt
from dotenv import load_dotenv, find_dotenv
import xclim.indices as xci
from xclim.frequency import fit, get_return_level
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default configuration values
DEFAULT_CONFIG = {
    'SERVICE_ACCOUNT_KEY': 'coiled-data-e4drr.json',
    'BUCKET_NAME': 'gefs-wgl',
    'EXTENT': [21.85, -11.72, 51.50, 23.14],  # [min_lon, min_lat, max_lon, max_lat]
    'START_TIME': '0h',
    'END_TIME': '21h',
    'STORE': 'https://data.dynamical.org/noaa/gefs/forecast-35-day/latest.zarr',
    'UPLOAD_TO_GCS': False  # Default to not upload to GCS
}

def load_environment(env_file=None):
    """
    Load environment variables from specified .env file
    If no file is specified, it will try to find .env in the current directory
    
    Args:
        env_file: Path to the .env file (optional)
    
    Returns:
        dict: Configuration dictionary with loaded values
    """
    # If env_file is specified, load it
    if env_file and os.path.exists(env_file):
        logger.info(f"Loading environment from: {env_file}")
        load_dotenv(env_file)
    # Otherwise, try to find .env in the current directory
    else:
        env_path = find_dotenv()
        if env_path:
            logger.info(f"Loading environment from: {env_path}")
            load_dotenv(env_path)
        else:
            logger.info("Warning: No .env file found, using default values")
    
    # Load configuration using DEFAULT_CONFIG as fallback values
    config = {
        'SERVICE_ACCOUNT_KEY': os.getenv('SERVICE_ACCOUNT_KEY', DEFAULT_CONFIG['SERVICE_ACCOUNT_KEY']),
        'BUCKET_NAME': os.getenv('BUCKET_NAME', DEFAULT_CONFIG['BUCKET_NAME']),
        'EXTENT': [
            float(os.getenv('EXTENT_X1', str(DEFAULT_CONFIG['EXTENT'][0]))),
            float(os.getenv('EXTENT_Y1', str(DEFAULT_CONFIG['EXTENT'][1]))),
            float(os.getenv('EXTENT_X2', str(DEFAULT_CONFIG['EXTENT'][2]))),
            float(os.getenv('EXTENT_Y2', str(DEFAULT_CONFIG['EXTENT'][3])))
        ],
        'START_TIME': os.getenv('START_TIME', DEFAULT_CONFIG['START_TIME']),
        'END_TIME': os.getenv('END_TIME', DEFAULT_CONFIG['END_TIME']),
        'STORE': os.getenv('STORE', DEFAULT_CONFIG['STORE']),
        'UPLOAD_TO_GCS': os.getenv('UPLOAD_TO_GCS', str(DEFAULT_CONFIG['UPLOAD_TO_GCS'])).lower() == 'true',
        'PROCESS_PROBS': os.getenv('PROCESS_PROBS', 'false').lower() == 'true'
    }
    
    return config

def load_gpm_imerg_data():
    """
    Load the GPM IMERG dataset from Planetary Computer
    
    Returns:
        xarray.Dataset: Lazily loaded GPM IMERG dataset
    """
    logger.info("Opening connection to Planetary Computer STAC catalog")
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    
    logger.info("Getting GPM IMERG collection")
    collection = catalog.get_collection("gpm-imerg-hhr")
    asset = collection.assets["zarr-abfs"]
    
    logger.info("Loading GPM IMERG dataset as xarray")
    ds = xr.open_zarr(
        asset.href,
        **asset.extra_fields["xarray:open_kwargs"],
        storage_options=asset.extra_fields["xarray:storage_options"]
    )
    
    logger.info(f"Dataset loaded with dimensions: {ds.dims}")
    return ds

def subset_dataset_to_east_africa(ds, config):
    """
    Subset dataset to East Africa region based on extent in config
    
    Args:
        ds: xarray Dataset
        config: Configuration dictionary with EXTENT
        
    Returns:
        xarray.Dataset: Subset dataset
    """
    extent = config['EXTENT']
    logger.info(f"Subsetting dataset to East Africa region: {extent}")
    
    # Extract extent coordinates
    min_lon, min_lat, max_lon, max_lat = extent
    
    # Subset dataset to the East Africa region
    ds_subset = ds.sel(lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat))
    
    logger.info(f"Subset dataset dimensions: {ds_subset.dims}")
    return ds_subset

def load_admin_boundaries(geojson_path):
    """
    Load administrative boundaries from GeoJSON file
    
    Args:
        geojson_path: Path to GeoJSON file
        
    Returns:
        geopandas.GeoDataFrame: Admin boundaries
    """
    logger.info(f"Loading administrative boundaries from: {geojson_path}")
    gdf = gpd.read_file(geojson_path)
    logger.info(f"Loaded {len(gdf)} administrative boundaries")
    return gdf

def create_region_masks(ds_subset, admin_gdf):
    """
    Create region masks for each administrative boundary
    
    Args:
        ds_subset: Subset xarray Dataset
        admin_gdf: GeoDataFrame with administrative boundaries
        
    Returns:
        xarray.DataArray: Region masks
    """
    logger.info("Creating region masks from administrative boundaries")
    
    # Create masks for each admin region
    masks = regionmask.from_geopandas(admin_gdf, 
                                      names=admin_gdf.index.tolist(),
                                      name="admin_region")
    
    # Create mask data array for the dataset grid
    mask_da = masks.mask(ds_subset, lon_name="lon", lat_name="lat")
    
    logger.info(f"Created {len(admin_gdf)} region masks")
    return mask_da

def aggregate_precipitation(ds_subset, aggregation_periods):
    """
    Aggregate precipitation data over multiple time periods
    
    Args:
        ds_subset: Subset xarray Dataset
        aggregation_periods: List of aggregation periods in minutes
        
    Returns:
        dict: Dictionary of aggregated datasets by aggregation period
    """
    logger.info(f"Aggregating precipitation over periods: {aggregation_periods}")
    
    # Get precipitation variable (assuming 'precipitationCal' is the variable name)
    precip_var = 'precipitationCal'
    
    # Create dictionary to store aggregated datasets
    aggregated_datasets = {}
    
    # Aggregate precipitation for each time period
    for period_min in aggregation_periods:
        period_name = f"{period_min}min"
        if period_min == 30:
            # 30-minute data is the native resolution, no need to aggregate
            aggregated_datasets[period_name] = ds_subset[precip_var]
            continue
            
        # Calculate number of 30-minute periods to aggregate
        n_periods = period_min // 30
        
        # Resample and aggregate
        logger.info(f"Aggregating to {period_name}")
        if period_min < 1440:  # Less than a day
            # For sub-daily aggregations, use rolling method
            aggregated = ds_subset[precip_var].rolling(time=n_periods).sum()
            # Take every n_periods step to avoid overlapping windows
            aggregated = aggregated.isel(time=slice(n_periods-1, None, n_periods))
        else:
            # For daily or longer aggregations, use resample
            freq = 'D' if period_min == 1440 else f"{period_min//1440}D"
            aggregated = ds_subset[precip_var].resample(time=freq).sum()
        
        aggregated_datasets[period_name] = aggregated
        logger.info(f"Completed {period_name} aggregation, shape: {aggregated.shape}")
    
    return aggregated_datasets

def calculate_return_periods(masked_data, return_periods):
    """
    Calculate return periods using GEV distribution for each masked region
    
    Args:
        masked_data: Dictionary of masked data by region and aggregation period
        return_periods: List of return periods in years
        
    Returns:
        dict: Dictionary of return level results
    """
    logger.info(f"Calculating return periods: {return_periods}")
    
    results = {}
    
    for region_name, region_data in masked_data.items():
        results[region_name] = {}
        
        for agg_period, data_array in region_data.items():
            logger.info(f"Fitting GEV for region {region_name}, aggregation {agg_period}")
            
            # Get annual maximum series
            annual_max = data_array.groupby('time.year').max('time')
            
            try:
                # Fit GEV distribution
                fit_result = fit(annual_max, dist='gev')
                
                # Calculate return levels for each return period
                return_levels = {}
                for rp in return_periods:
                    return_level = get_return_level(fit_result, return_period=rp)
                    return_levels[f"{rp}yr"] = return_level
                
                results[region_name][agg_period] = {
                    'fit': fit_result,
                    'return_levels': return_levels
                }
                
                logger.info(f"Completed GEV analysis for {region_name}, {agg_period}")
            except Exception as e:
                logger.error(f"Error fitting GEV for {region_name}, {agg_period}: {e}")
                results[region_name][agg_period] = {
                    'fit': None,
                    'return_levels': None,
                    'error': str(e)
                }
    
    return results

def main(geojson_path, time_range=None):
    """
    Main function to run the analysis pipeline
    
    Args:
        geojson_path: Path to GeoJSON with administrative boundaries
        time_range: Optional tuple of (start_date, end_date) to subset time
        
    Returns:
        dict: Dictionary with analysis results
    """
    # Load configuration
    config = load_environment()
    
    # Load GPM IMERG dataset
    ds = load_gpm_imerg_data()
    
    # Subset to time range if specified
    if time_range:
        start_date, end_date = time_range
        ds = ds.sel(time=slice(start_date, end_date))
    
    # Subset to East Africa region
    ds_subset = subset_dataset_to_east_africa(ds, config)
    
    # Load administrative boundaries
    admin_gdf = load_admin_boundaries(geojson_path)
    
    # Create region masks
    region_masks = create_region_masks(ds_subset, admin_gdf)
    
    # Define aggregation periods in minutes
    # 30min, 1h, 2h, 3h, 6h, 12h, 18h, 24h, 48h, 72h, 1 week
    aggregation_periods = [30, 60, 120, 180, 360, 720, 1080, 1440, 2880, 4320, 10080]
    
    # Aggregate precipitation
    aggregated_data = aggregate_precipitation(ds_subset, aggregation_periods)
    
    # Create dictionary to store masked data by region and aggregation period
    masked_data = {}
    
    # Apply region masks to each aggregated dataset
    for region_id in admin_gdf.index:
        region_name = admin_gdf.loc[region_id, 'name'] if 'name' in admin_gdf.columns else f"Region_{region_id}"
        masked_data[region_name] = {}
        
        for agg_period, agg_data in aggregated_data.items():
            # Mask data for this region
            region_mask = region_masks == region_id
            masked = agg_data.where(region_mask)
            
            # Spatial average over the masked region (ignoring NaNs)
            region_avg = masked.mean(dim=['lon', 'lat'], skipna=True)
            
            masked_data[region_name][agg_period] = region_avg
    
    # Define return periods in years
    return_periods = [2, 4, 8, 10, 15, 20, 40, 60, 100]
    
    # Calculate return periods
    results = calculate_return_periods(masked_data, return_periods)
    
    return results

def plot_return_periods(results, output_dir='./plots'):
    """
    Plot return periods for each region and aggregation period
    
    Args:
        results: Results dictionary from main analysis
        output_dir: Directory to save plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for region_name, region_results in results.items():
        for agg_period, agg_results in region_results.items():
            if agg_results['fit'] is None:
                logger.warning(f"Skipping plot for {region_name}, {agg_period}: No fit available")
                continue
                
            fit_result = agg_results['fit']
            return_levels = agg_results['return_levels']
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Get return periods and levels
            periods = [int(k.replace('yr', '')) for k in return_levels.keys()]
            levels = [float(v) for v in return_levels.values()]
            
            # Plot return levels
            ax.semilogx(periods, levels, 'o-', linewidth=2)
            
            # Set labels and title
            ax.set_xlabel('Return Period (years)')
            ax.set_ylabel('Precipitation (mm)')
            ax.set_title(f'Return Periods for {region_name}, Aggregation: {agg_period}')
            
            # Add grid
            ax.grid(True, which="both", ls="-")
            
            # Save figure
            fig_path = os.path.join(output_dir, f'return_periods_{region_name}_{agg_period}.png')
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"Saved plot to {fig_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Precipitation Return Period Analysis')
    parser.add_argument('geojson_path', help='Path to GeoJSON file with administrative boundaries')
    parser.add_argument('--start_date', help='Start date for analysis (YYYY-MM-DD)')
    parser.add_argument('--end_date', help='End date for analysis (YYYY-MM-DD)')
    parser.add_argument('--output_dir', default='./results', help='Directory for output files')
    
    args = parser.parse_args()
    
    # Set time range if specified
    time_range = None
    if args.start_date and args.end_date:
        time_range = (args.start_date, args.end_date)
    
    # Run analysis
    results = main(args.geojson_path, time_range)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Plot results
    plot_return_periods(results, os.path.join(args.output_dir, 'plots'))
    
    logger.info(f"Analysis completed. Results saved to {args.output_dir}")
