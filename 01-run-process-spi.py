import sys
import pandas as pd
import fsspec
from google.oauth2 import service_account
import time
import dask.dataframe as dd
import xarray as xr
from shapely import wkb
import geopandas as gpd
from xclim.indices.stats import standardized_index_fit_params
from xclim.indices import standardized_precipitation_index
import xesmf as xe
import numpy as np
import os
import regionmask
import logging
import argparse 

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def get_credentials(service_account_json):
    """Create and return Google Cloud credentials."""
    credentials = service_account.Credentials.from_service_account_file(
        service_account_json,
        scopes=["https://www.googleapis.com/auth/devstorage.read_only"],
    )
    return credentials

def old_get_region_bounds(region_id, credentials, buffer=0.5):
    """
    Get geographic bounds for a specific region with buffer.
    
    Args:
        region_id (str): The region identifier to filter by
        credentials: Google Cloud credentials
        buffer (float): Buffer in degrees to add around the region extent
        
    Returns:
        tuple: (lat_min, lat_max, lon_min, lon_max)
    """
    gcs_file_url = 'gs://seas51/ea_admin0_2_custom_polygon_shapefile_v5.parquet'
    ddf = dd.read_parquet(gcs_file_url, storage_options={'token': credentials}, engine='pyarrow')
    
    # Filter by region id
    fdf = ddf[ddf['gbid'].str.contains(region_id)]
    df1 = fdf.compute()
    
    if len(df1) == 0:
        raise ValueError(f"No regions found with id containing '{region_id}'")
    
    # Convert geometry from WKB to shapely geometry
    df1['geometry'] = df1['geometry'].apply(wkb.loads)
    gdf = gpd.GeoDataFrame(df1, geometry='geometry')
    
    # Get bounds
    bounds = gdf.bounds
    lat_min = bounds['miny'].min() - buffer
    lat_max = bounds['maxy'].max() + buffer
    lon_min = bounds['minx'].min() - buffer
    lon_max = bounds['maxx'].max() + buffer
    extent =  [lat_min, lat_max, lon_min, lon_max]  
    return gdf, extent 


def get_region_bounds(region_id, credentials=None, buffer=0.5, use_local=False, local_shapefile_path='../kmj_polygon.shp'):
    """
    Get geographic bounds for a specific region with buffer, supporting both local and GCS data sources.
    
    Args:
        region_id (str): The region identifier to filter by
        credentials: Google Cloud credentials (required if use_local=False)
        buffer (float): Buffer in degrees to add around the region extent
        use_local (bool): Whether to use local shapefile (True) or GCS (False)
        local_shapefile_path (str): Path to local shapefile if use_local=True
        
    Returns:
        tuple: (gdf, extent) where:
              - gdf is a GeoDataFrame containing the filtered region data
              - extent is [lat_min, lat_max, lon_min, lon_max]
    """
   
    if use_local:
        # Read from local shapefile
        try:
            gdf = gpd.read_file(local_shapefile_path)
            
            # Filter by region id (assuming column name is 'gbid', adjust if different)
            if 'gbid' in gdf.columns:
                gdf = gdf[gdf['gbid'].str.contains(region_id)]
            else:
                # If 'gbid' column doesn't exist, try to find a suitable ID column
                id_columns = [col for col in gdf.columns if 'id' in col.lower()]
                if id_columns:
                    gdf = gdf[gdf[id_columns[0]].astype(str).str.contains(region_id)]
                else:
                    raise ValueError("No suitable ID column found in local shapefile")
            
            if len(gdf) == 0:
                raise ValueError(f"No regions found with id containing '{region_id}' in local shapefile")
                
        except Exception as e:
            raise ValueError(f"Error reading local shapefile: {str(e)}")
    else:
        # Read from GCS
        if credentials is None:
            raise ValueError("Credentials required when not using local shapefile")
            
        import dask.dataframe as dd
        
        gcs_file_url = 'gs://seas51/ea_admin0_2_custom_polygon_shapefile_v5.parquet'
        ddf = dd.read_parquet(gcs_file_url, storage_options={'token': credentials}, engine='pyarrow')
        
        # Filter by region id
        fdf = ddf[ddf['gbid'].str.contains(region_id)]
        df1 = fdf.compute()
        
        if len(df1) == 0:
            raise ValueError(f"No regions found with id containing '{region_id}' in GCS dataset")
        
        # Convert geometry from WKB to shapely geometry
        df1['geometry'] = df1['geometry'].apply(wkb.loads)
        gdf = gpd.GeoDataFrame(df1, geometry='geometry')
    
     # Get bounds
    bounds = gdf.bounds
    lat_min = bounds['miny'].min() - buffer
    lat_max = bounds['maxy'].max() + buffer
    lon_min = bounds['minx'].min() - buffer
    lon_max = bounds['maxx'].max() + buffer
    extent = [lat_min, lat_max, lon_min, lon_max]
    
    return gdf, extent




def process_chirps_data(region_id, credentials, extent, chirps_file=None, output_dir='.', res=0.25):
    """
    Process CHIRPS observation data for a region and calculate SPI-3.
    
    Args:
        region_id (str): Region identifier 
        credentials: Google Cloud credentials
        output_dir (str): Directory to save output files
        res (float): Resolution in degrees for regridding
        
    Returns:
        str: Path to the created netCDF file
    """
    # Get region bounds and mask
    #gdf, lat_min, lat_max, lon_min, lon_max = get_region_bounds(region_id, credentials)
    # Log region info
    lat_min, lat_max, lon_min, lon_max = extent
    logger.info(f"Processing region: {region_id}")
    logger.info(f"Region bounds: lat_min={extent[0]}, lat_max={extent[1]}, lon_min={extent[2]}, lon_max={extent[3]}")
    logger.info(f"Processing CHIRPS data for region {region_id} using extent {extent}...")
    start = time.time()

    if chirps_file:
        logger.info(f"Using local CHIRPS file: {chirps_file}")
        ds = xr.open_dataset(chirps_file)
    else:
        logger.info(f"Loading CHIRPS data from GCP")
        uri = 'gs://seas51/ea_chirps_v3_monthly_20250312.zarr'
        ds = xr.open_dataset(uri, engine='zarr', consolidated=False, storage_options={'token': credentials})
    
       
    # Check if latitude is decreasing
    lat_decreasing = ds.latitude.values[0] > ds.latitude.values[-1]
    logger.info(f"CHIRPS latitude is {'decreasing' if lat_decreasing else 'increasing'}")
    
    # Subset to region with appropriate ordering
    if lat_decreasing:
        ds_subset = ds.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
    else:
        ds_subset = ds.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))
    
    ds_subset_computed = ds_subset.compute()
    logger.info(f'CHIRPS data loaded in {time.time() - start:.2f} seconds')
    logger.info(f'CHIRPS data shape: {ds_subset_computed.dims}')
    
    # Regrid to regular grid
    ds1 = ds_subset_computed.rename({'longitude': 'lon', 'latitude': 'lat'})
    dr = ds1["precip"]
    
    # Create output grid with regular spacing
    ds_out = xr.Dataset({
        "lat": (["lat"], np.arange(lat_min, lat_max, res), {"units": "degrees_north"}),
        "lon": (["lon"], np.arange(lon_min, lon_max, res), {"units": "degrees_east"}),
    })
    
    # Regrid
    logger.info("Regridding CHIRPS data to regular 25km grid")
    regridder = xe.Regridder(ds1, ds_out, "conservative")
    dr_out = regridder(dr, keep_attrs=True)
    ds2 = dr_out.to_dataset()
    
    # Apply region mask to regridded data
    #logger.info("Applying region mask to regridded CHIRPS data")
    #ds2_masked = apply_region_mask(ds2, region_mask)
    
    # Check for NaN values
    nan_count = np.isnan(ds2.precip.values).sum()
    total_cells = np.prod(ds2.precip.shape)
    logger.info(f"NaN cells after masking: {nan_count} out of {total_cells} ({nan_count/total_cells*100:.2f}%)")
    
    # Calculate SPI-3
    logger.info("Calculating SPI-3 for CHIRPS observations")
    ds2['precip'].attrs['units'] = 'mm/month'
    aa = ds2.precip
    
    # Calculate SPI-3
    spi_3 = standardized_precipitation_index(
        aa,
        freq="MS",
        window=3,
        dist="gamma",
        method="APP",
        cal_start='1991-01-01',
        cal_end='2018-01-01',
        fitkwargs={"floc": 0}
    )
    
    a_s3 = spi_3.compute()
    ch_spi = a_s3.to_dataset(name='spi3')
    
    # Save to NetCDF
    output_file = os.path.join(output_dir, f'{region_id}_obs_spi3.nc')
    ch_spi.to_netcdf(output_file)
    
    logger.info(f"Observation SPI-3 saved to {output_file}")
    return output_file


def merge_grib_files(main_file, additional_files, output_file=None):
    """
    Merge multiple GRIB files, removing any duplicate time periods.

    Args:
        main_file (str): Path to the main GRIB file (historical data)
        additional_files (list): List of paths to additional GRIB files
        output_file (str): Optional path to save the merged file

    Returns:
        xarray.Dataset: The merged dataset
    """
    logger.info(f"Merging main GRIB file: {main_file} with additional files")

    # Backend kwargs for cfgrib - use 'warn' to skip corrupted messages gracefully
    cfgrib_kwargs = dict(
        time_dims=('forecastMonth', 'time'),
        errors='warn'  # Skip corrupted messages with warning instead of failing
    )

    # Open the main file
    if main_file.endswith('.grib') or main_file.endswith('.grb') or main_file.endswith('.grib2'):
        main_ds = xr.open_dataset(main_file, engine='cfgrib',
                                 backend_kwargs=cfgrib_kwargs)
    else:
        main_ds = xr.open_dataset(main_file)

    logger.info(f"Main file loaded. Time range: {main_ds.time.values[0]} to {main_ds.time.values[-1]}, "
                f"Total time steps: {len(main_ds.time)}")

    # Process each additional file
    all_datasets = [main_ds]

    for add_file in additional_files:
        logger.info(f"Processing additional file: {add_file}")

        try:
            # Open the additional file
            if add_file.endswith('.grib') or add_file.endswith('.grb') or add_file.endswith('.grib2'):
                add_ds = xr.open_dataset(add_file, engine='cfgrib',
                                        backend_kwargs=cfgrib_kwargs)
            else:
                add_ds = xr.open_dataset(add_file)

            logger.info(f"Additional file loaded. Time range: {add_ds.time.values[0]} to {add_ds.time.values[-1]}, "
                        f"Total time steps: {len(add_ds.time)}")

            # Append to our list
            all_datasets.append(add_ds)

        except Exception as e:
            logger.error(f"Error processing file {add_file}: {e}")
            logger.warning(f"Skipping file {add_file} due to error. Processing will continue with remaining files.")
            continue
    
    # Handle merging carefully to ensure monotonic time index
    logger.info("Combining all datasets")
    
    try:
        # First attempt: Try to combine directly
        combined_ds = xr.combine_by_coords(all_datasets, combine_attrs="drop_conflicts", 
                                          data_vars="minimal", coords="minimal", compat="override")
    except ValueError as e:
        if "monotonic" in str(e) and "time" in str(e):
            logger.warning("Time coordinate not monotonic, attempting alternative merge approach")
            
            # Alternative approach: manually combine and sort time indices
            # First, concatenate all datasets along the time dimension
            concat_ds = xr.concat(all_datasets, dim="time")
            
            # Then remove duplicates by sorting and drop_duplicates
            time_values = concat_ds.time.values
            sorted_indices = np.argsort(time_values)
            sorted_ds = concat_ds.isel(time=sorted_indices)
            
            # Find unique time values
            _, unique_indices = np.unique(sorted_ds.time.values, return_index=True)
            combined_ds = sorted_ds.isel(time=unique_indices)
            
            logger.info(f"Successfully merged using alternative approach. Time periods: {len(combined_ds.time)}")
        else:
            # If it's not a monotonicity error, re-raise
            raise e
    
    # Remove duplicates if any
    # This assumes 'time' is the main coordinate for identifying duplicates
    if 'time' in combined_ds.coords:
        logger.info("Checking for duplicate time periods")
        time_vals = combined_ds['time'].values
        unique_times, indices = np.unique(time_vals, return_index=True)
        
        if len(indices) < len(time_vals):
            logger.info(f"Found {len(time_vals) - len(indices)} duplicate time periods - removing")
            combined_ds = combined_ds.isel(time=sorted(indices))
    
    # Explicitly sort by time to ensure monotonicity
    if 'time' in combined_ds.coords:
        combined_ds = combined_ds.sortby('time')
        logger.info(f"Sorted dataset by time. First time: {combined_ds.time.values[0]}, Last time: {combined_ds.time.values[-1]}")
    
    # Save to file if requested
    if output_file:
        logger.info(f"Saving merged dataset to {output_file}")
        combined_ds.to_netcdf(output_file)
    
    return combined_ds


def process_seas51_data(region_id, obs_file, credentials, extent, seas51_files=None, output_dir='.'):
    """
    Process SEAS51 forecast data for a region and calculate SPI-3.
    
    Args:
        region_id (str): Region identifier
        obs_file (str): Path to observation netCDF file for regridding
        credentials: Google Cloud credentials
        extent (tuple): Optional tuple of (lat_min, lat_max, lon_min, lon_max) 
        seas51_file (str): Optional path to local SEAS51 file
        output_dir (str): Directory to save output files
        
    Returns:
        str: Path to the created netCDF file
    """
    logger.info(f"Processing SEAS51 data for region {region_id}...")
    start = time.time()
    lat_min, lat_max, lon_min, lon_max = extent 
    
    # First, open the observation dataset to determine its extent
    logger.info(f"Opening observation dataset: {obs_file}")
    try:
        kn_obs = xr.open_dataset(obs_file)
        # Get the observation dataset extent
        obs_lat_min = float(kn_obs.lat.min().values)
        obs_lat_max = float(kn_obs.lat.max().values)
        obs_lon_min = float(kn_obs.lon.min().values)
        obs_lon_max = float(kn_obs.lon.max().values)
        
        # If no specific extent was provided, use the observation extent with a small buffer
        if extent is None:
            # Add a small buffer (e.g., 0.5 degrees) to ensure coverage
            buffer = 1.0
            lat_min = obs_lat_min - buffer
            lat_max = obs_lat_max + buffer
            lon_min = obs_lon_min - buffer
            lon_max = obs_lon_max + buffer
            logger.info(f"Using observation extent with buffer: {lat_min:.2f}, {lat_max:.2f}, {lon_min:.2f}, {lon_max:.2f}")
        else:
            lat_min, lat_max, lon_min, lon_max = extent
            logger.info(f"Using provided extent: {lat_min:.2f}, {lat_max:.2f}, {lon_min:.2f}, {lon_max:.2f}")
            
    except Exception as e:
        logger.error(f"Error opening observation file: {e}")
        raise
    
    # Load SEAS51 data
    # Backend kwargs for cfgrib - use 'warn' to skip corrupted messages gracefully
    cfgrib_kwargs = dict(
        time_dims=('forecastMonth', 'time'),
        errors='warn'  # Skip corrupted messages with warning instead of failing
    )

    if seas51_files:
        # Check if it's a single file or multiple files
        if isinstance(seas51_files, str):
            logger.info(f"Using single local SEAS51 file: {seas51_files}")
            if seas51_files.endswith('.grib') or seas51_files.endswith('.grb') or seas51_files.endswith('.grib2'):
                logger.info("Processing GRIB file format")
                sds = xr.open_dataset(seas51_files, engine='cfgrib',
                                     backend_kwargs=cfgrib_kwargs)
            else:
                sds = xr.open_dataset(seas51_files)
        else:
            # Multiple files - use the merge function
            logger.info(f"Merging {len(seas51_files)} SEAS51 files")
            main_file = seas51_files[0]
            additional_files = seas51_files[1:]
            merged_file = os.path.join(output_dir, f'{region_id}_merged_seas51.nc')
            sds = merge_grib_files(main_file, additional_files, output_file=merged_file)
    else:
        logger.info(f"Loading SEAS51 data from GCP")
        suri = 'gs://seas51/ea_seas51_20250312_v3.zarr'
        sds = xr.open_dataset(suri, engine='zarr', consolidated=False, storage_options={'token': credentials})
    
    # Subset to region (note: check the coordinate naming and order in the SEAS51 dataset)
    logger.info(f"Subsetting SEAS51 data to region")
    if 'latitude' in sds.dims:
        # Check if latitude is in descending order (common in some datasets)
        if sds.latitude[0] > sds.latitude[-1]:
            sds_subset = sds.sel(latitude=slice(lat_max+1, lat_min-1), longitude=slice(lon_min-1, lon_max+1))
        else:
            sds_subset = sds.sel(latitude=slice(lat_min-1, lat_max+1), longitude=slice(lon_min-1, lon_max+1))
    else:
        # Try alternative coordinate names
        if 'lat' in sds.dims:
            if sds.lat[0] > sds.lat[-1]:
                sds_subset = sds.sel(lat=slice(lat_max+1, lat_min-1), lon=slice(lon_min-1, lon_max+1))
            else:
                sds_subset = sds.sel(lat=slice(lat_min-1, lat_max+1), lon=slice(lon_min-1, lon_max+1))
    
    # Compute if using dask arrays
    if hasattr(sds_subset, 'compute'):
        sds_subset_computed = sds_subset.compute()
        logger.info(f'SEAS51 data loaded and computed in {time.time() - start:.2f} seconds')
    else:
        sds_subset_computed = sds_subset
        logger.info(f'SEAS51 data loaded in {time.time() - start:.2f} seconds')
    
    # Calculate SPI-3 for each lead time (1-6 months) using the new function
    all_leads = []
    
    for lead_val in range(1, 7):
        logger.info(f"Calculating SPI-3 for lead time {lead_val}...")
        
        # Use the new function for SPI calculation with parameter transfer
        cont_spi = vt_apply_spi3_with_parameter_transfer(sds_subset_computed, lead_val)
        
        # Check if we have valid results
        if not cont_spi:
            logger.warning(f"No valid members for lead time {lead_val}, skipping...")
            continue
            
        # Concatenate ensemble members
        try:
            lead_data = xr.concat(cont_spi, dim='member')
            all_leads.append(lead_data)
            logger.info(f"Successfully combined {len(cont_spi)} members for lead time {lead_val}")
        except Exception as e:
            logger.error(f"Error concatenating ensemble members for lead time {lead_val}: {e}")
            # Try to diagnose the issue
            if len(cont_spi) > 1:
                logger.debug(f"Diagnostic: First member shape: {cont_spi[0].shape}, Last member shape: {cont_spi[-1].shape}")
            continue
    
    if not all_leads:
        raise ValueError("No valid lead times processed. Cannot create forecast dataset.")
        
    # Concatenate all lead times
    ds_fc = xr.concat(all_leads, dim='lead')
    ds_fc = ds_fc.to_dataset(name='spi3')
    
    # Save raw forecast data
    raw_file = os.path.join(output_dir, f'{region_id}_raw_seas51_spi3.nc')
    ds_fc.to_netcdf(raw_file)
    logger.info(f"Raw forecast data saved to {raw_file}")
    
    # Regrid to observation grid
    cont_d = []
    
    for fm in range(min(6, len(all_leads))):
        logger.info(f"Regridding lead time {fm}...")
        try:
            ds_p_m1 = ds_fc.sel(lead=fm)
            
            # Create output grid matching observations
            ds_out = xr.Dataset({
                "lat": (["lat"], kn_obs['lat'].values, {"units": "degrees_north"}),
                "lon": (["lon"], kn_obs['lon'].values, {"units": "degrees_east"}),
            })
            
            # Rename coordinates for consistency if needed
            if 'longitude' in ds_p_m1.dims and 'latitude' in ds_p_m1.dims:
                gd2 = ds_p_m1.rename({'longitude': 'lon', 'latitude': 'lat'})
            else:
                gd2 = ds_p_m1
                
            # Check for coordinate consistency
            for coord in ['lat', 'lon']:
                if coord not in gd2.dims:
                    logger.error(f"Coordinate {coord} not found in forecast dataset")
                    raise ValueError(f"Coordinate {coord} not found in forecast dataset")
            
            # Perform regridding
            agd = gd2["spi3"]
            regridder = xe.Regridder(gd2, ds_out, "bilinear", periodic=False)
            dr_out = regridder(agd, keep_attrs=True)
            ds2 = dr_out.to_dataset()
            cont_d.append(ds2)
            
        except Exception as e:
            logger.error(f"Error regridding lead time {fm}: {e}")
            continue
    
    if not cont_d:
        raise ValueError("No lead times were successfully regridded")
        
    # Concatenate all regridded lead times
    kn_fct = xr.concat(cont_d, dim='lead')
    
    # Rename and add appropriate attributes
    if 'time' in kn_fct.dims or 'time' in kn_fct.coords:
        kn_fct = kn_fct.rename({'time': 'init'})
    
    if 'forecastMonth' in kn_fct.dims or 'forecastMonth' in kn_fct.coords:
        kn_fct = kn_fct.rename({'forecastMonth': 'lead'})
    
    if 'lead' in kn_fct.dims or 'lead' in kn_fct.coords:
        kn_fct['lead'].attrs['units'] = 'months'
    
    # Save to NetCDF
    output_file = os.path.join(output_dir, f'{region_id}_rgr_seas51_spi3.nc')
    kn_fct.to_netcdf(output_file)
    
    logger.info(f"Forecast SPI-3 saved to {output_file}")
    return output_file


def vt_apply_spi3_with_parameter_transfer(cont_db, lead_val):
    """
    Calculates SPI-3 for all ensemble members:
    - Members 0-24: Calculate normally using their own historical data
    - Members 25-51: Apply parameters derived from a reference member (e.g., member 0)
    
    Args:
        cont_db (xarray.Dataset): The SEAS51 dataset
        lead_val (int): The lead time value to process
        
    Returns:
        list: List of SPI-3 xarray.DataArrays for each ensemble member
    """
    lt1_db = cont_db.sel(forecastMonth=lead_val)
    lt1_db['tprate'].attrs['units'] = 'mm/month'
    cont_spi = []
    
    # Check if we have any members
    if 'number' not in lt1_db.dims:
        logger.error("No 'number' dimension found in dataset")
        return cont_spi
    # Process all members
    for nsl in lt1_db.number.values:
        try:
            logger.info(f"Processing ensemble member {nsl}")
            lt1_db2 = lt1_db.sel(number=nsl)
            aa = lt1_db2.tprate
            
            # Check data validity
            nan_count = np.isnan(aa.values).sum()
            if nan_count > 0:
                nan_percent = (nan_count / aa.size) * 100
                logger.warning(f"Warning: {nan_percent:.1f}% NaN values in input data for member {nsl}")
                if nan_percent > 90:
                    logger.warning(f"Skipping member {nsl} due to excessive NaNs in input")
                    continue
            
            if nsl < 25:
                # For members 0-24, calculate SPI normally with their own historical data
                spi_3 = standardized_precipitation_index(
                    aa,
                    freq="MS",
                    window=3,
                    dist="gamma",
                    method="APP",
                    cal_start='1991-01-01',
                    cal_end='2018-01-01',
                    fitkwargs={"floc": 0}
                )
            else:
                # Fall back to normal calculation
                spi_3 = standardized_precipitation_index(
                        aa,
                        freq="MS",
                        window=3,
                        dist="gamma",
                        method="APP",
                        cal_start='2017-01-01',
                        cal_end='2024-01-01',
                        fitkwargs={"floc": 0}
                )
            
            # Compute the SPI result
            a_s3 = spi_3.compute()
            
            # Check output validity
            spi_nan_count = np.isnan(a_s3.values).sum()
            if spi_nan_count > 0:
                spi_nan_percent = (spi_nan_count / a_s3.size) * 100
                logger.warning(f"Warning: {spi_nan_percent:.1f}% NaN values in SPI output for member {nsl}")
                
                # Only include if there are enough usable data points
                if spi_nan_percent > 95:
                    logger.warning(f"Skipping member {nsl} due to excessive NaNs in output")
                    continue
            
            # If "prob_of_zero" is missing, add it as a coordinate with NaN values
            if 'prob_of_zero' not in a_s3.coords:
                a_s3 = a_s3.assign_coords(
                    prob_of_zero=(('time', 'latitude', 'longitude'),
                                  np.full(a_s3.shape, np.nan))
                )
            
            cont_spi.append(a_s3)
            logger.info(f"Successfully processed ensemble member {nsl}")
            
        except Exception as e:
            logger.error(f"Error processing ensemble member {nsl}: {e}")
            continue
    
    logger.info(f"Processed {len(cont_spi)} out of {len(lt1_db.number)} members for lead time {lead_val}")
    return cont_spi# Example usage

def mask_netcdf_with_shapefile(forecast_path, obs_path, shapefile_df, buffer_size=0.25, 
                              output_forecast_path='masked_forecast.nc', 
                              output_obs_path='masked_obs.nc'):
    """
    Mask forecast and observation NetCDF files using a shapefile polygon.
    
    Parameters:
    -----------
    forecast_path : str
        Path to the forecast NetCDF file
    obs_path : str
        Path to the observation NetCDF file
    shapefile_path : str
        Path to the shapefile (.shp) containing the polygon
    buffer_size : float, optional
        Buffer size around the polygon in the same units as the coordinates (default: 0.25)
    output_forecast_path : str, optional
        Path where the masked forecast NetCDF will be saved
    output_obs_path : str, optional
        Path where the masked observation NetCDF will be saved
        
    Returns:
    --------
    tuple
        (masked_forecast, masked_obs) - the masked xarray Datasets
    """
    # Load datasets
    forecast_ds = xr.open_dataset(forecast_path)
    obs_ds = xr.open_dataset(obs_path)
    
    # Load shapefile
    gdf = shapefile_df
    
    # Create buffered geometry if needed
    buffered_gdf = gdf.copy()
    if buffer_size > 0:
        buffered_gdf['geometry'] = gdf.geometry.buffer(buffer_size)
    
    # Mask the forecast dataset
    # Extract coordinates
    f_lons = forecast_ds.lon.values
    f_lats = forecast_ds.lat.values
    
    # Create mask
    f_mask = regionmask.mask_geopandas(buffered_gdf.geometry, f_lons, f_lats)
    f_bool_mask = ~np.isnan(f_mask)
    
    # Expand mask to match forecast dimensions
    # Check if the dataset has the expected dimensions
    if 'lead' in forecast_ds.dims and 'member' in forecast_ds.dims and 'init' in forecast_ds.dims:
        f_expanded_mask = f_bool_mask.expand_dims({
            "lead": forecast_ds.lead,
            "member": forecast_ds.member,
            "init": forecast_ds.init
        })
    else:
        # Handle other dimension structures
        # This is just an example - adjust according to your specific dataset
        additional_dims = {}
        for dim in forecast_ds.dims:
            if dim not in ['lat', 'lon']:
                additional_dims[dim] = forecast_ds[dim]
        
        f_expanded_mask = f_bool_mask.expand_dims(additional_dims)
    
    # Apply mask
    masked_forecast = forecast_ds.copy(deep=True)
    for var in forecast_ds.data_vars:
        masked_forecast[var] = forecast_ds[var].where(f_expanded_mask, np.nan)
    
    # Mask the observation dataset
    # Extract coordinates
    o_lons = obs_ds.lon.values
    o_lats = obs_ds.lat.values
    
    # Create mask
    o_mask = regionmask.mask_geopandas(buffered_gdf.geometry, o_lons, o_lats)
    o_bool_mask = ~np.isnan(o_mask)
    
    # Expand mask to match observation dimensions
    # Check if the dataset has the expected 'time' dimension
    if 'time' in obs_ds.dims:
        o_expanded_mask = o_bool_mask.expand_dims({"time": obs_ds.time})
    else:
        # Handle other dimension structures
        additional_dims = {}
        for dim in obs_ds.dims:
            if dim not in ['lat', 'lon']:
                additional_dims[dim] = obs_ds[dim]
        
        o_expanded_mask = o_bool_mask.expand_dims(additional_dims)
    
    # Apply mask
    masked_obs = obs_ds.copy(deep=True)
    for var in obs_ds.data_vars:
        masked_obs[var] = obs_ds[var].where(o_expanded_mask, np.nan)
    
    # Save the masked datasets
    masked_forecast.to_netcdf(output_forecast_path)
    masked_obs.to_netcdf(output_obs_path)
    
    return output_obs_path , output_forecast_path 

def parse_arguments():
    """
    Parse command line arguments for the script.

    Returns:
        argparse.Namespace: The parsed arguments
    """
    description = """
SPI-3 Processing Script for Drought Anticipatory Action System

Calculates Standardized Precipitation Index (SPI-3) from CHIRPS observations
and/or SEAS51 forecasts for drought anticipatory action.

OPERATIONAL MODES:
  --mode both   : Mode 1 - Hindcast Verification (process CHIRPS + SEAS51)
  --mode chirps : Process only CHIRPS observations
  --mode seas51 : Mode 2 - Monthly Operational (process only SEAS51 forecasts)

For Mode 2 (seas51), an existing observation file (--obs-file) is required.
"""

    epilog = """
EXAMPLES:

  Mode 1 - Hindcast Verification (--mode both):
  ---------------------------------------------
  python 01-run-process-spi.py \\
      --region-id kmj \\
      --mode both \\
      --output-dir ./output \\
      --use-local \\
      --local-shapefile ./data/kmj_polygon.shp \\
      --chirps-file ./data/chirps-v2.0.monthly.nc \\
      --seas51-main-file ./data/seas5_precipitation_20260120_years1981-2025_months_12_months.grib \\
      --apply-mask \\
      --mask-buffer 0.25

  Mode 2 - Monthly Operational (--mode seas51):
  ---------------------------------------------
  python 01-run-process-spi.py \\
      --region-id kmj \\
      --mode seas51 \\
      --output-dir ./output \\
      --use-local \\
      --local-shapefile ./data/kmj_polygon.shp \\
      --obs-file ./output/kmj_obs_spi3.nc \\
      --seas51-main-file ./data/seas5_precipitation_20260120_years1981-2025_months_12_months.grib \\
      --seas51-additional-files ./data/seas5_precipitation_20260120_year2026_months_01.grib \\
      --apply-mask \\
      --mask-buffer 0.25

  CHIRPS only (create observation baseline):
  ------------------------------------------
  python 01-run-process-spi.py \\
      --region-id kmj \\
      --mode chirps \\
      --output-dir ./output \\
      --use-local \\
      --local-shapefile ./data/kmj_polygon.shp \\
      --chirps-file ./data/chirps-v2.0.monthly.nc

OUTPUT FILES:
  Mode 'both':   {region}_obs_spi3.nc, {region}_rgr_seas51_spi3.nc, + masked versions
  Mode 'chirps': {region}_obs_spi3.nc
  Mode 'seas51': {region}_rgr_seas51_spi3.nc, + masked version

WORKFLOW: First download data using 00-download-data.py, then process with this script.
"""

    parser = argparse.ArgumentParser(
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # General arguments
    parser.add_argument("--region-id", type=str, required=True,
                        help="Region identifier (e.g., 'kmj')")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Directory to save output files")
    parser.add_argument("--buffer", type=float, default=0.5,
                        help="Buffer size in degrees to add around region extent")
    
    # Mode selection (run both, only CHIRPS, or only SEAS51)
    parser.add_argument("--mode", type=str, choices=["both", "chirps", "seas51"], default="both",
                        help="Run mode: both=run both CHIRPS and SEAS51, chirps=only CHIRPS, seas51=only SEAS51")
                        
    # Local vs GCP data source options
    parser.add_argument("--use-local", action="store_true",
                        help="Use local shapefile instead of GCP")
    parser.add_argument("--local-shapefile", type=str, default="../kmj_polygon.shp",
                        help="Path to local shapefile if use-local is True")
    parser.add_argument("--credentials-file", type=str, default=None,
                        help="Path to GCP credentials JSON file")
    
    # CHIRPS options
    parser.add_argument("--chirps-file", type=str, default=None,
                        help="Path to local CHIRPS file (use GCP if not specified)")
    
    # SEAS51 options
    parser.add_argument("--seas51-main-file", type=str, default=None,
                        help="Path to main local SEAS51 GRIB file")
    parser.add_argument("--seas51-additional-files", type=str, nargs="+", default=[],
                        help="Paths to additional SEAS51 GRIB files to merge with the main file")
    parser.add_argument("--obs-file", type=str, default=None,
                        help="Path to observation file (required for SEAS51 only mode)")
    
    # Masking options
    parser.add_argument("--apply-mask", action="store_true",
                        help="Apply shapefile masking to output files")
    parser.add_argument("--mask-buffer", type=float, default=0.25,
                        help="Buffer size for masking in degrees")
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()

    # Setup output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Set up logging
    log_file = os.path.join(args.output_dir, f"{args.region_id}_spi_processing.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    
    logger.info(f"Starting SPI processing in {args.mode} mode for region {args.region_id}")
    
    # Get credentials if needed
    credentials = None
    if args.credentials_file:
        try:
            credentials = get_credentials(args.credentials_file)
            logger.info(f"Successfully loaded credentials from {args.credentials_file}")
        except Exception as e:
            logger.error(f"Failed to load credentials: {e}")
            if not args.use_local:
                logger.error("Cannot proceed without credentials when using GCP data")
                sys.exit(1)
    
    # Get region bounds
    try:
        if args.use_local:
            logger.info(f"Using local shapefile: {args.local_shapefile}")
            gdf, extent = get_region_bounds(args.region_id, use_local=True, 
                                         local_shapefile_path=args.local_shapefile,
                                         buffer=args.buffer)
        else:
            logger.info("Using GCP shapefile data")
            gdf, extent = get_region_bounds(args.region_id, credentials=credentials, 
                                         buffer=args.buffer)
        
        logger.info(f"Region extent: {extent}")
    except Exception as e:
        logger.error(f"Failed to get region bounds: {e}")
        sys.exit(1)
    
    obs_file = None
    fct_file = None
    
    # Process CHIRPS data if requested
    if args.mode in ["both", "chirps"]:
        try:
            logger.info("Starting CHIRPS observation data processing")
            obs_file = process_chirps_data(
                args.region_id,
                credentials,
                extent,
                chirps_file=args.chirps_file,
                output_dir=args.output_dir
            )
            logger.info(f"Successfully processed CHIRPS data: {obs_file}")
        except Exception as e:
            logger.error(f"Failed to process CHIRPS data: {e}")
            if args.mode == "chirps":
                sys.exit(1)
    
    # Process SEAS51 data if requested
    if args.mode in ["both", "seas51"]:
        # For SEAS51-only mode, we need an observation file
        if args.mode == "seas51" and not obs_file:
            if args.obs_file:
                obs_file = args.obs_file
                logger.info(f"Using provided observation file: {obs_file}")
            else:
                logger.error("An observation file is required for SEAS51-only mode")
                logger.error("Provide one with --obs-file or run in 'both' mode")
                sys.exit(1)
        
        try:
            logger.info("Starting SEAS51 forecast data processing")
            
            # Prepare SEAS51 files
            seas51_files = None
            if args.seas51_main_file:
                if args.seas51_additional_files:
                    seas51_files = [args.seas51_main_file] + args.seas51_additional_files
                    logger.info(f"Using main SEAS51 file {args.seas51_main_file} and " 
                               f"{len(args.seas51_additional_files)} additional files")
                else:
                    seas51_files = args.seas51_main_file
                    logger.info(f"Using single SEAS51 file: {args.seas51_main_file}")
            
            fct_file = process_seas51_data(
                args.region_id,
                obs_file,
                credentials,
                extent,
                seas51_files=seas51_files,
                output_dir=args.output_dir
            )
            logger.info(f"Successfully processed SEAS51 data: {fct_file}")
        except Exception as e:
            logger.error(f"Failed to process SEAS51 data: {e}")
            if args.mode == "seas51":
                sys.exit(1)
    
    # Apply masking if requested
    if args.apply_mask and obs_file and fct_file:
        try:
            logger.info("Applying shapefile masking to output files")
            masked_fct_path = os.path.join(args.output_dir, f'{args.region_id}_rgr_seas51_spi3_masked.nc')
            masked_obs_path = os.path.join(args.output_dir, f'{args.region_id}_obs_spi3_masked.nc')
            
            masked_obs, masked_fct = mask_netcdf_with_shapefile(
                forecast_path=fct_file,
                obs_path=obs_file,
                shapefile_df=gdf,
                buffer_size=args.mask_buffer,
                output_forecast_path=masked_fct_path,
                output_obs_path=masked_obs_path
            )
            
            logger.info(f"Masked datasets have been saved:")
            logger.info(f"  Masked observations: {masked_obs}")
            logger.info(f"  Masked forecast: {masked_fct}")
        except Exception as e:
            logger.error(f"Failed to apply masking: {e}")
    
    # Print summary at the end
    logger.info("=== Processing Summary ===")
    logger.info(f"Region: {args.region_id}")
    logger.info(f"Mode: {args.mode}")
    if obs_file:
        logger.info(f"Observation SPI-3 file: {obs_file}")
    if fct_file:
        logger.info(f"Forecast SPI-3 file: {fct_file}")
    logger.info("========================")
    
    print("\n=== SPI Processing Complete ===")
    print(f"Region: {args.region_id}")
    print(f"Mode: {args.mode}")
    print(f"Log file: {log_file}")
    if obs_file:
        print(f"Observation SPI-3 file: {obs_file}")
    if fct_file:
        print(f"Forecast SPI-3 file: {fct_file}")
    print("================================")
