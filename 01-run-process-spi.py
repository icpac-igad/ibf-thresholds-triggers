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

def get_region_bounds(region_id, credentials, buffer=0.5):
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

def process_seas51_data(region_id, obs_file, credentials, extent, seas51_file=None, output_dir='.'):
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
            
        # Check if the provided/calculated extent covers the observation dataset
        if (lat_min > obs_lat_min or lat_max < obs_lat_max or 
            lon_min > obs_lon_min or lon_max < obs_lon_max):
            logger.warning("Warning: The selected extent does not fully cover the observation dataset.")
    except Exception as e:
        logger.error(f"Error opening observation file: {e}")
        raise
    
    # Load SEAS51 data
    if seas51_file:
        logger.info(f"Using local SEAS51 file: {seas51_file}")
        if seas51_file.endswith('.grib') or seas51_file.endswith('.grb') or seas51_file.endswith('.grib2'):
            logger.info("Processing GRIB file format")
            sds = xr.open_dataset(seas51_file, engine='cfgrib', backend_kwargs=dict(time_dims=('forecastMonth', 'time')))
        else:
            sds = xr.open_dataset(seas51_file)
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
    output_file = os.path.join(output_dir, f'{region_id}_fct_spi3.nc')
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


if __name__ == "__main__":
    # Process a single region
    region_id ='kmj' 
    credentials = get_credentials('coiled-data.json')
    gdf, extent=get_region_bounds(region_id, credentials, buffer=0.5) 
    # Local file paths if available (set to None to use GCP data)
    chirps_file = 'chirps-v3.0.monthly.nc'  # 'path/to/local/chirps.nc'
    seas51_file = '3c58a474556eba4e1fd6a0d24e9824e8.grib'   # 'path/to/local/seas51.grib'
    # Process observation data
    obs_file = process_chirps_data(
        region_id, 
        credentials, 
        extent, 
        chirps_file=chirps_file
    )
    
    # Process forecast data
    fct_file = process_seas51_data(
        region_id, 
        obs_file, 
        credentials, 
        extent,
        seas51_file=seas51_file
    )

    # Process specific region
    #region_id = 'kmj'
    #obs_file = process_chirps_data(region_id, credentials)
    #fct_file = process_seas51_data(region_id, obs_file, credentials)
    
    logger.info(f"Completed processing for region {region_id}")
    logger.info(f"Observation file: {obs_file}")
    logger.info(f"Forecast file: {fct_file}")
    
    # Process all admin1 and custom regions
    # results = process_all_regions('coiled-data.json', 'output_data', ['admin1', 'custom'])
    # Apply shapefile masking to both outputs
    shapefile_df = gdf  # set your shapefile path
    
    # Define output paths for masked files
    masked_fct_path = f'{region_id}_fct_spi3_masked.nc'
    masked_obs_path = f'{region_id}_obs_spi3_masked.nc'
    
    # Run the masking function
    masked_fct, masked_obs = mask_netcdf_with_shapefile(
        forecast_path=fct_file, 
        obs_path=obs_file, 
        shapefile_df=shapefile_df,
        buffer_size=0.25,
        output_forecast_path=masked_fct_path,
        output_obs_path=masked_obs_path
    )
    
    logger.info(f"Masked datasets have been saved:")
    logger.info(f"Masked forecast: {masked_fct}")
    logger.info(f"Masked observations: {masked_obs}")
