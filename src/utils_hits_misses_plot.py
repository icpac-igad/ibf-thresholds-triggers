from io import StringIO
import os 
from dotenv import load_dotenv

import climpred
import xarray as xr
import xesmf as xe
import numpy as np
import pandas as pd
import regionmask
import geopandas as gp
from climpred import HindcastEnsemble
from datetime import datetime

import xhistogram.xarray as xhist
from sklearn.metrics import roc_auc_score
import pandas as pd

import xskillscore as xs
from xbootstrap import block_bootstrap
from dask.distributed import Client

import altair as alt


load_dotenv()

data_path=os.getenv("data_path")


def ken_mask_creator():
    """
    Utility for generating region/district masks using regionmask library

    Returns
    -------
    the_mask : regionmask.Regions
        Mask regions object for spatial selection
    rl_dict : dict
        Dictionary mapping region numbers to names
    mds2 : GeoDataFrame
        GeoDataFrame containing region geometries and metadata
    """
    try:
        # Read shapefiles
        dis = gp.read_file(f'{data_path}Karamoja_boundary_dissolved.shp')
        reg = gp.read_file(f'{data_path}wajir_mbt_extent.shp')
        
        # Combine and prepare regions
        mds = pd.concat([dis, reg])
        mds1 = mds.reset_index()
        mds1['region'] = [0, 1, 2]
        mds1['region_name'] = ['Karamoja', 'Marsabit', 'Wajir']
        mds2 = mds1[['geometry', 'region', 'region_name']]
        
        # Create regionmask object
        the_mask = regionmask.Regions(
            outlines=mds2.geometry.values,
            numbers=mds2.region.values,
            names=mds2.region_name.values,
            name='seas51_regions',
            overlap=False
        )
        
        # Create region name dictionary
        rl_dict = dict(zip(mds2.region, mds2.region_name))
        
        return the_mask, rl_dict, mds2
        
    except Exception as e:
        logger.error(f"Error in ken_mask_creator: {str(e)}")
        raise

def spi3_prod_name_creator(ds_ens,var_name):
    """
    Convenience function to generate a list of SPI product
    names, such as MAM, so that can be used to filter the 
    SPI product from dataframe

    added with method to convert the valid_time in CF format into datetime at
    line 3, which is the format given by climpred valid_time calculation 

    Parameters
    ----------
    ds_ens : xarray dataframe
        The data farme with SPI output organized for 
        the period 1981-2023.

    Returns
    -------
    spi_prod_list : String list
        List of names with iteration of SPI3 product names such as
        ['JFM','FMA','MAM',......]

    """
    db=pd.DataFrame()
    db['dt']=ds_ens[var_name].values
    db['dt1'] = db['dt'].apply(lambda x: datetime(x.year, x.month, x.day,
                                                                     x.hour, x.minute, x.second))
    #db['dt1']=db['dt'].to_datetimeindex()
    db['month']=db['dt1'].dt.strftime('%b').astype(str).str[0]
    db['year']=db['dt1'].dt.strftime('%Y')
    db['spi_prod'] = db.groupby('year')['month'].shift(2)+db.groupby('year')['month'].shift(1) + db.groupby('year')['month'].shift(0)
    spi_prod_list=db['spi_prod'].tolist()
    return spi_prod_list


def spi4_prod_name_creator(ds_ens,var_name):
    """
    Convenience function to generate a list of SPI product
    names, such as MAM, so that can be used to filter the 
    SPI product from dataframe

    added with method to convert the valid_time in CF format into datetime at
    line 3, which is the format given by climpred valid_time calculation 

    Parameters
    ----------
    ds_ens : xarray dataframe
        The data farme with SPI output organized for 
        the period 1981-2023.

    Returns
    -------
    spi_prod_list : String list
        List of names with iteration of SPI3 product names such as
        ['JFM','FMA','MAM',......]

    """
    db=pd.DataFrame()
    db['dt']=ds_ens[var_name].values
    db['dt1'] = db['dt'].apply(lambda x: datetime(x.year, x.month, x.day,
                                                                     x.hour, x.minute, x.second))
    #db['dt1']=db['dt'].to_datetimeindex()
    db['month']=db['dt1'].dt.strftime('%b').astype(str).str[0]
    db['year']=db['dt1'].dt.strftime('%Y')
    db['spi_prod'] = db.groupby('year')['month'].shift(3)+db.groupby('year')['month'].shift(2)+db.groupby('year')['month'].shift(1) + db.groupby('year')['month'].shift(0)
    spi_prod_list=db['spi_prod'].tolist()
    return spi_prod_list


def make_obs_fct_dataset(region_id,season_str,lead_int):
    """
    Prepares observed and forecasted dataset subsets for a specific region, season, and lead time.

    This function loads observed and forecasted datasets based on the season string length (indicating SPI3 or SPI4),
    applies regional masking, selects the data for the given region by its ID, and subsets the data for the specified
    season and lead time. It then aligns the observed dataset time coordinates with the forecasted dataset valid time
    coordinates and returns both datasets.

    Parameters:
    - region_id (int): The identifier for the region of interest.
    - season_str (str): A string representing the season. The length of this string determines whether SPI3 or SPI4
                        datasets are used ('mam', 'jjas', etc. for SPI3, and longer strings for SPI4).
    - lead_int (int): The lead time index for which the forecast dataset is to be subset.

    Returns:
    - obs_data (xarray.DataArray): The subsetted observed data array for the specified region, season, and aligned time coordinates.
    - ens_data (xarray.DataArray): The subsetted forecast data array for the specified region, season, lead time, and aligned time coordinates.

    Notes:
    - The function assumes the existence of a `data_path` variable that specifies the base path to the dataset files.
    - It requires the `xarray` library for data manipulation and assumes specific naming conventions for the dataset files.
    - Regional masking and season-specific processing rely on externally defined functions and naming conventions.
    - The final alignment of observed dataset time coordinates with forecasted dataset valid time coordinates ensures
      comparability between observed and forecasted values for verification purposes.

    Example Usage:
    >>> obs_data, ens_data = make_obs_fct_dataset(1, 'mam', 0)
    >>> print(obs_data)
    >>> print(ens_data)

    This would load the observed and forecasted SPI3 datasets for region 1 during the 'mam' season and subset them
    for lead time index 0, aligning the observed data time coordinates with the forecasted data valid time coordinates.
    """
    if len(season_str) == 3:
        kn_fct=xr.open_dataset(f'{data_path}clip_kn_fct_spi3.nc')
        kn_obs=xr.open_dataset(f'{data_path}clip_kn_obs_spi3.nc')
    else:
        kn_fct=xr.open_dataset(f'{data_path}clip_kn_fct_spi4.nc')
        kn_obs=xr.open_dataset(f'{data_path}clip_kn_obs_spi4.nc')
    the_mask, rl_dict,mds1=ken_mask_creator()
    bounds = mds1.bounds
    #bounds.iloc[0].minx
    llon=bounds.iloc[region_id].minx
    llat=bounds.iloc[region_id].miny
    ulon=bounds.iloc[region_id].maxx
    ulat=bounds.iloc[region_id].maxy
    a_fc=kn_fct.sel(lon=slice(llon, ulon), lat=slice(llat,ulat))
    a_obs=kn_obs.sel(lon=slice(llon, ulon), lat=slice(llat,ulat))
    hindcast = HindcastEnsemble(a_fc)
    hindcast = hindcast.add_observations(a_obs)
    #hindcast
    #spi_cdb1spi3_prod_name_creator(ds_ens)
    a_fc1=hindcast.get_initialized()
    a_fc2=a_fc1.isel(lead=lead_int)
    if len(season_str) == 3:
        spi_prod_list=spi3_prod_name_creator(a_fc2,'valid_time')
        obs_spi_prod_list=spi3_prod_name_creator(a_obs,'time')
    else:
        spi_prod_list=spi4_prod_name_creator(a_fc2,'valid_time')
        obs_spi_prod_list=spi4_prod_name_creator(a_obs,'time')
    a_fc2 = a_fc2.assign_coords(spi_prod=('init',spi_prod_list))
    a_fc3=a_fc2.where(a_fc2.spi_prod==season_str, drop=True)
    #obsertations
    a_obs1 = a_obs.assign_coords(spi_prod=('time',obs_spi_prod_list))
    a_obs2=a_obs1.where(a_obs1.spi_prod==season_str, drop=True)
    #valid_time_series = a_fc3.valid_time.to_series().reset_index(drop=True).drop_duplicates()
    valid_time_flattened = a_fc2.valid_time.to_dataframe().reset_index().drop_duplicates(subset='valid_time')['valid_time']
    valid_time_flattened.columns=['valid_time','cc']
    #valid_time_flattened['valid_time'] = pd.to_datetime(valid_time_flattened['valid_time'])
    # Apply lambda function to create 'dt1' column
    #valid_time_flattened['dt1'] = valid_time_flattened['valid_time'].apply(
    #    lambda x: datetime(x.year, x.month, x.day, x.hour, x.minute, x.second)
    #)
    #
    valid_time_flattened['dt1'] =valid_time_flattened['valid_time'].apply(lambda x: datetime(x.year, x.month, x.day,x.hour, x.minute, x.second))
    # Ensure the valid_time is in 'YYYY-MM-DD' string format
    #valid_time_flattened['dt2'] = valid_time_flattened['dt1'].dt.strftime('%Y-%m-%d')
    valid_time_flattened['dt1'] = valid_time_flattened['dt1'].dt.strftime('%Y-%m-%dT%H:%M:%S.%f')
    valid_time_flattened['dt1'] = pd.to_datetime(valid_time_flattened['dt1'])
    # Convert to xarray DataArray with time as the dimension name
    #valid_time_da = xr.DataArray(valid_time_flattened['dt1'], dims=['time'])
    valid_time_da = xr.DataArray(valid_time_flattened['dt1'], dims=['time'],coords=valid_time_flattened['dt1'])
    a_obs3 = a_obs2.reindex(time=valid_time_da)
    #a_obs4 = a_obs3.reindex(time=a_obs2.time)
    #a_obs4 = a_obs3.sel(time=a_obs2.time, drop=True)
    a_obs3 = a_obs3.dropna(dim='time')
    if len(season_str) == 3:
        obs_data=a_obs3['spi3']
        ens_data=a_fc3['spi3']
    else:
        obs_data=a_obs3['spi4']
        ens_data=a_fc3['spi4']
    return obs_data, ens_data, a_fc, a_obs


def make_obs_fct_dataset(region_id,season_str,lead_int):
    """
    Prepares observed and forecasted dataset subsets for a specific region, season, and lead time.

    This function loads observed and forecasted datasets based on the season string length (indicating SPI3 or SPI4),
    applies regional masking, selects the data for the given region by its ID, and subsets the data for the specified
    season and lead time. It then aligns the observed dataset time coordinates with the forecasted dataset valid time
    coordinates and returns both datasets.

    Parameters:
    - region_id (int): The identifier for the region of interest.
    - season_str (str): A string representing the season. The length of this string determines whether SPI3 or SPI4
                        datasets are used ('mam', 'jjas', etc. for SPI3, and longer strings for SPI4).
    - lead_int (int): The lead time index for which the forecast dataset is to be subset.

    Returns:
    - obs_data (xarray.DataArray): The subsetted observed data array for the specified region, season, and aligned time coordinates.
    - ens_data (xarray.DataArray): The subsetted forecast data array for the specified region, season, lead time, and aligned time coordinates.

    Notes:
    - The function assumes the existence of a `data_path` variable that specifies the base path to the dataset files.
    - It requires the `xarray` library for data manipulation and assumes specific naming conventions for the dataset files.
    - Regional masking and season-specific processing rely on externally defined functions and naming conventions.
    - The final alignment of observed dataset time coordinates with forecasted dataset valid time coordinates ensures
      comparability between observed and forecasted values for verification purposes.

    Example Usage:
    >>> obs_data, ens_data = make_obs_fct_dataset(1, 'mam', 0)
    >>> print(obs_data)
    >>> print(ens_data)

    This would load the observed and forecasted SPI3 datasets for region 1 during the 'mam' season and subset them
    for lead time index 0, aligning the observed data time coordinates with the forecasted data valid time coordinates.
    """
    if len(season_str) == 3:
        kn_fct=xr.open_dataset(f'{data_path}kn_fct_spi3.nc')
        kn_obs=xr.open_dataset(f'{data_path}kn_obs_spi3.nc')
    else:
        kn_fct=xr.open_dataset(f'{data_path}kn_fct_spi4.nc')
        kn_obs=xr.open_dataset(f'{data_path}kn_obs_spi4.nc')
    the_mask, rl_dict,mds1=ken_mask_creator()
    bounds = mds1.bounds
    #bounds.iloc[0].minx
    llon=bounds.iloc[region_id].minx
    llat=bounds.iloc[region_id].miny
    ulon=bounds.iloc[region_id].maxx
    ulat=bounds.iloc[region_id].maxy
    a_fc=kn_fct.sel(lon=slice(llon, ulon), lat=slice(llat,ulat))
    a_obs=kn_obs.sel(lon=slice(llon, ulon), lat=slice(llat,ulat))
    hindcast = HindcastEnsemble(a_fc)
    hindcast = hindcast.add_observations(a_obs)
    #hindcast
    #spi_cdb1spi3_prod_name_creator(ds_ens)
    a_fc1=hindcast.get_initialized()
    a_fc2=a_fc1.isel(lead=lead_int)
    if len(season_str) == 3:
        spi_prod_list=spi3_prod_name_creator(a_fc2,'valid_time')
        obs_spi_prod_list=spi3_prod_name_creator(a_obs,'time')
    else:
        spi_prod_list=spi4_prod_name_creator(a_fc2,'valid_time')
        obs_spi_prod_list=spi4_prod_name_creator(a_obs,'time')
    a_fc2 = a_fc2.assign_coords(spi_prod=('init',spi_prod_list))
    a_fc3=a_fc2.where(a_fc2.spi_prod==season_str, drop=True)
    #obsertations
    a_obs1 = a_obs.assign_coords(spi_prod=('time',obs_spi_prod_list))
    a_obs2=a_obs1.where(a_obs1.spi_prod==season_str, drop=True)
    #valid_time_series = a_fc3.valid_time.to_series().reset_index(drop=True).drop_duplicates()
    valid_time_flattened = a_fc2.valid_time.to_dataframe().reset_index().drop_duplicates(subset='valid_time')['valid_time']
    valid_time_flattened.columns=['valid_time','cc']
    #valid_time_flattened['valid_time'] = pd.to_datetime(valid_time_flattened['valid_time'])
    # Apply lambda function to create 'dt1' column
    #valid_time_flattened['dt1'] = valid_time_flattened['valid_time'].apply(
    #    lambda x: datetime(x.year, x.month, x.day, x.hour, x.minute, x.second)
    #)
    #
    valid_time_flattened['dt1'] =valid_time_flattened['valid_time'].apply(lambda x: datetime(x.year, x.month, x.day,x.hour, x.minute, x.second))
    # Ensure the valid_time is in 'YYYY-MM-DD' string format
    #valid_time_flattened['dt2'] = valid_time_flattened['dt1'].dt.strftime('%Y-%m-%d')
    valid_time_flattened['dt1'] = valid_time_flattened['dt1'].dt.strftime('%Y-%m-%dT%H:%M:%S.%f')
    valid_time_flattened['dt1'] = pd.to_datetime(valid_time_flattened['dt1'])
    # Convert to xarray DataArray with time as the dimension name
    #valid_time_da = xr.DataArray(valid_time_flattened['dt1'], dims=['time'])
    valid_time_da = xr.DataArray(valid_time_flattened['dt1'], dims=['time'],coords=valid_time_flattened['dt1'])
    a_obs3 = a_obs2.reindex(time=valid_time_da)
    #a_obs4 = a_obs3.reindex(time=a_obs2.time)
    #a_obs4 = a_obs3.sel(time=a_obs2.time, drop=True)
    a_obs3 = a_obs3.dropna(dim='time')
    if len(season_str) == 3:
        obs_data=a_obs3['spi3']
        ens_data=a_fc3['spi3']
    else:
        obs_data=a_obs3['spi4']
        ens_data=a_fc3['spi4']
    return obs_data, ens_data, a_fc, a_obs



def get_threshold(region_id, season):
    """
    Retrieves the drought threshold value for a specified region, season, and drought level.

    The function reads predefined threshold values from a CSV-format string. It looks up the threshold for the given
    region ID, season, and drought level ('mod' for moderate, 'sev' for severe, or 'ext' for extreme). These thresholds
    are specific to certain regions and seasons and indicate the level at which a drought event of a particular severity
    is considered to occur.

    Parameters:
    - region_id (int): The integer identifier for the region of interest.
    - season (str): The season for which the threshold is required. Expected values are season codes such as 'mam' (March-April-May),
                    'jjas' (June-July-August-September), 'ond' (October-November-December), etc.
    - level (str): The drought severity level for which the threshold is requested. Valid options are 'mod' for moderate,
                   'sev' for severe, and 'ext' for extreme drought conditions.

    Returns:
    - float: The threshold value for the specified region, season, and drought level. Returns None if no threshold is found for the given inputs.

    Note:
    - This function uses a hardcoded CSV string as its data source. In a production environment, it's recommended to
      store and retrieve such data from a more robust data management system.
    - The function requires the pandas library for data manipulation and the StringIO module from io for string-based data input.

    Example usage:
    >>> threshold = get_threshold(1, 'mam', 'mod')
    >>> print(threshold)
    -0.14
    """
    data = """region_id,region,season,mod,sev,ext
    0,kmj,mam,-0.03,-0.56,-0.99
    0,kmj,jjas,-0.01,-0.41,-0.99
    1,mbt,mam,-0.14,-0.38,-0.8
    1,mbt,ond,-0.15,-0.53,-0.71
    2,wjr,mam,-0.19,-0.45,-0.75
    2,wjr,ond,-0.29,-0.76,-0.9
    """
    # Use StringIO to convert the string data to a file-like object
    data_io = StringIO(data)
    # Read the data into a pandas DataFrame
    df = pd.read_csv(data_io)
    thresholds_dict = { (row['region_id'], row['season']): {'mod': row['mod'], 'sev': row['sev'], 'ext': row['ext']}
                   for _, row in df.iterrows() }
    # Retrieve the dictionary for the given region_id and season
    season_thresholds = thresholds_dict.get((region_id, season), {})
    # Return the threshold for the given level (mod, sev, ext), or None if not found
    return season_thresholds


def get_triggers_bin_edges():
    """
    Generate bin edges for triggers based on forecast category edges.

    Returns:
    list of lists: Bin edges arranged with three elements each.
    """
    forecast_category_edges = np.linspace(0, 1, 101)
    # Initialize an empty list to hold your list of lists
    list_of_lists = []
    # Iterate through forecast_category_edges to construct each [n1, n2, n3]
    for i, edge in enumerate(forecast_category_edges):
        if i == 0:
            # For the first element, there is no lower edge within the range, so you might set n1 to 0 or any other logic
            n1 = 0  # or edge itself if you want to keep it within valid probability bounds
        else:
            n1 = forecast_category_edges[i-1]
        n2 = edge  # The current edge value
        if i == len(forecast_category_edges) - 1:
            # For the last element, there is no upper edge within the range, so you might set n3 to 1 or any other logic
            n3 = 1  # or edge itself if you want to keep it within valid probability bounds
        else:
            n3 = forecast_category_edges[i+1]
        # Append the [n1, n2, n3] list to your list of lists
        list_of_lists.append([n1, n2, n3])
    return list_of_lists

def get_thresholds_bin_edges(threshold_dict, lowest_bound=-4.0, highest_bound=4.0):
    """
    Generate bin edges based on provided thresholds, ensuring all sublists have three elements:
    [lower_edge, threshold, upper_edge], including the extreme bounds.
    
    Parameters:
    - threshold_dict (dict): Dictionary with levels as keys and thresholds as values.
    - lowest_bound (float): Lowest boundary for the bins.
    - highest_bound (float): Highest boundary for the bins.
    
    Returns:
    - list of lists: Bin edges arranged with three elements each.
    
    TODO
    merge the dict call on level and then return the sepcific bin edges for that level
    """
    # Extract thresholds and sort them in ascending order
    sorted_thresholds = sorted(threshold_dict.values())
    
    # Initialize list of lists with the first bin
    list_of_lists = []
    
    # Handle the first bin separately
    if sorted_thresholds:
        list_of_lists.append([lowest_bound, sorted_thresholds[0], sorted_thresholds[1] if len(sorted_thresholds) > 1 else highest_bound])
    
    # Loop through the sorted thresholds to create bins for the middle thresholds
    for i in range(1, len(sorted_thresholds) - 1):
        list_of_lists.append([sorted_thresholds[i-1], sorted_thresholds[i], sorted_thresholds[i+1]])
    
    # Handle the last bin separately if there are at least two thresholds
    if len(sorted_thresholds) > 1:
        list_of_lists.append([sorted_thresholds[-2], sorted_thresholds[-1], highest_bound])
    
    # Special case: If there is only one threshold, adjust the initial list to include highest_bound
    if len(sorted_thresholds) == 1:
        list_of_lists[0][-1] = highest_bound  # Replace the last element of the first sublist with highest_bound
    
    return list_of_lists


def del_emprical_probablity(ens_data,threshold_dict):
    mod_thr=threshold_dict['mod']
    fct_mod= (ens_data <= mod_thr).mean(dim='member')
    fct_mod_mean = fct_mod.mean(dim=['lat', 'lon'])
    fct_mod_min = fct_mod.min(dim=['lat', 'lon'])
    fct_mod_max = fct_mod.max(dim=['lat', 'lon'])
    ####
    sev_thr=threshold_dict['sev']
    fct_sev= (ens_data <= sev_thr).mean(dim='member')
    fct_sev_mean = fct_sev.mean(dim=['lat', 'lon'])
    fct_sev_min = fct_sev.min(dim=['lat', 'lon'])
    fct_sev_max = fct_sev.max(dim=['lat', 'lon'])
    ext_thr=threshold_dict['ext']
    fct_ext= (ens_data <= ext_thr).mean(dim='member')
    fct_ext_mean = fct_ext.mean(dim=['lat', 'lon'])
    fct_ext_min = fct_ext.min(dim=['lat', 'lon'])
    fct_ext_max = fct_ext.max(dim=['lat', 'lon'])
    
    
def emprical_probablity(ens_data,threshold_dict):
    mod_thr=threshold_dict['mod']
    fct_mod= (ens_data <= mod_thr).mean(dim='member')
    ####
    sev_thr=threshold_dict['sev']
    fct_sev= (ens_data <= sev_thr).mean(dim='member')
    ####
    ext_thr=threshold_dict['ext']
    fct_ext= (ens_data <= ext_thr).mean(dim='member')
    return fct_mod, fct_sev, fct_ext


def mean_emp_prob(fct_mod, fct_sev, fct_ext):
    fct_mod_mean = fct_mod.mean(dim=['lat', 'lon'])
    fct_mod_df=fct_mod_mean.to_dataframe().reset_index()
    fct_mod_df1=fct_mod_df[['valid_time','spi3']]
    fct_mod_df1 = fct_mod_df1.assign(cat='mod')
    fct_sev_mean = fct_sev.mean(dim=['lat', 'lon'])
    fct_sev_df=fct_sev_mean.to_dataframe().reset_index()
    fct_sev_df1=fct_sev_df[['valid_time','spi3']]
    fct_sev_df1 = fct_sev_df1.assign(cat='sev')
    fct_ext_mean = fct_ext.mean(dim=['lat', 'lon'])
    fct_ext_df=fct_ext_mean.to_dataframe().reset_index()
    fct_ext_df1=fct_ext_df[['valid_time','spi3']]
    fct_ext_df1 = fct_ext_df1.assign(cat='ext')
    wdf=pd.concat([fct_mod_df1,fct_sev_df1,fct_ext_df1])
    wdf['year0']=wdf['valid_time'].apply(lambda x: datetime(x.year, x.month, x.day,x.hour, x.minute, x.second))
    wdf['year']=wdf['year0'].dt.strftime('%Y')
    wdf1=wdf[['spi3','cat','year']]
    return wdf1


def min_emp_prob(fct_mod,fct_sev,fct_ext):
    fct_mod_mean = fct_mod.min(dim=['lat', 'lon'])
    fct_mod_df=fct_mod_mean.to_dataframe().reset_index()
    fct_mod_df1=fct_mod_df[['valid_time','spi3']]
    fct_mod_df1 = fct_mod_df1.assign(cat='mod')
    fct_sev_mean = fct_sev.min(dim=['lat', 'lon'])
    fct_sev_df=fct_sev_mean.to_dataframe().reset_index()
    fct_sev_df1=fct_sev_df[['valid_time','spi3']]
    fct_sev_df1 = fct_sev_df1.assign(cat='sev')
    fct_ext_mean = fct_ext.min(dim=['lat', 'lon'])
    fct_ext_df=fct_ext_mean.to_dataframe().reset_index()
    fct_ext_df1=fct_ext_df[['valid_time','spi3']]
    fct_ext_df1 = fct_ext_df1.assign(cat='ext')
    wdf=pd.concat([fct_mod_df1,fct_sev_df1,fct_ext_df1])
    wdf['year0']=wdf['valid_time'].apply(lambda x: datetime(x.year, x.month, x.day,x.hour, x.minute, x.second))
    wdf['year']=wdf['year0'].dt.strftime('%Y')
    wdf1=wdf[['spi3','cat','year']]
    return wdf1


def max_emp_prob(fct_mod,fct_sev,fct_ext):
    fct_mod_max = fct_mod.max(dim=['lat', 'lon'])
    fct_mod_df=fct_mod_max.to_dataframe().reset_index()
    fct_mod_df1=fct_mod_df[['valid_time','spi3']]
    fct_mod_df1 = fct_mod_df1.assign(cat='mod')
    fct_sev_max = fct_sev.max(dim=['lat', 'lon'])
    fct_sev_df=fct_sev_max.to_dataframe().reset_index()
    fct_sev_df1=fct_sev_df[['valid_time','spi3']]
    fct_sev_df1 = fct_sev_df1.assign(cat='sev')
    fct_ext_max = fct_ext.max(dim=['lat', 'lon'])
    fct_ext_df=fct_ext_max.to_dataframe().reset_index()
    fct_ext_df1=fct_ext_df[['valid_time','spi3']]
    fct_ext_df1 = fct_ext_df1.assign(cat='ext')
    wdf=pd.concat([fct_mod_df1,fct_sev_df1,fct_ext_df1])
    wdf['year0']=wdf['valid_time'].apply(lambda x: datetime(x.year, x.month, x.day,x.hour, x.minute, x.second))
    wdf['year']=wdf['year0'].dt.strftime('%Y')
    wdf1=wdf[['spi3','cat','year']]
    return wdf1


def create_new_column(df):
    new_column = []
    
    for _, row in df.iterrows():
        lt = row['lt']
        cat = row['cat']
        season = row['season']
        
        if season == 'MAM':
            if lt == 1:
                if cat == 'mod':
                    new_column.append('mar_x')
                elif cat == 'sev':
                    new_column.append('mar_y')
                elif cat == 'ext':
                    new_column.append('mar_z')
            elif lt == 2:
                if cat == 'mod':
                    new_column.append('feb_x')
                elif cat == 'sev':
                    new_column.append('feb_y')
                elif cat == 'ext':
                    new_column.append('feb_z')
            elif lt == 3:
                if cat == 'mod':
                    new_column.append('jan_x')
                elif cat == 'sev':
                    new_column.append('jan_y')
                elif cat == 'ext':
                    new_column.append('jan_z')
            elif lt == 4:
                if cat == 'mod':
                    new_column.append('dec_x')
                elif cat == 'sev':
                    new_column.append('dec_y')
                elif cat == 'ext':
                    new_column.append('dec_z')
            elif lt == 5:
                if cat == 'mod':
                    new_column.append('nov_x')
                elif cat == 'sev':
                    new_column.append('nov_y')
                elif cat == 'ext':
                    new_column.append('nov_z')
        elif season == 'OND':
            if lt == 1:
                if cat == 'mod':
                    new_column.append('oct_x')
                elif cat == 'sev':
                    new_column.append('oct_y')
                elif cat == 'ext':
                    new_column.append('oct_z')
            elif lt == 2:
                if cat == 'mod':
                    new_column.append('sep_x')
                elif cat == 'sev':
                    new_column.append('sep_y')
                elif cat == 'ext':
                    new_column.append('sep_z')
            elif lt == 3:
                if cat == 'mod':
                    new_column.append('aug_x')
                elif cat == 'sev':
                    new_column.append('aug_y')
                elif cat == 'ext':
                    new_column.append('aug_z')
            elif lt == 4:
                if cat == 'mod':
                    new_column.append('jul_x')
                elif cat == 'sev':
                    new_column.append('jul_y')
                elif cat == 'ext':
                    new_column.append('jul_z')
            elif lt == 5:
                if cat == 'mod':
                    new_column.append('jun_x')
                elif cat == 'sev':
                    new_column.append('jun_y')
                elif cat == 'ext':
                    new_column.append('jun_z')
        elif season == 'JJAS':
            if lt == 2:
                if cat == 'mod':
                    new_column.append('jun_x')
                elif cat == 'sev':
                    new_column.append('jun_y')
                elif cat == 'ext':
                    new_column.append('jun_z')
            elif lt == 3:
                if cat == 'mod':
                    new_column.append('may_x')
                elif cat == 'sev':
                    new_column.append('may_y')
                elif cat == 'ext':
                    new_column.append('may_z')
            elif lt == 4:
                if cat == 'mod':
                    new_column.append('apr_x')
                elif cat == 'sev':
                    new_column.append('apr_y')
                elif cat == 'ext':
                    new_column.append('apr_z')
            elif lt == 5:
                if cat == 'mod':
                    new_column.append('mar_x')
                elif cat == 'sev':
                    new_column.append('mar_y')
                elif cat == 'ext':
                    new_column.append('mar_z')
        else:
            new_column.append('')
    
    #df['new_column'] = new_column
    df = df.assign(new_column=new_column)
    return df


def get_subset(df):
    # Filter out rows with null values in 'hit_rate' and 'false_alarm_ratio'
    #df = df.dropna(subset=['hit_rate', 'false_alarm_ratio'])
    
    # Sort the DataFrame by 'peirce_score' in descending order
    df = df.sort_values(by='peirce_score', ascending=False)
    
    # Get the row with the maximum 'peirce_score'
    max_peirce_row = df.iloc[0]
    
    # Sort the DataFrame by 'bias_score' in descending order, and filter for 'bias_score' < 1.0
    df = df.loc[df['bias_score'] < 1.0].sort_values(by='bias_score', ascending=False)
    
    # Get the row with the maximum 'bias_score' < 1.0
    max_bias_row = df.iloc[0]
    
    # Sort the DataFrame by 'heidke_score' in descending order
    df = df.sort_values(by='heidke_score', ascending=False)
    
    # Get the row with the maximum 'heidke_score'
    max_heidke_row = df.iloc[0]
    
    # Combine the three rows into a subset
    subset = pd.concat([pd.DataFrame([max_peirce_row]), pd.DataFrame([max_bias_row]), pd.DataFrame([max_heidke_row])], ignore_index=True)
    
    return subset


def choose_row(df):
    # Filter out rows where false_alarm_ratio or hit_rate is 1.0 or 0.0
    #filtered_df = df[(df['false_alarm_ratio'] != 1.0) & (df['false_alarm_ratio'] != 0.0) &
    #                 (df['hit_rate'] != 1.0) & (df['hit_rate'] != 0.0)]
    
    # Filter out rows where percentage_spi is less than or equal to 10
    #filtered_df = df[df['percentage_spi'] > 10]
    
    # If there are no rows left after filtering, return None
    #if filtered_df.empty:
    #    return None
    
    # Sort the filtered DataFrame by percentage_spi in descending order
    filtered_df = df.sort_values(by='percentage_spi', ascending=False)
    
    # Return the first row of the sorted DataFrame
    #chosen_row = filtered_df.iloc[0]
    chosen_row = pd.DataFrame([filtered_df.iloc[0]])
    
    return chosen_row

def replace_with_list(x):
    """
    Replaces NaN float values with a predefined list of replacement values.

    Parameters:
    - x (float): The input value to be checked and potentially replaced.

    Returns:
    - A list of replacement values if `x` is a float and is NaN. Otherwise, returns `x` unchanged.

    Note:
    - This function is designed to handle cases where cell values in a dataset need to be replaced with a list of values
      for indicating missing or special cases.
    """
    replacement_values = [-999.0, -999.0]
    # If x is a float and it is nan (meaning the cell was originally empty), return the replacement list
    if isinstance(x, float) and np.isnan(x):
        return replacement_values
    # Otherwise, return x as it is
    return x


def round_list(lst, decimal_places):
    """
    Rounds each element in a list to a specified number of decimal places.

    Parameters:
    - lst (list of float): The list of numbers to be rounded.
    - decimal_places (int): The number of decimal places to round each number to.

    Returns:
    - A list containing the rounded values of the input list.

    Note:
    - This function is useful for rounding numerical values in a list to ensure consistency or to improve readability.
    """
    return [round(x, decimal_places) for x in lst]


def decide_for_region_season(df,region_id,season):
    df_no0 = df[df["lt"] != 0]
    mask = (df_no0["lt"] == 1) & (df_no0["season"] == "JJAS")
    df_no1 = df_no0[~mask]
    df_no2=df_no1[df_no1['subset']=='mean']
    #db2.info()
    df_no3=df_no2[df_no2['region_id']==region_id]
    df_no4=df_no3[df_no3['season']==season]
    df3=create_new_column(df_no4)
    df3 = df3.assign(identify=df3['region_id'].astype(str) + '-' + df3['season'] + '-' + df3['new_column']+ '-' + df3['cat'])
    _=df3.drop_duplicates('identify')
    identify_list=_['identify'].tolist()
    d_odb=[]
    for idl in identify_list:
        odb=df3[df3['identify']==idl]
        odb1=get_subset(odb)
        odb2=choose_row(odb1)
        d_odb.append(odb2)
    ddf=pd.concat(d_odb)
    #ddf1=ddf[ddf['region_id']==region_id]
    #ddf2=ddf1[ddf1['season']==season]
    return ddf


def get_ep_for_region_season(df,region_id,season):
    df_no0 = df[df["lt"] != 0]
    mask = (df_no0["lt"] == 1) & (df_no0["season"] == "JJAS")
    df_no1 = df_no0[~mask]
    df_no2=df_no1[df_no1['subset']=='mean']
    df_no3=df_no2[df_no2['region_id']==region_id]
    df_no4=df_no3[df_no3['season']==season]
    df_no5=create_new_column(df_no4)
    df_no5 = df_no5.assign(identify=df_no5['region_id'].astype(str) + '-' + df_no5['season'] + '-' + df_no5['new_column']+ '-' + df_no5['cat'])
    return df_no5


def mean_obs_spi(obs_data, spi_string_name):
    obs_data_mean = obs_data.mean(dim=["lat", "lon"])
    obs_data_df = obs_data_mean.to_dataframe().reset_index()
    obs_data_df1 = obs_data_df[["time", spi_string_name]]
    wdf=obs_data_df1
    wdf["year0"] = wdf["time"].apply(
        lambda x: datetime(x.year, x.month, x.day, x.hour, x.minute, x.second)
    )
    wdf["year"] = wdf["year0"].dt.strftime("%Y")
    wdf1 = wdf[[spi_string_name, "year"]]
    return wdf1

def decided_triggers(df, lt_value):
    df1 = df[df['lt'] == lt_value]
    if not df1.empty:
        ext_val = df1[df1['cat'] == 'ext']['decided_trigger'].iloc[0] if not df1[df1['cat'] == 'ext'].empty else 0
        sev_val = df1[df1['cat'] == 'sev']['decided_trigger'].iloc[0] if not df1[df1['cat'] == 'sev'].empty else 0
        mod_val = df1[df1['cat'] == 'mod']['decided_trigger'].iloc[0] if not df1[df1['cat'] == 'mod'].empty else 0
        decided_tr_dict = {'ext': ext_val, 'sev': sev_val, 'mod': mod_val }
        return decided_tr_dict
    


def obs_chart_with_triggers(plot_type,df, year_column, spi_column, threshold_dict):
    """
    Create an Altair chart with a bar chart overlaid by trigger lines.
    
    Parameters:
    df : pandas.DataFrame
        The DataFrame containing the data.
    year_column : str
        The name of the DataFrame column containing the year.
    spi_column : str
        The name of the DataFrame column containing SPI values.
    threshold_dict : dict
        A dictionary with keys as threshold names and values as threshold values.
    """
    
    # Bar chart
    if plot_type=='obs':
        bar_chart = alt.Chart(df).mark_bar().encode(
            x=alt.X(f'{year_column}:N', axis=alt.Axis(labelAngle=90)),
            y=alt.Y(f'{spi_column}:Q', title=spi_column, scale=alt.Scale(domain=[-4, 4])),
            color=alt.condition(
                alt.datum[spi_column] > 0,
                alt.value('orange'),  # Color for positive values
                alt.value('blue')     # Color for negative values
            )
        ).properties(
            width=400,
            height=200
        )
    else:
        color_scale = alt.Scale(
        # domain=["ext", "sev", "mod"], range=["#880203", "#ffa400", "#fffe00"]
        domain=["mod", "sev", "ext"],
        range=["#fffe00", "#ffa400", "#880203"],)

        bar_chart = alt.Chart(df).mark_bar().encode(
            x=alt.X(f'{year_column}:N', axis=alt.Axis(labelAngle=90)),
            y=alt.Y(f'{spi_column}:Q', title='Probability (%)', stack=None),
            color=alt.Color('cat:N', scale=color_scale, sort=["sev", "mod", "ext"]),
        ).properties(
            width=400,
            height=200
        )

    
    # Adding trigger lines
    rules = []
    for key, value in threshold_dict.items():
        rule = alt.Chart(pd.DataFrame({'y': [value]})).mark_rule(
            strokeWidth=2,
            stroke= {'ext': '#880203', 'sev': '#ffa400', 'mod': '#fffe00'}[key]  # Conditional color assignment
        ).encode(
            y='y:Q'
        )
        rules.append(rule)
    
    # Combine the bar chart with trigger lines
    final_chart = alt.layer(bar_chart, *rules).configure_view(
        stroke=None
    ).configure_axis(
        grid=False
    ).configure_axisY(
        labelFontSize=12,
        titleFontSize=14
    ).configure_axisX(
        labelFontSize=10,
        titleFontSize=12
    ).configure_legend(
        labelFontSize=12,
        titleFontSize=14
    )
    
    return final_chart


def table(df):
    return (
        alt.Chart(df.reset_index())
        .mark_text()
        .transform_fold(df.columns.tolist())
        .encode(
            alt.X(
                "key",
                type="nominal",
                axis=alt.Axis(
                    # flip x labels upside down
                    orient="top",
                    # put x labels into horizontal direction
                    labelAngle=0,
                    title=None,
                    ticks=False
                ),
                scale=alt.Scale(padding=10),
                sort=None,
            ),
            alt.Y("index", type="ordinal", axis=None),
            alt.Text("value", type="nominal"),
        )
    )






def Aobs_chart_with_triggers(plot_type, df, year_column, spi_column, threshold_dict):
    """
    Create an Altair chart with a bar chart overlaid by trigger lines.

    Parameters:
    df : pandas.DataFrame
        The DataFrame containing the data.
    year_column : str
        The name of the DataFrame column containing the year.
    spi_column : str
        The name of the DataFrame column containing SPI values.
    threshold_dict : dict
        A dictionary with keys as threshold names and values as threshold values.
    """

    # Bar chart
    if plot_type == 'obs':
        bar_chart = alt.Chart(df).mark_bar().encode(
            x=alt.X(f'{year_column}:N', axis=alt.Axis(labelAngle=90)),
            y=alt.Y(f'{spi_column}:Q', title=spi_column, scale=alt.Scale(domain=[-4, 4])),
            color=alt.condition(
                alt.datum[spi_column] > 0,
                alt.value('orange'),  # Color for positive values
                alt.value('blue')  # Color for negative values
            )
        ).properties(
            width=400,
            height=200
        )
    else:
        color_scale = alt.Scale(
            # domain=["ext", "sev", "mod"], range=["#880203", "#ffa400", "#fffe00"]
            domain=["mod", "sev", "ext"],
            range=["#fffe00", "#ffa400", "#880203"],
        )

        bar_chart = alt.Chart(df).mark_bar().encode(
            x=alt.X(f'{year_column}:N', axis=alt.Axis(labelAngle=90)),
            y=alt.Y(f'{spi_column}:Q', title='Probability (%)', stack=None),
            color=alt.Color('cat:N', scale=color_scale, sort=["sev", "mod", "ext"]),
        ).properties(
            width=400,
            height=200
        )

    # Adding trigger lines
    rules = []
    for key, value in threshold_dict.items():
        rule = alt.Chart(pd.DataFrame({'y': [value]})).mark_rule(
            strokeWidth=2,
            stroke={'ext': '#880203', 'sev': '#ffa400', 'mod': '#fffe00'}[key]  # Conditional color assignment
        ).encode(
            y='y:Q'
        )
        rules.append(rule)

    # Combine the bar chart with trigger lines
    final_chart = alt.layer(bar_chart, *rules)

    # Configure the final chart
    final_chart = final_chart.configure_view(
        stroke=None
    ).configure_axis(
        grid=False
    ).configure_axisY(
        labelFontSize=12,
        titleFontSize=14
    ).configure_axisX(
        labelFontSize=10,
        titleFontSize=12
    ).configure_legend(
        labelFontSize=12,
        titleFontSize=14
    )

    return final_chart



def obs_chart_with_triggers(plot_type, df, year_column, spi_column, threshold_dict,row_annotations ):
    """
    Create an Altair chart with a bar chart overlaid by trigger lines.

    Parameters:
    df : pandas.DataFrame
        The DataFrame containing the data.
    year_column : str
        The name of the DataFrame column containing the year.
    spi_column : str
        The name of the DataFrame column containing SPI values.
    threshold_dict : dict
        A dictionary with keys as threshold names and values as threshold values.
    """

    # Bar chart
    if plot_type == 'obs':
        bar_chart = alt.Chart(df).mark_bar().encode(
            x=alt.X(f'{year_column}:N', axis=alt.Axis(labelAngle=90)),
            y=alt.Y(f'{spi_column}:Q', title=spi_column, scale=alt.Scale(domain=[-4, 4])),
            color=alt.condition(
                alt.datum[spi_column] > 0,
                alt.value('blue'),  # Color for positive values
                alt.value('red')  # Color for negative values
            )
        ).properties(
            width=400,
            height=200
        )
    else:
        color_scale = alt.Scale(
            # domain=["ext", "sev", "mod"], range=["#880203", "#ffa400", "#fffe00"]
            domain=["mod", "sev", "ext"],
            range=["#f4eb13", "#f89821", "#ed2227"],
        )

        bar_chart = alt.Chart(df).mark_bar().encode(
            x=alt.X(f'{year_column}:N', axis=alt.Axis(labelAngle=90)),
            y=alt.Y(f'{spi_column}:Q', title='Probability (%)', stack=None),
            color=alt.Color('cat:N', scale=color_scale, sort=["sev", "mod", "ext"]),
        ).properties(
            width=400,
            height=200
        )+row_annotations

    # Adding trigger lines
    rules = []
    for key, value in threshold_dict.items():
        rule = alt.Chart(pd.DataFrame({'y': [value]})).mark_rule(
            strokeWidth=2,
            stroke={'ext': '#ed2227', 'sev': '#f89821', 'mod': '#f4eb13'}[key]  # Conditional color assignment
        ).encode(
            y='y:Q'
        )
        rules.append(rule)

    # Combine the bar chart with trigger lines
    final_chart = alt.layer(bar_chart, *rules)

    # Configure the final chart
    # final_chart = final_chart.configure_view(
    #     stroke=None
    # ).configure_axis(
    #     grid=False
    # ).configure_axisY(
    #     labelFontSize=12,
    #     titleFontSize=14
    # ).configure_axisX(
    #     labelFontSize=10,
    #     titleFontSize=12
    # ).configure_legend(
    #     labelFontSize=12,
    #     titleFontSize=14
    # )

    return final_chart


def create_edges(fct_decided_tr_dict, obs_dict):
    """
    Creates category edge arrays for forecast and observation thresholds.

    Parameters:
    fct_decided_tr_dict : dict
        Dictionary containing the forecast thresholds with keys 'ext', 'sev', 'mod'.
    obs_dict : dict
        Dictionary containing the observation thresholds with keys 'ext', 'sev', 'mod'.

    Returns:
    dict
        A dictionary containing 'o_edges' and 'f_edges' as keys with the respective arrays as values.
    """
    o_edges = {}
    f_edges = {}
    
    # Creating the observation edges array
    for key, value in obs_dict.items():
        o_edges[f'o_edges_{key}'] = np.array([-np.inf, value, np.inf])
    
    # Creating the forecast edges array, rounding up to the nearest integer
    for key, value in fct_decided_tr_dict.items():
        f_edges[f'f_edges_{key}'] = np.array([-np.inf, np.ceil(value), np.inf])
    
    return {'o_edges': o_edges, 'f_edges': f_edges}


def process_data_for_region_and_season(mdb, region_id, season_str, lead_int, spi_string_name):
    """
    Processes data for a given region and season, merges various data sources, and extracts triggers.
    
    Parameters:
    mdb : DataFrame or similar
        Main database containing the data.
    region_id : int
        The ID of the region.
    season_str : str
        The season string (e.g., 'OND').
    lead_int : int
        The lead time integer.
    spi_string_name : str
        The name of the SPI variable (e.g., 'spi3').
    
    Returns:
    df1 : DataFrame
        Merged and filtered DataFrame based on the 'lt' value.
    decided_tr_dict : dict
        Dictionary containing decided triggers.
    """

    # Decide for region and season
    ddf = decide_for_region_season(mdb, region_id, season_str)
    ddf1 = ddf[['identify', 'percentage_spi']]
    ddf1.columns = ['identify', 'decided_trigger']

    # Get EP for region and season
    ep_df = get_ep_for_region_season(mdb, region_id, season_str)
    ep_df1 = ep_df[['identify', 'cat', 'lt', 'subset', 'year', 'percentage_spi']]

    # Merge dataframes
    df = pd.merge(ddf1, ep_df1, on='identify')
    fct_df = df[df['lt'] == lead_int]

    # Make observation and forecast dataset
    obs_data, ens_data, a_fc, a_obs = make_obs_fct_dataset(region_id, season_str, lead_int)
    sc_season_str = season_str.lower()

    # Get threshold
    obs_threshold_dict = get_threshold(region_id, sc_season_str)

    # Mean observation SPI
    obs_df = mean_obs_spi(obs_data, spi_string_name)
    
    # Decide triggers
    decided_fct_dict = decided_triggers(df, lead_int)

    return fct_df, obs_df, decided_fct_dict, obs_threshold_dict


def compute_contingency_table_metrics(merged_df, spi_column, percentage_spi_column, o_edges, f_edges):
    """
    Computes contingency table metrics for given observation and forecast data.

    Parameters:
    merged_df : pandas.DataFrame
        DataFrame containing the merged data with observation and forecast columns.
    spi_column : str
        The column name for SPI data in `merged_df`.
    percentage_spi_column : str
        The column name for percentage SPI data in `merged_df`.
    o_edges : array-like
        Category edges for making the observation data dichotomous.
    f_edges : array-like
        Category edges for making the forecast data dichotomous.

    Returns:
    pd.DataFrame
        DataFrame containing hits, misses, false alarms, and correct negatives.
    """

    # Convert Pandas DataFrame to xarray DataArray
    oda = xr.DataArray(
        merged_df[spi_column].values,
        coords={'time': merged_df['year']},
        dims=['time']
    )

    fda = xr.DataArray(
        merged_df[percentage_spi_column].values,
        coords={'time': merged_df['year']},
        dims=['time']
    )

    # Split the DataArray into 'observation' and 'forecast'
    observation = oda
    forecast = fda

    # Define the category edges to make the data dichotomous
    o_category_edges = np.array(o_edges)
    f_category_edges = np.array(f_edges)

    # Create the contingency table
    contingency_table = xs.Contingency(observation, forecast, o_category_edges, f_category_edges, dim=['time'])

    # Calculate the contingency table metrics
    hit_rate = contingency_table.hit_rate()
    hits = contingency_table.hits()
    misses = contingency_table.misses()
    false_alarms = contingency_table.false_alarms()
    correct_negatives = contingency_table.correct_negatives()

#     # Print the contingency table
#     print("Contingency Table:")
#     print(contingency_table)

#     # Print the contingency table metrics
#     print("hit rate:", hit_rate.values)
#     print("hits:", hits.values)
#     print("misses:", misses.values)
#     print("false alarms:", false_alarms.values)
#     print("correct negatives:", correct_negatives.values)

    # Prepare the results as a DataFrame
    data = {
        'Trigger': [merged_df['decided_trigger'].iloc[0]],
        'cat': [merged_df['cat'].iloc[0]],
        'lead_time':[merged_df['lt'].iloc[0]],
        'hits': [hits.values.item()],
        'misses': [misses.values.item()],
        'FA': [false_alarms.values.item()],
        'CN': [correct_negatives.values.item()]
    }

    results_df = pd.DataFrame(data)
    return results_df


def make_hit_misses(fct_df, obs_df, decided_fct_dict, obs_threshold_dict):
    spi_column='spi3'
    percentage_spi_column='percentage_spi'
    edges = create_edges(decided_fct_dict, obs_threshold_dict)
    obs_df['year'] = obs_df['year'].astype(int)
    
    mask = (fct_df['cat'] == 'mod')
    mod_fct_df=fct_df[mask]
    # Merge the two DataFrames on the 'year' column
    mod_merged_df = pd.merge(obs_df, mod_fct_df, on='year', how='inner')
    mod_o_edges=edges['o_edges']['o_edges_mod']
    mod_f_edges=edges['f_edges']['f_edges_mod']
    df_mod=compute_contingency_table_metrics(mod_merged_df, spi_column, percentage_spi_column, mod_o_edges, mod_f_edges)
    
    mask = (fct_df['cat'] == 'sev')
    sev_fct_df=fct_df[mask]
    # Merge the two DataFrames on the 'year' column
    sev_merged_df = pd.merge(obs_df, sev_fct_df, on='year', how='inner')
    sev_o_edges=edges['o_edges']['o_edges_sev']
    sev_f_edges=edges['f_edges']['f_edges_sev']
    df_sev=compute_contingency_table_metrics(sev_merged_df, spi_column, percentage_spi_column, sev_o_edges, sev_f_edges)
    
    mask = (fct_df['cat'] == 'ext')
    ext_fct_df=fct_df[mask]
    # Merge the two DataFrames on the 'year' column
    ext_merged_df = pd.merge(obs_df, ext_fct_df, on='year', how='inner')
    ext_o_edges=edges['o_edges']['o_edges_ext']
    ext_f_edges=edges['f_edges']['f_edges_ext']
    df_ext=compute_contingency_table_metrics(ext_merged_df, spi_column, percentage_spi_column, ext_o_edges, ext_f_edges)
    
    df=pd.concat([df_mod,df_sev,df_ext])
    return df


def make_barchart_annotations():
    row_annotations = [
    alt.Chart(pd.DataFrame({'text': ['lt=1']})).mark_text(
        align='left',
        baseline='middle',
        fontSize=14,
        fontWeight='bold',
        dx=-190,
        dy=-90
    ).encode(
        text='text:N'
    ).properties(width=400, height=200),
    alt.Chart(pd.DataFrame({'text': ['lt=2, Sep']})).mark_text(
        align='left',
        baseline='middle',
        fontSize=14,
        fontWeight='bold',
        dx=-190,
        dy=-90
    ).encode(
        text='text:N'
    ).properties(width=400, height=200),
    alt.Chart(pd.DataFrame({'text': ['lt=3, Aug']})).mark_text(
        align='left',
        baseline='middle',
        fontSize=14,
        fontWeight='bold',
        dx=-190,
        dy=-90
    ).encode(
        text='text:N'
    ).properties(width=400, height=200),
    alt.Chart(pd.DataFrame({'text': ['lt=4, Jul']})).mark_text(
        align='left',
        baseline='middle',
        fontSize=14,
        fontWeight='bold',
        dx=-190,
        dy=-90
    ).encode(
        text='text:N'
    ).properties(width=400, height=200),
    alt.Chart(pd.DataFrame({'text': ['lt=5']})).mark_text(
        align='left',
        baseline='middle',
        fontSize=14,
        fontWeight='bold',
        dx=-190,
        dy=-90
    ).encode(
        text='text:N'
    ).properties(width=400, height=200)]
    return row_annotations


# Usage:
mdb = pd.read_csv(f"{data_path}kimwa-metrix-v1.csv")
region_id = 1
season_str = 'OND'
lead_int = 2
spi_string_name = 'spi3'
row_annotations=make_barchart_annotations()

fct_df_lt2, obs_df, lt2_decided_fct_dict, obs_threshold_dict = process_data_for_region_and_season(mdb, region_id, season_str, lead_int, spi_string_name)

plot_type='obs'
obs_plot=obs_chart_with_triggers(plot_type,obs_df, 'year', 'spi3', obs_threshold_dict,row_annotations[2])

plot_type='fct'

lt2_plot=obs_chart_with_triggers(plot_type,fct_df_lt2, 'year', 'percentage_spi', lt2_decided_fct_dict,row_annotations[1])
df_lt2=make_hit_misses(fct_df_lt2, obs_df, lt2_decided_fct_dict, obs_threshold_dict)

lead_int = 3
fct_df_lt3, obs_df, lt3_decided_fct_dict, obs_threshold_dict = process_data_for_region_and_season(mdb, region_id, season_str, lead_int, spi_string_name)
plot_type='fct'
lt3_plot=obs_chart_with_triggers(plot_type,fct_df_lt3, 'year', 'percentage_spi', lt3_decided_fct_dict,row_annotations[2])
df_lt3=make_hit_misses(fct_df_lt3, obs_df, lt3_decided_fct_dict, obs_threshold_dict)
a

lead_int = 4
fct_df_lt4, obs_df, lt4_decided_fct_dict, obs_threshold_dict = process_data_for_region_and_season(mdb, region_id, season_str, lead_int, spi_string_name)
plot_type='fct'
lt4_plot=obs_chart_with_triggers(plot_type,fct_df_lt4, 'year', 'percentage_spi', lt4_decided_fct_dict,row_annotations[3])
df_lt4=make_hit_misses(fct_df_lt4, obs_df, lt4_decided_fct_dict, obs_threshold_dict)


df=pd.concat([df_lt2,df_lt3,df_lt4])
df1=df.reset_index()
df2=df1[['lead_time','cat','Trigger',  'hits', 'misses', 'FA','CN']]
df2 = df2.round({'Trigger': 1})

tab_plot=table(df2).properties(height=200,width=400)
#tab_plot


emtpy_plot=alt.Chart(pd.DataFrame({'A': []})).mark_text().encode().properties(
    width=400,
    height=200)

panels = alt.vconcat(
    alt.hconcat(obs_plot,lt2_plot),
    alt.hconcat(tab_plot, lt3_plot),
    alt.hconcat(emtpy_plot, lt4_plot),
)

panels.configure_view(stroke=None).configure_axisY(
    labelFontSize=12,
    titleFontSize=14
).configure_axisX(
    labelFontSize=10,
    titleFontSize=12
).configure_legend(
    labelFontSize=12,
    titleFontSize=14
)

panels.save(f"{data_path}{region_id}-{sc_season_str}.png")



# Usage:
mdb = pd.read_csv(f"{data_path}kimwa-metrix-v1.csv")
region_id = 2
season_str = 'OND'
sc_season_str=season_str.lower()
lead_int = 2
spi_string_name = 'spi3'
fct_df_lt2, obs_df, lt2_decided_fct_dict, obs_threshold_dict = process_data_for_region_and_season(mdb, region_id, season_str, lead_int, spi_string_name)

row_annotations=make_barchart_annotations()

plot_type='obs'
obs_plot=obs_chart_with_triggers(plot_type,obs_df, 'year', 'spi3', obs_threshold_dict,row_annotations[2])
plot_type='fct'
lt2_plot=obs_chart_with_triggers(plot_type,fct_df_lt2, 'year', 'percentage_spi', lt2_decided_fct_dict,row_annotations[1])
df_lt2=make_hit_misses(fct_df_lt2, obs_df, lt2_decided_fct_dict, obs_threshold_dict)

lead_int = 3
fct_df_lt3, obs_df, lt3_decided_fct_dict, obs_threshold_dict = process_data_for_region_and_season(mdb, region_id, season_str, lead_int, spi_string_name)
plot_type='fct'
lt3_plot=obs_chart_with_triggers(plot_type,fct_df_lt3, 'year', 'percentage_spi', lt3_decided_fct_dict,row_annotations[2])
df_lt3=make_hit_misses(fct_df_lt3, obs_df, lt3_decided_fct_dict, obs_threshold_dict)


lead_int = 4
fct_df_lt4, obs_df, lt4_decided_fct_dict, obs_threshold_dict = process_data_for_region_and_season(mdb, region_id, season_str, lead_int, spi_string_name)
plot_type='fct'
lt4_plot=obs_chart_with_triggers(plot_type,fct_df_lt4, 'year', 'percentage_spi', lt4_decided_fct_dict,row_annotations[3])
df_lt4=make_hit_misses(fct_df_lt4, obs_df, lt4_decided_fct_dict, obs_threshold_dict)


df=pd.concat([df_lt2,df_lt3,df_lt4])
df1=df.reset_index()
df2=df1[['lead_time','cat','Trigger',  'hits', 'misses', 'FA','CN']]
df2 = df2.round({'Trigger': 1})

tab_plot=table(df2).properties(height=200,width=400)
#tab_plot


emtpy_plot=alt.Chart(pd.DataFrame({'A': []})).mark_text().encode().properties(
    width=400,
    height=200)

panels = alt.vconcat(
    alt.hconcat(obs_plot,lt2_plot),
    alt.hconcat(tab_plot, lt3_plot),
    alt.hconcat(emtpy_plot, lt4_plot),
)

panels.configure_view(stroke=None).configure_axisY(
    labelFontSize=12,
    titleFontSize=14
).configure_axisX(
    labelFontSize=10,
    titleFontSize=12
).configure_legend(
    labelFontSize=12,
    titleFontSize=14
)

panels.save(f"{data_path}{region_id}-{sc_season_str}.png")
