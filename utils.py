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

load_dotenv()

data_path = os.getenv("data_path")


def ken_mask_creator():
    """
    Utiliity for generating region/district masks using regionmask library

    Returns
    -------
    the_mask : TYPE
        DESCRIPTION.
    rl_dict : TYPE
        DESCRIPTION.

    """
    dis = gp.read_file(f"{data_path}Karamoja_boundary_dissolved.shp")
    mbt_path = os.getenv("mbt_path")
    reg = gp.read_file(f"{data_path}wajir_mbt_extent.shp")
    mds = pd.concat([dis, reg])
    mds1 = mds.reset_index()
    mds1["region"] = [0, 1, 2]
    mds1["region_name"] = ["Karamoja", "Marsabit", "Wajir"]
    mds2 = mds1[["geometry", "region", "region_name"]]
    rl_dict = dict(zip(mds2.region, mds2.region_name))
    the_mask = regionmask.from_geopandas(mds2, numbers="region", overlap=True)
    return the_mask, rl_dict, mds2


def spi3_prod_name_creator(ds_ens, var_name):
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
    db = pd.DataFrame()
    db["dt"] = ds_ens[var_name].values
    db["dt1"] = db["dt"].apply(
        lambda x: datetime(x.year, x.month, x.day, x.hour, x.minute, x.second)
    )
    # db['dt1']=db['dt'].to_datetimeindex()
    db["month"] = db["dt1"].dt.strftime("%b").astype(str).str[0]
    db["year"] = db["dt1"].dt.strftime("%Y")
    db["spi_prod"] = (
        db.groupby("year")["month"].shift(2)
        + db.groupby("year")["month"].shift(1)
        + db.groupby("year")["month"].shift(0)
    )
    spi_prod_list = db["spi_prod"].tolist()
    return spi_prod_list


def spi4_prod_name_creator(ds_ens, var_name):
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
    db = pd.DataFrame()
    db["dt"] = ds_ens[var_name].values
    db["dt1"] = db["dt"].apply(
        lambda x: datetime(x.year, x.month, x.day, x.hour, x.minute, x.second)
    )
    # db['dt1']=db['dt'].to_datetimeindex()
    db["month"] = db["dt1"].dt.strftime("%b").astype(str).str[0]
    db["year"] = db["dt1"].dt.strftime("%Y")
    db["spi_prod"] = (
        db.groupby("year")["month"].shift(3)
        + db.groupby("year")["month"].shift(2)
        + db.groupby("year")["month"].shift(1)
        + db.groupby("year")["month"].shift(0)
    )
    spi_prod_list = db["spi_prod"].tolist()
    return spi_prod_list


def make_obs_fct_dataset(region_id, season_str, lead_int):
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
        kn_fct = xr.open_dataset(f"{data_path}clip_kn_fct_spi3.nc")
        kn_obs = xr.open_dataset(f"{data_path}clip_kn_obs_spi3.nc")
    else:
        kn_fct = xr.open_dataset(f"{data_path}clip_kn_fct_spi4.nc")
        kn_obs = xr.open_dataset(f"{data_path}clip_kn_obs_spi4.nc")
    the_mask, rl_dict, mds1 = ken_mask_creator()
    bounds = mds1.bounds
    # bounds.iloc[0].minx
    llon = bounds.iloc[region_id].minx
    llat = bounds.iloc[region_id].miny
    ulon = bounds.iloc[region_id].maxx
    ulat = bounds.iloc[region_id].maxy
    a_fc = kn_fct.sel(lon=slice(llon, ulon), lat=slice(llat, ulat))
    a_obs = kn_obs.sel(lon=slice(llon, ulon), lat=slice(llat, ulat))
    hindcast = HindcastEnsemble(a_fc)
    hindcast = hindcast.add_observations(a_obs)
    # hindcast
    # spi_cdb1spi3_prod_name_creator(ds_ens)
    a_fc1 = hindcast.get_initialized()
    a_fc2 = a_fc1.isel(lead=lead_int)
    if len(season_str) == 3:
        spi_prod_list = spi3_prod_name_creator(a_fc2, "valid_time")
        obs_spi_prod_list = spi3_prod_name_creator(a_obs, "time")
    else:
        spi_prod_list = spi4_prod_name_creator(a_fc2, "valid_time")
        obs_spi_prod_list = spi4_prod_name_creator(a_obs, "time")
    a_fc2 = a_fc2.assign_coords(spi_prod=("init", spi_prod_list))
    a_fc3 = a_fc2.where(a_fc2.spi_prod == season_str, drop=True)
    # obsertations
    a_obs1 = a_obs.assign_coords(spi_prod=("time", obs_spi_prod_list))
    a_obs2 = a_obs1.where(a_obs1.spi_prod == season_str, drop=True)
    # valid_time_series = a_fc3.valid_time.to_series().reset_index(drop=True).drop_duplicates()
    valid_time_flattened = (
        a_fc2.valid_time.to_dataframe()
        .reset_index()
        .drop_duplicates(subset="valid_time")["valid_time"]
    )
    valid_time_flattened.columns = ["valid_time", "cc"]
    # valid_time_flattened['valid_time'] = pd.to_datetime(valid_time_flattened['valid_time'])
    # Apply lambda function to create 'dt1' column
    # valid_time_flattened['dt1'] = valid_time_flattened['valid_time'].apply(
    #    lambda x: datetime(x.year, x.month, x.day, x.hour, x.minute, x.second)
    # )
    #
    valid_time_flattened["dt1"] = valid_time_flattened["valid_time"].apply(
        lambda x: datetime(x.year, x.month, x.day, x.hour, x.minute, x.second)
    )
    # Ensure the valid_time is in 'YYYY-MM-DD' string format
    # valid_time_flattened['dt2'] = valid_time_flattened['dt1'].dt.strftime('%Y-%m-%d')
    valid_time_flattened["dt1"] = valid_time_flattened["dt1"].dt.strftime(
        "%Y-%m-%dT%H:%M:%S.%f"
    )
    valid_time_flattened["dt1"] = pd.to_datetime(valid_time_flattened["dt1"])
    # Convert to xarray DataArray with time as the dimension name
    # valid_time_da = xr.DataArray(valid_time_flattened['dt1'], dims=['time'])
    valid_time_da = xr.DataArray(
        valid_time_flattened["dt1"], dims=["time"], coords=valid_time_flattened["dt1"]
    )
    a_obs3 = a_obs2.reindex(time=valid_time_da)
    # a_obs4 = a_obs3.reindex(time=a_obs2.time)
    # a_obs4 = a_obs3.sel(time=a_obs2.time, drop=True)
    a_obs3 = a_obs3.dropna(dim="time")
    if len(season_str) == 3:
        obs_data = a_obs3["spi3"]
        ens_data = a_fc3["spi3"]
    else:
        obs_data = a_obs3["spi4"]
        ens_data = a_fc3["spi4"]
    return obs_data, ens_data, a_fc, a_obs


def make_obs_fct_dataset(region_id, season_str, lead_int):
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
        kn_fct = xr.open_dataset(f"{data_path}kn_fct_spi3.nc")
        kn_obs = xr.open_dataset(f"{data_path}kn_obs_spi3.nc")
    else:
        kn_fct = xr.open_dataset(f"{data_path}kn_fct_spi4.nc")
        kn_obs = xr.open_dataset(f"{data_path}kn_obs_spi4.nc")
    the_mask, rl_dict, mds1 = ken_mask_creator()
    bounds = mds1.bounds
    # bounds.iloc[0].minx
    llon = bounds.iloc[region_id].minx
    llat = bounds.iloc[region_id].miny
    ulon = bounds.iloc[region_id].maxx
    ulat = bounds.iloc[region_id].maxy
    a_fc = kn_fct.sel(lon=slice(llon, ulon), lat=slice(llat, ulat))
    a_obs = kn_obs.sel(lon=slice(llon, ulon), lat=slice(llat, ulat))
    hindcast = HindcastEnsemble(a_fc)
    hindcast = hindcast.add_observations(a_obs)
    # hindcast
    # spi_cdb1spi3_prod_name_creator(ds_ens)
    a_fc1 = hindcast.get_initialized()
    a_fc2 = a_fc1.isel(lead=lead_int)
    if len(season_str) == 3:
        spi_prod_list = spi3_prod_name_creator(a_fc2, "valid_time")
        obs_spi_prod_list = spi3_prod_name_creator(a_obs, "time")
    else:
        spi_prod_list = spi4_prod_name_creator(a_fc2, "valid_time")
        obs_spi_prod_list = spi4_prod_name_creator(a_obs, "time")
    a_fc2 = a_fc2.assign_coords(spi_prod=("init", spi_prod_list))
    a_fc3 = a_fc2.where(a_fc2.spi_prod == season_str, drop=True)
    # obsertations
    a_obs1 = a_obs.assign_coords(spi_prod=("time", obs_spi_prod_list))
    a_obs2 = a_obs1.where(a_obs1.spi_prod == season_str, drop=True)
    # valid_time_series = a_fc3.valid_time.to_series().reset_index(drop=True).drop_duplicates()
    valid_time_flattened = (
        a_fc2.valid_time.to_dataframe()
        .reset_index()
        .drop_duplicates(subset="valid_time")["valid_time"]
    )
    valid_time_flattened.columns = ["valid_time", "cc"]
    # valid_time_flattened['valid_time'] = pd.to_datetime(valid_time_flattened['valid_time'])
    # Apply lambda function to create 'dt1' column
    # valid_time_flattened['dt1'] = valid_time_flattened['valid_time'].apply(
    #    lambda x: datetime(x.year, x.month, x.day, x.hour, x.minute, x.second)
    # )
    #
    valid_time_flattened["dt1"] = valid_time_flattened["valid_time"].apply(
        lambda x: datetime(x.year, x.month, x.day, x.hour, x.minute, x.second)
    )
    # Ensure the valid_time is in 'YYYY-MM-DD' string format
    # valid_time_flattened['dt2'] = valid_time_flattened['dt1'].dt.strftime('%Y-%m-%d')
    valid_time_flattened["dt1"] = valid_time_flattened["dt1"].dt.strftime(
        "%Y-%m-%dT%H:%M:%S.%f"
    )
    valid_time_flattened["dt1"] = pd.to_datetime(valid_time_flattened["dt1"])
    # Convert to xarray DataArray with time as the dimension name
    # valid_time_da = xr.DataArray(valid_time_flattened['dt1'], dims=['time'])
    valid_time_da = xr.DataArray(
        valid_time_flattened["dt1"], dims=["time"], coords=valid_time_flattened["dt1"]
    )
    a_obs3 = a_obs2.reindex(time=valid_time_da)
    # a_obs4 = a_obs3.reindex(time=a_obs2.time)
    # a_obs4 = a_obs3.sel(time=a_obs2.time, drop=True)
    a_obs3 = a_obs3.dropna(dim="time")
    if len(season_str) == 3:
        obs_data = a_obs3["spi3"]
        ens_data = a_fc3["spi3"]
    else:
        obs_data = a_obs3["spi4"]
        ens_data = a_fc3["spi4"]
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
    thresholds_dict = {
        (row["region_id"], row["season"]): {
            "mod": row["mod"],
            "sev": row["sev"],
            "ext": row["ext"],
        }
        for _, row in df.iterrows()
    }
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
            n1 = forecast_category_edges[i - 1]
        n2 = edge  # The current edge value
        if i == len(forecast_category_edges) - 1:
            # For the last element, there is no upper edge within the range, so you might set n3 to 1 or any other logic
            n3 = 1  # or edge itself if you want to keep it within valid probability bounds
        else:
            n3 = forecast_category_edges[i + 1]
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
        list_of_lists.append(
            [
                lowest_bound,
                sorted_thresholds[0],
                sorted_thresholds[1] if len(sorted_thresholds) > 1 else highest_bound,
            ]
        )

    # Loop through the sorted thresholds to create bins for the middle thresholds
    for i in range(1, len(sorted_thresholds) - 1):
        list_of_lists.append(
            [sorted_thresholds[i - 1], sorted_thresholds[i], sorted_thresholds[i + 1]]
        )

    # Handle the last bin separately if there are at least two thresholds
    if len(sorted_thresholds) > 1:
        list_of_lists.append(
            [sorted_thresholds[-2], sorted_thresholds[-1], highest_bound]
        )

    # Special case: If there is only one threshold, adjust the initial list to include highest_bound
    if len(sorted_thresholds) == 1:
        list_of_lists[0][
            -1
        ] = highest_bound  # Replace the last element of the first sublist with highest_bound

    return list_of_lists


def del_emprical_probablity(ens_data, threshold_dict):
    mod_thr = threshold_dict["mod"]
    fct_mod = (ens_data <= mod_thr).mean(dim="member")
    fct_mod_mean = fct_mod.mean(dim=["lat", "lon"])
    fct_mod_min = fct_mod.min(dim=["lat", "lon"])
    fct_mod_max = fct_mod.max(dim=["lat", "lon"])
    ####
    sev_thr = threshold_dict["sev"]
    fct_sev = (ens_data <= sev_thr).mean(dim="member")
    fct_sev_mean = fct_sev.mean(dim=["lat", "lon"])
    fct_sev_min = fct_sev.min(dim=["lat", "lon"])
    fct_sev_max = fct_sev.max(dim=["lat", "lon"])
    ext_thr = threshold_dict["ext"]
    fct_ext = (ens_data <= ext_thr).mean(dim="member")
    fct_ext_mean = fct_ext.mean(dim=["lat", "lon"])
    fct_ext_min = fct_ext.min(dim=["lat", "lon"])
    fct_ext_max = fct_ext.max(dim=["lat", "lon"])


def emprical_probablity(ens_data, threshold_dict):
    mod_thr = threshold_dict["mod"]
    fct_mod = (ens_data <= mod_thr).mean(dim="member")
    ####
    sev_thr = threshold_dict["sev"]
    fct_sev = (ens_data <= sev_thr).mean(dim="member")
    ####
    ext_thr = threshold_dict["ext"]
    fct_ext = (ens_data <= ext_thr).mean(dim="member")
    return fct_mod, fct_sev, fct_ext


def mean_emp_prob(fct_mod, fct_sev, fct_ext):
    fct_mod_mean = fct_mod.mean(dim=["lat", "lon"])
    fct_mod_df = fct_mod_mean.to_dataframe().reset_index()
    fct_mod_df1 = fct_mod_df[["valid_time", "spi3"]]
    fct_mod_df1 = fct_mod_df1.assign(cat="mod")
    fct_sev_mean = fct_sev.mean(dim=["lat", "lon"])
    fct_sev_df = fct_sev_mean.to_dataframe().reset_index()
    fct_sev_df1 = fct_sev_df[["valid_time", "spi3"]]
    fct_sev_df1 = fct_sev_df1.assign(cat="sev")
    fct_ext_mean = fct_ext.mean(dim=["lat", "lon"])
    fct_ext_df = fct_ext_mean.to_dataframe().reset_index()
    fct_ext_df1 = fct_ext_df[["valid_time", "spi3"]]
    fct_ext_df1 = fct_ext_df1.assign(cat="ext")
    wdf = pd.concat([fct_mod_df1, fct_sev_df1, fct_ext_df1])
    wdf["year0"] = wdf["valid_time"].apply(
        lambda x: datetime(x.year, x.month, x.day, x.hour, x.minute, x.second)
    )
    wdf["year"] = wdf["year0"].dt.strftime("%Y")
    wdf1 = wdf[["spi3", "cat", "year"]]
    return wdf1


def min_emp_prob(fct_mod, fct_sev, fct_ext):
    fct_mod_mean = fct_mod.min(dim=["lat", "lon"])
    fct_mod_df = fct_mod_mean.to_dataframe().reset_index()
    fct_mod_df1 = fct_mod_df[["valid_time", "spi3"]]
    fct_mod_df1 = fct_mod_df1.assign(cat="mod")
    fct_sev_mean = fct_sev.min(dim=["lat", "lon"])
    fct_sev_df = fct_sev_mean.to_dataframe().reset_index()
    fct_sev_df1 = fct_sev_df[["valid_time", "spi3"]]
    fct_sev_df1 = fct_sev_df1.assign(cat="sev")
    fct_ext_mean = fct_ext.min(dim=["lat", "lon"])
    fct_ext_df = fct_ext_mean.to_dataframe().reset_index()
    fct_ext_df1 = fct_ext_df[["valid_time", "spi3"]]
    fct_ext_df1 = fct_ext_df1.assign(cat="ext")
    wdf = pd.concat([fct_mod_df1, fct_sev_df1, fct_ext_df1])
    wdf["year0"] = wdf["valid_time"].apply(
        lambda x: datetime(x.year, x.month, x.day, x.hour, x.minute, x.second)
    )
    wdf["year"] = wdf["year0"].dt.strftime("%Y")
    wdf1 = wdf[["spi3", "cat", "year"]]
    return wdf1


def max_emp_prob(fct_mod, fct_sev, fct_ext):
    fct_mod_max = fct_mod.max(dim=["lat", "lon"])
    fct_mod_df = fct_mod_max.to_dataframe().reset_index()
    fct_mod_df1 = fct_mod_df[["valid_time", "spi3"]]
    fct_mod_df1 = fct_mod_df1.assign(cat="mod")
    fct_sev_max = fct_sev.max(dim=["lat", "lon"])
    fct_sev_df = fct_sev_max.to_dataframe().reset_index()
    fct_sev_df1 = fct_sev_df[["valid_time", "spi3"]]
    fct_sev_df1 = fct_sev_df1.assign(cat="sev")
    fct_ext_max = fct_ext.max(dim=["lat", "lon"])
    fct_ext_df = fct_ext_max.to_dataframe().reset_index()
    fct_ext_df1 = fct_ext_df[["valid_time", "spi3"]]
    fct_ext_df1 = fct_ext_df1.assign(cat="ext")
    wdf = pd.concat([fct_mod_df1, fct_sev_df1, fct_ext_df1])
    wdf["year0"] = wdf["valid_time"].apply(
        lambda x: datetime(x.year, x.month, x.day, x.hour, x.minute, x.second)
    )
    wdf["year"] = wdf["year0"].dt.strftime("%Y")
    wdf1 = wdf[["spi3", "cat", "year"]]
    return wdf1


def ep_process_data(region_id, season_str, lead_int):
    sc_season_str = season_str.lower()
    threshold_dict = get_threshold(region_id, sc_season_str)
    obs_data, ens_data, a_fc, a_obs = make_obs_fct_dataset(
        region_id, season_str, lead_int
    )
    fct_mod, fct_sev, fct_ext = emprical_probablity(ens_data, threshold_dict)

    df_mn = mean_emp_prob(fct_mod, fct_sev, fct_ext)
    df_mn = df_mn.assign(subset="mean", lt=str(lead_int))

    df_mi = min_emp_prob(fct_mod, fct_sev, fct_ext)
    df_mi = df_mi.assign(subset="min", lt=str(lead_int))

    df_mx = max_emp_prob(fct_mod, fct_sev, fct_ext)
    df_mx = df_mx.assign(subset="max", lt=str(lead_int))

    return df_mn, df_mi, df_mx
