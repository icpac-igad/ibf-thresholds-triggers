from io import StringIO
import os
from dotenv import load_dotenv
import logging
from pathlib import Path

import climpred
import xarray as xr
import xesmf as xe
import numpy as np
import pandas as pd
import regionmask
import geopandas as gp
from climpred import HindcastEnsemble
from datetime import datetime
from datatree import DataTree

import xhistogram.xarray as xhist
from sklearn.metrics import roc_auc_score

import xskillscore as xs
from xbootstrap import block_bootstrap
from dask.distributed import Client

# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import six
import textwrap as tw
from functools import reduce
import json
from dateutil.relativedelta import relativedelta
from calendar import monthrange
from PIL import Image

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# load_dotenv()

# data_path = os.getenv("data_path")


# latex_path = os.getenv("latex_path")


class BinCreateParams:
    def __init__(self, region_id, season_str, lead_int, level, spi_prod_name):
        self.region_id = region_id
        self.season_str = season_str
        self.lead_int = lead_int
        self.sc_season_str = season_str.lower()
        self.level = level
        self.spi_prod_name = spi_prod_name


def transform_data(data_at_time):
    """
    Transforms the input data to calculate total precipitation and adjust for the number of days in each month.

    Parameters:
    - data_at_time (xarray.Dataset): Input dataset containing precipitation data at different forecast times.

    Returns:
    - data_at_time_tp (xarray.Dataset): Transformed dataset with total precipitation adjusted for the number of days in each month.
    """
    valid_time = [
        pd.to_datetime(data_at_time.time.values) + relativedelta(months=fcmonth - 1)
        for fcmonth in data_at_time.forecastMonth
    ]
    data_at_time = data_at_time.assign_coords(valid_time=("forecastMonth", valid_time))
    numdays = [monthrange(dtat.year, dtat.month)[1] for dtat in valid_time]
    data_at_time = data_at_time.assign_coords(numdays=("forecastMonth", numdays))
    data_at_time_tp = data_at_time * data_at_time.numdays * 24 * 60 * 60 * 1000
    data_at_time_tp.attrs["units"] = "mm"
    data_at_time_tp.attrs["long_name"] = "Total precipitation"
    return data_at_time_tp


def apply_spi(cont_db, lead_val, spi_name_int):
    """
    Calculates given spi_name_int value Standardized Precipitation Index (SPI)
    for a specified lead time.

    Parameters:
    - cont_db (xarray.Dataset): The input dataset containing total monthly precipitation data.
    - lead_val (int): The lead time value for which the SPI is calculated.

    Returns:
    - cont_spi (list): A list of xarray.DataArrays containing the SPI values for each ensemble member.
    """
    lt1_db = cont_db.sel(forecastMonth=lead_val)
    lt1_db["tprate"].attrs["units"] = "mm/month"
    cont_spi = []
    for nsl in lt1_db.number.values:
        lt1_db2 = lt1_db.sel(number=nsl)
        # lt1_db3 = lt1_db2.chunk({'time': 4, 'latitude': 2, 'longitude': 2})
        lt1_db3 = lt1_db2.chunk(-1)
        aa = lt1_db3.tprate
        spi_3 = standardized_precipitation_index(
            aa,
            freq="MS",
            window=spi_name_int,
            dist="gamma",
            method="APP",
            cal_start="1991-01-01",
            cal_end="2018-01-01",
        )
        a_s3 = spi_3.compute()
        cont_spi.append(a_s3)
        aa = []
        lt1_db3 = []
        lt1_db2 = []
        print(nsl)
    return cont_spi


def apply_spii_mem(cont_db, lead_val, spi_name_int):
    """
    Calculates given spi_name_int value Standardized Precipitation Index (SPI)
    for a specified lead time.

    Parameters:
    - cont_db (xarray.Dataset): The input dataset containing total monthly precipitation data.
    - lead_val (int): The lead time value for which the SPI is calculated.

    Returns:
    - cont_spi (list): A list of xarray.DataArrays containing the SPI values for each ensemble member.
    """
    lt1_db = cont_db.sel(forecastMonth=lead_val)
    lt1_db["tprate"].attrs["units"] = "mm/month"
    cont_spi = []
    for nsl in lt1_db.number.values:
        lt1_db2 = lt1_db.sel(number=nsl)
        # lt1_db3 = lt1_db2.chunk({'time': 4, 'latitude': 2, 'longitude': 2})
        lt1_db3 = lt1_db2.chunk(-1)
        aa = lt1_db3.tprate
        spi_3 = standardized_precipitation_index(
            aa,
            freq="MS",
            window=spi_name_int,
            dist="gamma",
            method="APP",
            cal_start="2017-01-01",
            cal_end="2023-12-01",
        )
        a_s3 = spi_3.compute()
        cont_spi.append(a_s3)
        aa = []
        lt1_db3 = []
        lt1_db2 = []
        print(nsl)
    return cont_spi


def ken_mask_creator(data_path):
    """
    Utility for generating region/district masks using regionmask library

    Returns
    -------
    the_mask : regionmask.Regions
        The created mask for the regions.
    rl_dict : dict
        Dictionary mapping region numbers to region names.
    mds2 : geopandas.GeoDataFrame
        GeoDataFrame containing geometry, region, and region_name information.
    """
    logger.info("Starting ken_mask_creator function")

    try:
        logger.info(
            f"Reading Karamoja boundary file from {data_path}Karamoja_boundary_dissolved.shp"
        )
        dis = gp.read_file(f"{data_path}Karamoja_boundary_dissolved.shp")
        logger.info(
            f"Reading Wajir and Marsabit extent file from {data_path}wajir_mbt_extent.shp"
        )
        reg = gp.read_file(f"{data_path}wajir_mbt_extent.shp")

        # Check if the geometries are valid
        # if not dis.geometry.is_valid.all() or not reg.geometry.is_valid.all():
        #    raise ValueError("Invalid geometries found in shapefiles")

        logger.info("Concatenating district and region data")
        mds = pd.concat([dis, reg])
        mds1 = mds.reset_index()

        logger.info("Assigning region numbers and names")
        mds1["region"] = [0, 1, 2]
        mds1["region_name"] = ["Karamoja", "Marsabit", "Wajir"]
        mds2 = mds1[["geometry", "region", "region_name"]]
        # valid_types = ('Polygon', 'MultiPolygon')
        # if not all(geom.geom_type in valid_types for geom in mds2.geometry):
        #    raise ValueError("All geometries must be Polygon or MultiPolygon")
        if mds2.empty:
            raise ValueError("GeoDataFrame is empty")
        logger.info("Creating region-name dictionary")
        rl_dict = dict(zip(mds2.region, mds2.region_name))

        logger.info("Creating regionmask from GeoDataFrame")
        # mds2['geometry'] = mds2['geometry'].apply(lambda x: [x])
        # the_mask = regionmask.from_geopandas(mds2, numbers="region", overlap=False)
        the_mask = []

        logger.info("ken_mask_creator function completed successfully")
        return the_mask, rl_dict, mds2

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except Exception as e:
        logger.error(f"An error occurred in ken_mask_creator: {e}")
        raise


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


def make_obs_fct_dataset(data_path, region_id, season_str, lead_int):
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
    try:
        the_mask, rl_dict, mds1 = ken_mask_creator(data_path)
        bounds = mds1.bounds
        llon, llat = bounds.iloc[region_id][["minx", "miny"]]
        ulon, ulat = bounds.iloc[region_id][["maxx", "maxy"]]

        logger.debug(
            f"Region bounds: llon={llon}, llat={llat}, ulon={ulon}, ulat={ulat}"
        )

        if len(season_str) == 3:
            kn_fct = xr.open_dataset(f"{data_path}kn_fct_spi3_20240717.nc")
            kn_obs = xr.open_dataset(f"{data_path}kn_obs_spi3_20240717.nc")
            logger.info("Loaded SPI3 datasets")
        else:
            kn_fct = xr.open_dataset(f"{data_path}kn_fct_spi4.nc")
            kn_obs = xr.open_dataset(f"{data_path}kn_obs_spi4.nc")
            logger.info("Loaded SPI4 datasets")

        a_fc = kn_fct.sel(lon=slice(llon, ulon), lat=slice(llat, ulat))
        a_obs = kn_obs.sel(lon=slice(llon, ulon), lat=slice(llat, ulat))
        logger.info("subsetted obs and fcst to given region")
        logger.debug("Created HindcastEnsemble")
        hindcast = HindcastEnsemble(a_fc)
        hindcast = hindcast.add_observations(a_obs)

        a_fc1 = hindcast.get_initialized()
        logger.debug("Added climpred HindcastEnsemble to add valid_time in fcst")
        a_fc2 = a_fc1.isel(lead=lead_int)

        if len(season_str) == 3:
            spi_prod_list = spi3_prod_name_creator(a_fc2, "valid_time")
            obs_spi_prod_list = spi3_prod_name_creator(a_obs, "time")
        else:
            spi_prod_list = spi4_prod_name_creator(a_fc2, "valid_time")
            obs_spi_prod_list = spi4_prod_name_creator(a_obs, "time")
        logger.info(
            f"added SPI prodcut in obs and fcst dataset, filtered to {season_str}"
        )
        a_fc2 = a_fc2.assign_coords(spi_prod=("init", spi_prod_list))
        a_fc3 = a_fc2.where(a_fc2.spi_prod == season_str, drop=True)

        a_obs1 = a_obs.assign_coords(spi_prod=("time", obs_spi_prod_list))
        a_obs2 = a_obs1.where(a_obs1.spi_prod == season_str, drop=True)

        common_dates = np.unique(a_fc3.valid_time.values.ravel())
        a_obs3 = a_obs2.sel(time=common_dates)

        logger.info("Successfully prepared observed and forecasted datasets")
        return a_obs3, a_fc3

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except ValueError as e:
        logger.error(f"Value error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in make_obs_fct_dataset: {e}")
        raise
    return a_obs3, a_fc3


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
    data_v1 = region_id,region,season,mod,sev,ext
    0,kmj,mam,-0.03,-0.56,-0.99
    0,kmj,jjas,-0.01,-0.41,-0.99
    1,mbt,mam,-0.14,-0.38,-0.8
    1,mbt,ond,-0.15,-0.53,-0.71
    2,wjr,mam,-0.19,-0.45,-0.75
    2,wjr,ond,-0.29,-0.76,-0.9

    Example usage:
    >>> threshold = get_threshold(1, 'mam', 'mod')
    >>> print(threshold)
    -0.14
    """
    data = """region_id,region,season,mod,sev,ext
    0,kmj,mam,-0.03,-0.56,-0.99
    0,kmj,jjas,-0.01,-0.41,-0.99
    1,mbt,mam,-0.14,-0.38,-1.0
    1,mbt,ond,-0.44,-0.71,-1.0
    2,wjr,mam,-0.19,-0.45,-1.0
    2,wjr,ond,-0.46,-0.76,-1.0
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


def mean_obs_spi(obs_data, spi_string_name):
    obs_data_mean = obs_data.mean(dim=["lat", "lon"])
    obs_data_df = obs_data_mean.to_dataframe().reset_index()
    obs_data_df1 = obs_data_df[["time", spi_string_name]]
    wdf = obs_data_df1
    wdf["year0"] = wdf["time"].apply(
        lambda x: datetime(x.year, x.month, x.day, x.hour, x.minute, x.second)
    )
    wdf["year"] = wdf["year0"].dt.strftime("%Y")
    wdf1 = wdf[[spi_string_name, "year"]]
    return wdf1


def empirical_probability(ens_data, threshold_dict):
    """
    Calculate empirical probabilities for moderate, severe, and extreme drought conditions.

    Args:
        ens_data (xarray.DataArray): Ensemble data containing drought index values.
        threshold_dict (dict): Dictionary containing threshold values for moderate, severe, and extreme drought.

    Returns:
        tuple: Three xarray.DataArrays containing empirical probabilities for moderate, severe, and extreme drought.

    Raises:
        KeyError: If required keys are missing from threshold_dict.
        ValueError: If ens_data is not an xarray.DataArray or doesn't have a 'member' dimension.
    """
    try:
        if not isinstance(ens_data, xr.Dataset):
            raise ValueError("ens_data must be an xarray.Dataset")

        if "member" not in ens_data.dims:
            raise ValueError("ens_data must have a 'member' dimension")

        for key in ["mod", "sev", "ext"]:
            if key not in threshold_dict:
                raise KeyError(f"threshold_dict is missing required key: {key}")

        mod_thr = threshold_dict["mod"]
        fct_mod = (ens_data <= mod_thr).mean(dim="member")

        sev_thr = threshold_dict["sev"]
        fct_sev = (ens_data <= sev_thr).mean(dim="member")

        ext_thr = threshold_dict["ext"]
        fct_ext = (ens_data <= ext_thr).mean(dim="member")

        logger.info("Empirical probabilities calculated successfully")
        return fct_mod, fct_sev, fct_ext

    except Exception as e:
        logger.error(f"Error in empirical_probability: {str(e)}")
        raise


def seas51_patch_empirical_probability(ens_data, threshold_dict):
    """
    Calculate empirical probabilities for SEAS5.1 forecast system, handling the transition from 25(1981-2017) to 51(2017-current) members.

    Args:
        ens_data (xarray.DataArray): Ensemble data containing drought index values.
        threshold_dict (dict): Dictionary containing threshold values for moderate, severe, and extreme drought.

    Returns:
        tuple: Three xarray.DataArrays containing empirical probabilities for moderate, severe, and extreme drought.

    Raises:
        ValueError: If ens_data is not an xarray.DataArray or doesn't have required dimensions.
    """
    try:
        if not isinstance(ens_data, xr.Dataset):
            raise ValueError("ens_data must be an xarray.DataArray")

        if "init" not in ens_data.dims or "member" not in ens_data.dims:
            raise ValueError("ens_data must have 'init' and 'member' dimensions")

        m26_ens_data = ens_data.sel(init=slice("1981", "2016"))
        m26_ens_data1 = m26_ens_data.isel(member=slice(0, 25))
        m26_fct_mod, m26_fct_sev, m26_fct_ext = empirical_probability(
            m26_ens_data1, threshold_dict
        )

        m51_ens_data = ens_data.sel(init=slice("2017", None))
        m51_fct_mod, m51_fct_sev, m51_fct_ext = empirical_probability(
            m51_ens_data, threshold_dict
        )

        fct_mod = xr.concat(
            [m26_fct_mod, m51_fct_mod], dim="init", coords="minimal", compat="override"
        )
        fct_sev = xr.concat(
            [m26_fct_sev, m51_fct_sev], dim="init", coords="minimal", compat="override"
        )
        fct_ext = xr.concat(
            [m26_fct_ext, m51_fct_ext], dim="init", coords="minimal", compat="override"
        )

        logger.info("SEAS5.1 patch empirical probabilities calculated successfully")
        return fct_mod, fct_sev, fct_ext

    except Exception as e:
        logger.error(f"Error in seas51_patch_empirical_probability: {str(e)}")
        raise


def print_data_stats(obs_data, ens_prob_data, params):
    try:
        logger.info("Calculating and printing data statistics")
        obs_stats = obs_data[params.spi_prod_name].compute()
        ens_stats = ens_prob_data[params.spi_prod_name].compute()

        logger.info(
            f"Observed data stats: min={obs_stats.min().item():.4f}, "
            f"max={obs_stats.max().item():.4f}, "
            f"mean={obs_stats.mean().item():.4f}, "
            f"std={obs_stats.std().item():.4f}"
        )
        logger.info(
            f"Ensemble data stats: min={ens_stats.min().item():.4f}, "
            f"max={ens_stats.max().item():.4f}, "
            f"mean={ens_stats.mean().item():.4f}, "
            f"std={ens_stats.std().item():.4f}"
        )
    except Exception as e:
        logger.error(f"Error in print_data_stats: {str(e)}")
        raise


def print_histogram(ens_prob_data, params):
    try:
        logger.info("Calculating and printing histogram")
        ens_prob_data_np = ens_prob_data[params.spi_prod_name].values.flatten()
        hist, bin_edges = np.histogram(ens_prob_data_np, bins=20, range=(0, 1))
        logger.info("Histogram of drought forecast probabilities:")
        for i, count in enumerate(hist):
            logger.info(f"  {bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}: {count}")
    except Exception as e:
        logger.error(f"Error in print_histogram: {str(e)}")
        raise


def calculate_contingency_table(obs_event, forecast_event, params):
    try:
        logger.info("Calculating contingency table")
        obs_event1 = obs_event[params.spi_prod_name]
        obs_event1.name = "observed_event"
        forecast_event1 = forecast_event[params.spi_prod_name]
        forecast_event1.name = "forecasted_event"
        forecast_event3 = forecast_event1.assign_coords(
            init=forecast_event1.coords["valid_time"]
        )
        # Step 2: Rename 'valid_time' to 'time'
        forecast_event3 = forecast_event3.rename({"init": "time"})
        time_strings = [str(t) for t in forecast_event3["time"].values]
        # Step 2: Convert the strings to numpy.datetime64 in ISO 8601 format
        time_np64 = np.array(time_strings, dtype="datetime64[ns]")
        # Step 3: Update the 'time' coordinate with numpy.datetime64 values
        forecast_event3 = forecast_event3.assign_coords(time=time_np64)

        contingency_table = xhist.histogram(
            obs_event1, forecast_event3, bins=[2, 2], density=False, dim=["lat", "lon"]
        )
        logger.info(f"Contingency table shape: {contingency_table.shape}")
        logger.info(f"Contingency table contents:\n{contingency_table}")
        return contingency_table
    except Exception as e:
        logger.error(f"Error in calculate_contingency_table: {str(e)}")
        raise


def calculate_scores(contingency_table, trigger_value):
    try:
        logger.info(f"Calculating scores for trigger value {trigger_value:.4f}")
        # Ensure the contingency table has the expected dimensions
        if contingency_table.ndim != 3 or contingency_table.shape[1:] != (2, 2):
            raise ValueError(
                f"Unexpected contingency table shape: {contingency_table.shape}"
            )

        # Calculate scores for each time step
        time_steps = contingency_table.shape[0]
        scores = []

        for t in range(time_steps):
            ct = contingency_table[t].values.flatten()
            correct_negatives, false_alarms, misses, hits = ct
            total = hits + false_alarms + misses + correct_negatives

            hit_rate = hits / (hits + misses) if (hits + misses) > 0 else np.nan
            false_alarm_ratio = (
                false_alarms / (false_alarms + hits)
                if (false_alarms + hits) > 0
                else np.nan
            )
            bias_score = (
                (hits + false_alarms) / (hits + misses)
                if (hits + misses) > 0
                else np.nan
            )
            hanssen_kuipers_score = hit_rate - (
                false_alarms / (false_alarms + correct_negatives)
                if (false_alarms + correct_negatives) > 0
                else np.nan
            )
            heidke_skill_score = (
                (hits * correct_negatives - misses * false_alarms)
                / (
                    (hits + misses) * (misses + correct_negatives)
                    + (hits + false_alarms) * (false_alarms + correct_negatives)
                )
                if total > 0
                else np.nan
            )

            scores.append(
                {
                    "trigger_value": trigger_value,
                    "time_step": t,
                    "hit_rate": hit_rate,
                    "false_alarm_ratio": false_alarm_ratio,
                    "bias_score": bias_score,
                    "hanssen_kuipers_score": hanssen_kuipers_score,
                    "heidke_skill_score": heidke_skill_score,
                    "total": total,
                }
            )

        logger.info(f"Scores calculated for {time_steps} time steps")
        return scores
    except Exception as e:
        logger.error(f"Error in calculate_scores: {str(e)}")
        raise


def calculate_auroc(hits, misses, false_alarms, correct_negatives):
    """
    Calculates the Area Under the Receiver Operating Characteristic (AUROC) curve for a set of forecasts relative to observations.

    This function computes the AUROC score as a measure of the forecast's ability to discriminate between two classes:
    events that occurred (drought) and events that did not occur (no drought). The AUROC score ranges from 0 to 1,
    where a score of 0.5 suggests no discriminative ability (equivalent to random chance), and a score of 1 indicates perfect discrimination.

    Parameters:
    - hits (int): The number of correctly forecasted events (true positives).
    - misses (int): The number of events that were observed but not forecasted (false negatives).
    - false_alarms (int): The number of non-events that were incorrectly forecasted as events (false positives).
    - correct_negatives (int): The number of non-events that were correctly forecasted (true negatives).

    Returns:
    - auroc (float): The calculated AUROC score for the given contingency table values.

    Note:
    - This function is designed to work with binary classification problems, such as predicting the occurrence or non-occurrence of drought events.
    - It requires the `roc_auc_score` function from the `sklearn.metrics` module and `numpy` for handling arrays.

    Example usage:
    >>> auroc_score = calculate_auroc(50, 30, 20, 100)
    >>> print(f"AUROC Score: {auroc_score}")
    """
    total_positives = hits + misses
    total_negatives = correct_negatives + false_alarms
    y_true = np.concatenate((np.ones(total_positives), np.zeros(total_negatives)))
    y_scores = np.concatenate(
        (np.ones(hits), np.zeros(misses + false_alarms + correct_negatives))
    )
    auroc = roc_auc_score(y_true, y_scores)
    return auroc


def calculate_auroc_score(contingency_table, trigger_value, n_bootstrap=1000):
    try:
        logger.info(
            f"Calculating AUROC score for trigger value {trigger_value:.4f} with {n_bootstrap} bootstrap iterations"
        )
        # Ensure the contingency table has the expected dimensions
        if contingency_table.ndim != 3 or contingency_table.shape[1:] != (2, 2):
            raise ValueError(
                f"Unexpected contingency table shape: {contingency_table.shape}"
            )

        # Sum the contingency table across all time steps
        summed_table = contingency_table.sum(dim="time")

        # Flatten the summed contingency table
        ct = summed_table.values.flatten()
        correct_negatives, false_alarms, misses, hits = ct
        total = hits + false_alarms + misses + correct_negatives

        auroc_bootstrap_scores = []
        for _ in range(n_bootstrap):
            bootstrap_counts = np.random.multinomial(
                total,
                [
                    hits / total,
                    misses / total,
                    false_alarms / total,
                    correct_negatives / total,
                ],
                size=1,
            )
            (
                bootstrap_hits,
                bootstrap_misses,
                bootstrap_false_alarms,
                bootstrap_correct_negatives,
            ) = bootstrap_counts[0]
            auroc_bootstrap_scores.append(
                calculate_auroc(
                    bootstrap_hits,
                    bootstrap_misses,
                    bootstrap_false_alarms,
                    bootstrap_correct_negatives,
                )
            )

        auroc_score = np.mean(auroc_bootstrap_scores)
        auroc_lb, auroc_ub = np.percentile(auroc_bootstrap_scores, [2.5, 97.5])

        logger.info(f"AUROC score calculated for trigger value {trigger_value:.4f}")
        return {
            "trigger_value": trigger_value,
            "auroc_score": auroc_score,
            "auroc_lb": auroc_lb,
            "auroc_ub": auroc_ub,
        }
    except Exception as e:
        logger.error(f"Error in calculate_auroc_score: {str(e)}")
        raise


def process_trigger_values(
    obs_data, ens_prob_data, params, threshold, calculate_auroc=True
):
    try:
        logger.info("Processing trigger values")
        trigger_values = xr.DataArray(
            np.linspace(0, 1, num=100), dims=["trigger_value"]
        )
        results = []
        auroc_results = []

        obs_event = obs_data <= threshold

        for i, trigger_value in enumerate(trigger_values):
            forecast_event = ens_prob_data >= trigger_value
            contingency_table = calculate_contingency_table(
                obs_event, forecast_event, params
            )

            scores = calculate_scores(contingency_table, trigger_value.item())
            results.extend(scores)

            if calculate_auroc:
                auroc_score = calculate_auroc_score(
                    contingency_table, trigger_value.item()
                )
                auroc_results.append(auroc_score)

            if i % 10 == 0:
                logger.info(f"Processed trigger value: {trigger_value:.2f}")

        # Create the main DataFrame
        df = pd.DataFrame(results)

        # If AUROC was calculated, merge it with the main DataFrame
        if calculate_auroc:
            auroc_df = pd.DataFrame(auroc_results)
            df = pd.merge(df, auroc_df, on="trigger_value", how="left")

        return df
    except Exception as e:
        logger.error(f"Error in process_trigger_values: {str(e)}")
        raise


def xhist_metrics_2d(obs_data, ens_prob_data, params, calculate_auroc=True):
    try:
        logger.info("Starting xhist_metrics_2d calculation")
        threshold_dict = get_threshold(params.region_id, params.sc_season_str)
        threshold = threshold_dict[params.level]

        logger.info(f"Threshold: {threshold:.4f}")
        print_data_stats(obs_data, ens_prob_data, params)
        print_histogram(ens_prob_data, params)
        logger.info("Started all possible trigger value contingency table creation ")
        df = process_trigger_values(
            obs_data, ens_prob_data, params, threshold, calculate_auroc
        )

        output_file = f"{params.region_id}_{params.season_str}_{params.level}_lt{params.lead_int}.csv"
        df.to_csv(output_file)
        logger.info(f"Results saved to {output_file}")
        return df
    except Exception as e:
        logger.error(f"Error in xhist_metrics_2d: {str(e)}")
        raise


def xhist_metrices_1d(pdb, trigger_value, threshold_dict, cat_str):
    ds = xr.Dataset.from_dataframe(pdb)
    obs_ext = ds[f"spi3_{cat_str}"]
    fct_ext = ds[f"ep_{cat_str}"]
    obs_event = obs_ext <= threshold_dict[cat_str]
    fct_event = fct_ext >= trigger_value
    obs_event_int = obs_event.astype(int)
    fct_event_int = fct_event.astype(int)
    contingency_table = xhist.histogram(
        obs_event_int, fct_event_int, bins=[2, 2], density=False
    )
    contingency_table = contingency_table.data
    correct_negatives = contingency_table[0, 0]
    false_alarms = contingency_table[0, 1]
    misses = contingency_table[1, 0]
    hits = contingency_table[1, 1]
    total = hits + false_alarms + misses + correct_negatives
    hit_rates = hits / (hits + misses) if (hits + misses) > 0 else np.nan
    false_alarm_ratios = (
        false_alarms / (false_alarms + hits) if (false_alarms + hits) > 0 else np.nan
    )
    # false_alarm_ratios[i] = false_alarms / (false_alarms + correct_negatives) if (false_alarms + correct_negatives) > 0 else np.nan
    bias_scores = (
        (hits + false_alarms) / (hits + misses) if (hits + misses) > 0 else np.nan
    )
    n_hit_rates = np.mean(hits.astype(int))  # Calculate hit rate as mean of hits
    n_false_alarm_ratios = np.mean(false_alarm_ratios.astype(int))
    hanssen_kuipers_scores = n_hit_rates - n_false_alarm_ratios
    heidke_skill_scores = (hits * correct_negatives - misses * false_alarms) / total

    fct_ext_pb = fct_ext / 100
    tv_pb = trigger_value / 100
    o1 = block_bootstrap(
        obs_event_int,
        blocks={"index": 1},
        n_iteration=1000,
        circular=True,
    )
    f1 = block_bootstrap(
        fct_ext_pb,
        blocks={"index": 1},
        n_iteration=1000,
        circular=True,
    )
    fpr, tpr, auroc_bootstrap_scores = xs.roc(
        o1,
        f1,
        bin_edges=[0, tv_pb, 1],
        dim=["index"],
        return_results="all_as_metric_dim",
    )
    auroc_scores = np.mean(auroc_bootstrap_scores)
    auroc_lb, auroc_ub = np.percentile(auroc_bootstrap_scores, [2.5, 97.5])
    df = pd.DataFrame(
        {
            "#dry-seas": len(obs_ext.index.values),
            "hits": [hits],
            "misses": [misses],
            "FA": [false_alarms],
            "CN": [correct_negatives],
            "hit_rates": [hit_rates],
            "false_alarm_ratios": [false_alarm_ratios],
            "bias_scores": [bias_scores],
            "hanssen_kuipers_scores": [hanssen_kuipers_scores],
            "heidke_skill_scores": [heidke_skill_scores],
            "auroc_scores": auroc_scores.values,
            "auroc_lb": auroc_lb,
            "auroc_ub": auroc_ub,
        }
    )
    df.insert(0, "threshold", threshold_dict[cat_str])
    df.insert(0, "trigger_values", trigger_value)
    return df


def mean_emp_prob(fct_mod, fct_sev, fct_ext, spi_string_name):
    fct_mod_mean = fct_mod.mean(dim=["lat", "lon"])
    fct_mod_df = fct_mod_mean.to_dataframe().reset_index()
    fct_mod_df1 = fct_mod_df[["valid_time", spi_string_name]]
    fct_mod_df1 = fct_mod_df1.assign(cat="mod")
    fct_sev_mean = fct_sev.mean(dim=["lat", "lon"])
    fct_sev_df = fct_sev_mean.to_dataframe().reset_index()
    fct_sev_df1 = fct_sev_df[["valid_time", spi_string_name]]
    fct_sev_df1 = fct_sev_df1.assign(cat="sev")
    fct_ext_mean = fct_ext.mean(dim=["lat", "lon"])
    fct_ext_df = fct_ext_mean.to_dataframe().reset_index()
    fct_ext_df1 = fct_ext_df[["valid_time", spi_string_name]]
    fct_ext_df1 = fct_ext_df1.assign(cat="ext")
    wdf = pd.concat([fct_mod_df1, fct_sev_df1, fct_ext_df1])
    wdf["year0"] = wdf["valid_time"].apply(
        lambda x: datetime(x.year, x.month, x.day, x.hour, x.minute, x.second)
    )
    wdf["year"] = wdf["year0"].dt.strftime("%Y")
    wdf1 = wdf[[spi_string_name, "cat", "year"]]
    wdf1.columns = ["ep", "cat", "year"]
    # wdf1['ep']=wdf1['ep']*100
    wdf1.loc[:, "ep"] = wdf1["ep"] * 100
    return wdf1



