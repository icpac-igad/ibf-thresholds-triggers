from io import StringIO
import os
from dotenv import load_dotenv
import logging

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


import numpy as np
import pandas as pd

import pandas as pd
import numpy as np
import matplotlib

import os
from dotenv import load_dotenv

# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import six
from datetime import datetime
import textwrap as tw
from functools import reduce
import json
from dateutil.relativedelta import relativedelta
from calendar import monthrange


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#load_dotenv()

#data_path = os.getenv("data_path")


#latex_path = os.getenv("latex_path")

def transform_data(data_at_time):
    """
    Transforms the input data to calculate total precipitation and adjust for the number of days in each month.

    Parameters:
    - data_at_time (xarray.Dataset): Input dataset containing precipitation data at different forecast times.

    Returns:
    - data_at_time_tp (xarray.Dataset): Transformed dataset with total precipitation adjusted for the number of days in each month.
    """
    valid_time = [pd.to_datetime(data_at_time.time.values) + relativedelta(months=fcmonth-1) 
                  for fcmonth in data_at_time.forecastMonth]
    data_at_time = data_at_time.assign_coords(valid_time=('forecastMonth', valid_time))
    numdays = [monthrange(dtat.year, dtat.month)[1] for dtat in valid_time]
    data_at_time = data_at_time.assign_coords(numdays=('forecastMonth', numdays))
    data_at_time_tp = data_at_time * data_at_time.numdays * 24 * 60 * 60 * 1000
    data_at_time_tp.attrs['units'] = 'mm'
    data_at_time_tp.attrs['long_name'] = 'Total precipitation' 
    return data_at_time_tp


def apply_spi(cont_db,lead_val,spi_name_int):
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
    lt1_db['tprate'].attrs['units'] = 'mm/month'
    cont_spi=[]
    for nsl in lt1_db.number.values:
        lt1_db2=lt1_db.sel(number=nsl)
        #lt1_db3 = lt1_db2.chunk({'time': 4, 'latitude': 2, 'longitude': 2})
        lt1_db3 = lt1_db2.chunk(-1)
        aa=lt1_db3.tprate
        spi_3 = standardized_precipitation_index(
             aa,
             freq="MS",
             window=spi_name_int,
             dist="gamma",
             method="APP",
             cal_start='1991-01-01',
             cal_end='2018-01-01',
        )  
        a_s3=spi_3.compute()
        cont_spi.append(a_s3)
        aa=[]
        lt1_db3 = []
        lt1_db2 = []
        print(nsl)
    return cont_spi


def apply_spii_mem(cont_db,lead_val,spi_name_int):
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
    lt1_db['tprate'].attrs['units'] = 'mm/month'
    cont_spi=[]
    for nsl in lt1_db.number.values:
        lt1_db2=lt1_db.sel(number=nsl)
        #lt1_db3 = lt1_db2.chunk({'time': 4, 'latitude': 2, 'longitude': 2})
        lt1_db3 = lt1_db2.chunk(-1)
        aa=lt1_db3.tprate
        spi_3 = standardized_precipitation_index(
             aa,
             freq="MS",
             window=spi_name_int,
             dist="gamma",
             method="APP",
             cal_start='2017-01-01',
             cal_end='2023-12-01',
        )  
        a_s3=spi_3.compute()
        cont_spi.append(a_s3)
        aa=[]
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
        logger.info(f"Reading Karamoja boundary file from {data_path}Karamoja_boundary_dissolved.shp")
        dis = gp.read_file(f"{data_path}Karamoja_boundary_dissolved.shp")
        logger.info(f"Reading Wajir and Marsabit extent file from {data_path}wajir_mbt_extent.shp")
        reg = gp.read_file(f"{data_path}wajir_mbt_extent.shp")

        # Check if the geometries are valid
        #if not dis.geometry.is_valid.all() or not reg.geometry.is_valid.all():
        #    raise ValueError("Invalid geometries found in shapefiles")

        logger.info("Concatenating district and region data")
        mds = pd.concat([dis, reg])
        mds1 = mds.reset_index()

        logger.info("Assigning region numbers and names")
        mds1["region"] = [0, 1, 2]
        mds1["region_name"] = ["Karamoja", "Marsabit", "Wajir"]
        mds2 = mds1[["geometry", "region", "region_name"]]
        #valid_types = ('Polygon', 'MultiPolygon')
        #if not all(geom.geom_type in valid_types for geom in mds2.geometry):
        #    raise ValueError("All geometries must be Polygon or MultiPolygon")
        if mds2.empty:
            raise ValueError("GeoDataFrame is empty")
        logger.info("Creating region-name dictionary")
        rl_dict = dict(zip(mds2.region, mds2.region_name))

        logger.info("Creating regionmask from GeoDataFrame")
        #mds2['geometry'] = mds2['geometry'].apply(lambda x: [x])
        #the_mask = regionmask.from_geopandas(mds2, numbers="region", overlap=False)
        the_mask=[]

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


def make_obs_fct_dataset(data_path,region_id, season_str, lead_int):
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
        llon, llat = bounds.iloc[region_id][['minx', 'miny']]
        ulon, ulat = bounds.iloc[region_id][['maxx', 'maxy']]
        
        logger.debug(f"Region bounds: llon={llon}, llat={llat}, ulon={ulon}, ulat={ulat}")

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
        logger.info(f"added SPI prodcut in obs and fcst dataset, filtered to {season_str}")
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
        
        if 'member' not in ens_data.dims:
            raise ValueError("ens_data must have a 'member' dimension")

        for key in ['mod', 'sev', 'ext']:
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
        
        if 'init' not in ens_data.dims or 'member' not in ens_data.dims:
            raise ValueError("ens_data must have 'init' and 'member' dimensions")

        m26_ens_data = ens_data.sel(init=slice('1981', '2016'))
        m26_ens_data1 = m26_ens_data.isel(member=slice(0, 25))
        m26_fct_mod, m26_fct_sev, m26_fct_ext = empirical_probability(m26_ens_data1, threshold_dict)

        m51_ens_data = ens_data.sel(init=slice('2017', None))
        m51_fct_mod, m51_fct_sev, m51_fct_ext = empirical_probability(m51_ens_data, threshold_dict)

        fct_mod = xr.concat([m26_fct_mod, m51_fct_mod], dim='init', coords='minimal', compat='override')
        fct_sev = xr.concat([m26_fct_sev, m51_fct_sev], dim='init', coords='minimal', compat='override')
        fct_ext = xr.concat([m26_fct_ext, m51_fct_ext], dim='init', coords='minimal', compat='override')


        logger.info("SEAS5.1 patch empirical probabilities calculated successfully")
        return fct_mod, fct_sev, fct_ext

    except Exception as e:
        logger.error(f"Error in seas51_patch_empirical_probability: {str(e)}")
        raise



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


# Calculate AUROC using bootstrap
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


def get_subset(dfa, cat_str):
    # Filter out rows with null values in 'hit_rate' and 'false_alarm_ratio'
    # df = df.dropna(subset=['hit_rate', 'false_alarm_ratio'])
    df = dfa[dfa["cat"] == cat_str]
    # Sort the DataFrame by 'peirce_score' in descending order
    df = df.sort_values(by="hanssen_kuipers_scores", ascending=False)

    # Get the row with the maximum 'peirce_score'
    max_peirce_row = df.iloc[0]

    # Sort the DataFrame by 'bias_score' in descending order, and filter for 'bias_score' < 1.0
    df = df.loc[df["bias_scores"] < 1.0].sort_values(by="bias_scores", ascending=False)

    # Get the row with the maximum 'bias_score' < 1.0
    max_bias_row = df.iloc[0]

    # Sort the DataFrame by 'heidke_score' in descending order
    df = df.sort_values(by="heidke_skill_scores", ascending=False)

    # Get the row with the maximum 'heidke_score'
    max_heidke_row = df.iloc[0]

    # Combine the three rows into a subset
    subset = pd.concat(
        [
            pd.DataFrame([max_peirce_row]),
            pd.DataFrame([max_bias_row]),
            pd.DataFrame([max_heidke_row]),
        ],
        ignore_index=True,
    )

    return subset


def trigger_decision_dict(df0):
    df = df0[df0["auroc_scores"] >= 0.5]
    df_mod = get_subset(df, "mod")
    mod_max_cn = df_mod["CN"].max()
    mod_df_max_cn = df_mod[df_mod["CN"] == mod_max_cn]
    mod_max_hits = mod_df_max_cn["hits"].max()
    mod_df_max_hits = mod_df_max_cn[mod_df_max_cn["hits"] == mod_max_hits]

    df_sev = get_subset(df, "sev")
    sev_max_cn = df_sev["CN"].max()
    sev_df_max_cn = df_sev[df_sev["CN"] == sev_max_cn]
    sev_max_hits = sev_df_max_cn["hits"].max()
    sev_df_max_hits = sev_df_max_cn[sev_df_max_cn["hits"] == sev_max_hits]

    df_ext = get_subset(df, "ext")
    ext_max_cn = df_ext["CN"].max()
    ext_df_max_cn = df_ext[df_ext["CN"] == ext_max_cn]
    ext_max_hits = ext_df_max_cn["hits"].max()
    ext_df_max_hits = ext_df_max_cn[ext_df_max_cn["hits"] == ext_max_hits]
    tri_dict = {
        "mod": mod_df_max_hits["trigger_values"].values[0],
        "sev": sev_df_max_hits["trigger_values"].values[0],
        "ext": ext_df_max_hits["trigger_values"].values[0],
    }
    df0 = pd.concat([mod_df_max_hits, sev_df_max_hits, ext_df_max_hits])
    return tri_dict, df0


def get_mean_ens_triggers(data_path,region_id, season_str, lead_int):
    if len(season_str) == 3:
        spi_string_name = "spi3"
    else:
        spi_string_name = "spi4"
    sc_season_str = season_str.lower()
    obs_data, ens_data = make_obs_fct_dataset(data_path,region_id, season_str, lead_int)
    obs_df = mean_obs_spi(obs_data, spi_string_name)
    threshold_dict = get_threshold(region_id, sc_season_str)
    ###
    m26_ens_data=ens_data.isel(init=slice(0,36))
    m26_ens_data1=m26_ens_data.isel(member=slice(0, 25))
    m26_fct_mod, m26_fct_sev, m26_fct_ext=emprical_probablity(m26_ens_data1, threshold_dict)
    m51_ens_data=ens_data.isel(init=slice(36,len(ens_data)))
    m51_fct_mod, m51_fct_sev, m51_fct_ext=emprical_probablity(m51_ens_data, threshold_dict)
    fct_mod=xr.concat([m26_fct_mod,m51_fct_mod],dim='init')
    fct_sev=xr.concat([m26_fct_sev,m51_fct_sev],dim='init')
    fct_ext=xr.concat([m26_fct_ext,m51_fct_ext],dim='init')
    ####
    #fct_mod, fct_sev, fct_ext = emprical_probablity(ens_data, threshold_dict)
    fct_df = mean_emp_prob(fct_mod, fct_sev, fct_ext, spi_string_name)
    db = pd.merge(fct_df, obs_df, on="year")
    pdb = db.pivot(index="year", columns="cat", values=["spi3", "ep"])
    pdb.columns = ["{}_{}".format(val[0], val[1]) for val in pdb.columns]
    pdb1 = pdb[pdb["spi3_ext"] <= 0]
    pdb2 = pdb.reset_index()
    cnt_df = []
    for idx, row in pdb2.iterrows():
        mod_trigger_value = row["ep_mod"]
        mod_df = xhist_metrices(pdb2, mod_trigger_value, threshold_dict, "mod")
        mod_df.insert(0, "region", region_id)
        mod_df.insert(1, "season", season_str)
        mod_df.insert(2, "cat", "mod")
        mod_df.insert(3, "year", row["year"])
        cnt_df.append(mod_df)

        sev_trigger_value = row["ep_sev"]
        sev_df = xhist_metrices(pdb2, sev_trigger_value, threshold_dict, "sev")
        sev_df.insert(0, "region", region_id)
        sev_df.insert(1, "season", season_str)
        sev_df.insert(2, "cat", "sev")
        sev_df.insert(3, "year", row["year"])
        cnt_df.append(sev_df)

        ext_trigger_value = row["ep_ext"]
        ext_df = xhist_metrices(pdb2, ext_trigger_value, threshold_dict, "ext")
        ext_df.insert(0, "region", region_id)
        ext_df.insert(1, "season", season_str)
        ext_df.insert(2, "cat", "ext")
        ext_df.insert(3, "year", row["year"])
        cnt_df.append(ext_df)

    metrix_df = pd.concat(cnt_df)
    decision_dict, decision_df = trigger_decision_dict(metrix_df)
    decision_df["lead_time"] = lead_int
    pdb_melt = pdb.rename(columns={"ep_ext": "ext", "ep_sev": "sev", "ep_mod": "mod"})
    plot_df = pd.melt(
        pdb_melt.reset_index(),
        id_vars=["year"],
        value_vars=["mod", "sev", "ext"],
        var_name="cat",
        value_name="ep_pb",
    )
    return obs_df, fct_df, metrix_df, decision_dict, decision_df, plot_df



def prepare_data_for_concat(data, ens_data, dataset_type):
    """
    Prepares data (observations or calcualted emprical probablity dataset-triggers ) to be concatenated with ensemble data.

    Parameters:
    data (xarray.Dataset): The original dataset (observations or forecasts).
    ens_data (xarray.Dataset): The ensemble dataset to match structure with.
    dataset_type (str): A string identifier for the type of dataset ('obs', 'fmod', 'fsev', 'fext').

    Returns:
    xarray.Dataset: The prepared dataset ready for concatenation.
    """
    try:
        logger.info(f"Preparing {dataset_type} data for concatenation")
        
        # Identify the variable name (assumes single variable dataset)
        var_name = list(data.data_vars)[0]
        logger.debug(f"Variable name identified: {var_name}")

        if 'init' not in data.coords:
            data = data.rename({'time': 'init'})
            logger.debug("Renamed 'time' coordinate to 'init'")

        # Extend the 'init' dimension to match ens_data
        extended_data = np.full((len(ens_data.init), len(data.lat), len(data.lon)), np.nan)
        extended_data[:len(data.init), :, :] = data[var_name].values
        logger.debug(f"Extended data shape: {extended_data.shape}")

        # Create a new DataArray with extended data and matching coordinates
        data_extended = xr.DataArray(
            extended_data,
            dims=['init', 'lat', 'lon'],
            coords={'init': ens_data['init'], 'lat': data['lat'], 'lon': data['lon']},
            name=var_name
        )

        # Convert DataArray to Dataset
        data_ex = data_extended.to_dataset()

        # Add a new coordinate to identify the dataset type
        data_ex = data_ex.expand_dims({"dataset": [dataset_type]})
        
        logger.info(f"Successfully prepared {dataset_type} data for concatenation")
        return data_ex

    except KeyError as e:
        logger.error(f"KeyError in prepare_data_for_concat: {str(e)}")
        raise
    except ValueError as e:
        logger.error(f"ValueError in prepare_data_for_concat: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in prepare_data_for_concat: {str(e)}")
        raise

def helper_stamp_plot(ens_data, obs_data, fct_mod, fct_sev, fct_ext):
    """
    Helper function to prepare and combine data for stamp plot.

    Parameters:
    ens_data (xarray.Dataset): Ensemble data.
    obs_data (xarray.Dataset): Observation data.
    fct_mod (xarray.Dataset): Moderate forecast data.
    fct_sev (xarray.Dataset): Severe forecast data.
    fct_ext (xarray.Dataset): Extreme forecast data.

    Returns:
    xarray.Dataset: Combined dataset for stamp plot.
    """
    try:
        logger.info("Starting helper_stamp_plot function")

        obs_cast = prepare_data_for_concat(obs_data, ens_data, 'obs')
        fmod_cast = prepare_data_for_concat(fct_mod, ens_data, 'fmod')
        fsev_cast = prepare_data_for_concat(fct_sev, ens_data, 'fsev')
        fext_cast = prepare_data_for_concat(fct_ext, ens_data, 'fext')

        ens_data_prepared = ens_data.expand_dims({"dataset": ["ens"]})
        logger.debug("All datasets prepared for concatenation")

        # Concatenate all datasets along the new 'dataset' dimension
        combined_data = xr.concat([ens_data_prepared, obs_cast, fmod_cast, fsev_cast, fext_cast], dim="dataset")
        logger.debug("Datasets concatenated successfully")

        # Create a mapping between dataset types and numeric values
        dataset_mapping = {'ens': 0, 'obs': 51, 'fmod': 52, 'fsev': 53, 'fext': 54}
        combined_data = combined_data.assign_coords(dataset_num=("dataset", [dataset_mapping[d] for d in combined_data.dataset.values]))
        logger.info(f'made the combined_data as {combined_data}')        
        logger.info("helper_stamp_plot function completed successfully")
        return combined_data

    except KeyError as e:
        logger.error(f"KeyError in helper_stamp_plot: {str(e)}")
        raise
    except ValueError as e:
        logger.error(f"ValueError in helper_stamp_plot: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in helper_stamp_plot: {str(e)}")
        raise


def create_single_row_plot(tree, init, variable='spi3', output_dir='single_row_plots', is_last_plot=False):
    members = list(tree['ensemble'].children.keys())
    valid_times = tree['ensemble/member_0'].ds.valid_time.values
    lats = tree['ensemble/member_0'].ds.lat.values
    lons = tree['ensemble/member_0'].ds.lon.values
    num_members = len(members)
    num_additional_plots = 4  # Obs, mod, sev, ext
    total_plots = num_members + num_additional_plots
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create a wide figure for a single row
    fig, axs = plt.subplots(1, total_plots, figsize=(2 * total_plots, 2), 
                            subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Define the color scale ranges
    ensemble_cmap_range = (-4, 4)
    fct_cmap_range = (0.0, 1.0)
    
    valid_time = valid_times[np.where(tree['ensemble/member_0'].ds.init.values == init)[0][0]]
    
    # Plot ensemble members
    for j, member_key in enumerate(members):
        member_data = tree[f'ensemble/{member_key}'].ds[variable]
        data = member_data.sel(init=init).values
        im = axs[j].pcolormesh(lons, lats, data, cmap='RdBu', 
                               transform=ccrs.PlateCarree(), 
                               vmin=ensemble_cmap_range[0], vmax=ensemble_cmap_range[1])
        axs[j].set_title(f'm{j}', fontsize=6)
        axs[j].set_xticks([])
        axs[j].set_yticks([])
    
    # Add the observation and additional models as the last plots
    plot_titles = ['Obs', 'mod', 'sev', 'ext']
    plot_keys = ['observation', 'fct_mod', 'fct_sev', 'fct_ext']
    for k, (title, key) in enumerate(zip(plot_titles, plot_keys)):
        dataset = tree[key].ds[variable]
        if 'time' in dataset.coords:
            coord_key = 'time'
            obs_init = np.datetime64(valid_time.strftime('%Y-%m-%d %H:%M:%S'))
        elif 'init' in dataset.coords:
            coord_key = 'init'
            obs_init = init
        else:
            raise ValueError(f"Neither 'time' nor 'init' found in dataset coordinates for {key}")
        
        obs_data = dataset.sel({coord_key: obs_init}).values
        
        # Use different color scales for different plot keys
        if key == 'observation':
            vmin, vmax = ensemble_cmap_range
            cmap = 'RdBu'
        else:
            vmin, vmax = fct_cmap_range
            cmap = 'Blues'
        
        im = axs[num_members + k].pcolormesh(lons, lats, obs_data, cmap=cmap, 
                                             transform=ccrs.PlateCarree(), 
                                             vmin=vmin, vmax=vmax)
        axs[num_members + k].set_title(f'{title}', fontsize=6)
        axs[num_members + k].set_xticks([])
        axs[num_members + k].set_yticks([])
    # if is_last_plot:
    #     # Add colorbar for ensemble and observation
    #     cbar_ax = fig.add_axes([0.92, 0.2, 0.01, 0.6])
    #     cbar = plt.colorbar(im, cax=cbar_ax)
    #     cbar.set_label('SPI3 (Ensemble & Obs)')

    #     # Add colorbar for forecasts
    #     cbar_ax2 = fig.add_axes([0.94, 0.2, 0.01, 0.6])
    #     cbar2 = plt.colorbar(axs[-1].collections[0], cax=cbar_ax2)
    #     cbar2.set_label('Forecasts (mod/sev/ext)')

    #     # Adjust layout to accommodate colorbars
    #     plt.subplots_adjust(right=0.91)
    # else:
    #     plt.tight_layout()

    if is_last_plot:
        # ... (rest of the plotting code is the same until the colorbar section)

        # Create a horizontal colorbar axis below the plots
        cbar_ax = fig.add_axes([0.95, 0.5, 0.05, 0.1])  # Adjust position and size as needed

        # Colorbar for ensemble and observation (wider)
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('SPI3 (Ensemble & Obs)')

        # Colorbar for forecasts (narrower, to the right)
        cbar_ax2 = fig.add_axes([0.95, 0.2, 0.05, 0.1]) 
        cbar2 = plt.colorbar(axs[-1].collections[0], cax=cbar_ax2, orientation='horizontal')
        cbar2.set_label('Forecasts (mod/sev/ext)')

    else:
        plt.tight_layout()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/stamp_plot_{init.strftime("%Y%m%d")}.png', dpi=100, bbox_inches='tight')
    plt.close()

# Example usage:
inits = tree['ensemble/member_0'].ds.init.values
for i, init in enumerate(inits[35:None]):
    is_last_plot = (i == len(inits[35:None]) - 1)
    create_single_row_plot(tree, init, is_last_plot=is_last_plot)


def merge_png_files(input_dir='single_row_plots', output_file='merged_stamp_plots.png'):
    # Get all PNG files in the input directory
    png_files = sorted(Path(input_dir).glob('*.png'))
    
    # Open the first image to get dimensions
    with Image.open(png_files[0]) as img:
        row_width, row_height = img.size
    
    # Create a new image with the calculated dimensions
    merged_height = row_height * len(png_files)
    merged_image = Image.new('RGB', (row_width, merged_height))
    
    # Paste each row image into the merged image
    for i, png_file in enumerate(png_files):
        with Image.open(png_file) as img:
            merged_image.paste(img, (0, i * row_height))
    
    # Save the merged image
    merged_image.save(output_file, dpi=(300, 300))
    print(f"Merged image saved as {output_file}")

def DEPR_arrange_obs_fct_stampplot(obs_data,ens_data):
    """
    take forecast and observations dataset into single xarray dataset
    The learning curve on extending a xarray is large and the lines in this funcitons
    are added after lot of iterations

    Where the observation dataset is added as an 51th memeber to have a stampl plot of forecast versus observations

    """
    obs_data1=obs_data.to_dataset()
    ens_data1=ens_data.to_dataset()
    obs_data1 = obs_data1.rename_dims({'time': 'init'})

    # Step 2: Extend the 'init' dimension in obs_data1 to match the length of 'init' in ens_data1
    # Create a new array with NaN values for the 43rd time step
    extended_spi3 = np.full((43,13, 13), np.nan)
    extended_spi3[:42,:,:] = obs_data1['spi3'].values

    # Create a new 'init' coordinate with 43 time steps
    new_init = ens_data1['init']

    # Create a new DataArray for the extended obs_data1
    obs_data1_extended = xr.DataArray(
        extended_spi3,
        dims=['init','lat', 'lon'],
        coords={'init': new_init,'lat': obs_data1['lat'], 'lon': obs_data1['lon'] },
        name='spi3'
    )

    obs_data_ex=obs_data1_extended.to_dataset()

    member_coord = xr.DataArray([51], dims="member")

    # Expand the Dataset with the new dimension
    obs_data_ex = obs_data_ex.expand_dims(
        {"member": member_coord}
    )

    obs_data_ex = obs_data_ex.assign_coords(member=[52])
    ens_data2 = ens_data1.rename({'number':'member'})
    ens_data3 = ens_data2.set_xindex('member')

    ds = xr.concat([ens_data3, obs_data_ex], dim='member')
    return ds 


def plot_obs_fct_stamp(dataset, region, lead_time, variable='spi3'):
    members = dataset.member.values
    inits = dataset.init.values
    lats = dataset.lat.values
    lons = dataset.lon.values

    fig = plt.figure(figsize=(24, 20))  # Adjusted figure size for the new layout
    
    for i, init in enumerate(inits):
        for j, member in enumerate(members):
            ax = fig.add_subplot(len(inits), len(members), i*len(members) + j + 1,
                                 projection=ccrs.PlateCarree())
            
            data = dataset[variable].sel(member=member, init=init).values
            
            # Plot the data
            im = ax.pcolormesh(lons, lats, data, cmap='RdBu', 
                               transform=ccrs.PlateCarree(), 
                               vmin=-2, vmax=2)
            
            ax.set_title(f'M{member}-{init.strftime("%Y")}', fontsize=6)
            
            # Remove axis labels for cleaner look
            ax.set_xticks([])
            ax.set_yticks([])
            
    # Add a colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='SPI3')
    
    plt.tight_layout()
    plt.savefig(f'{region}_stamp_plots_{lead_time}.png', dpi=300, bbox_inches='tight')
    plt.close()
    


def plot_create_obs_fct_stamps(dataset, variable='spi3'):
    members = dataset.member.values
    inits = dataset.init.values
    lats = dataset.lat.values
    lons = dataset.lon.values

    fig = plt.figure(figsize=(24, 20))  # Adjusted figure size for the new layout
    
    for i, init in enumerate(inits):
        for j, member in enumerate(members):
            ax = fig.add_subplot(len(inits), len(members), i*len(members) + j + 1,
                                 projection=ccrs.PlateCarree())
            
            data = dataset[variable].sel(member=member, init=init).values
            
            # Plot the data
            im = ax.pcolormesh(lons, lats, data, cmap='RdBu', 
                               transform=ccrs.PlateCarree(), 
                               vmin=-2, vmax=2)
            
            ax.set_title(f'M{member}-{init.strftime("%Y")}', fontsize=6)
            
            # Remove axis labels for cleaner look
            ax.set_xticks([])
            ax.set_yticks([])
            
    # Add a colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='SPI3')
    
    plt.tight_layout()
    plt.savefig(f'init_rows_20240810.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_obs_chart_with_triggers(
    plot_type, df, year_column, spi_column, threshold_dict, row_annotations
):
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
    if plot_type == "obs":
        bar_chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X(f"{year_column}:N", axis=alt.Axis(labelAngle=90)),
                y=alt.Y(
                    f"{spi_column}:Q", title=spi_column, scale=alt.Scale(domain=[-4, 4])
                ),
                color=alt.condition(
                    alt.datum[spi_column] > 0,
                    alt.value("blue"),  # Color for positive values
                    alt.value("red"),  # Color for negative values
                ),
            )
            .properties(width=400, height=200)
        )
    else:
        color_scale = alt.Scale(
            # domain=["ext", "sev", "mod"], range=["#880203", "#ffa400", "#fffe00"]
            domain=["mod", "sev", "ext"],
            range=["#f4eb13", "#f89821", "#ed2227"],
        )

        bar_chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X(f"{year_column}:N", axis=alt.Axis(labelAngle=90)),
                y=alt.Y(f"{spi_column}:Q", title="Probability (%)", stack=None),
                color=alt.Color("cat:N", scale=color_scale, sort=["sev", "mod", "ext"]),
            )
            .properties(width=400, height=200)
            + row_annotations
        )

    # Adding trigger lines
    rules = []
    for key, value in threshold_dict.items():
        rule = (
            alt.Chart(pd.DataFrame({"y": [value]}))
            .mark_rule(
                strokeWidth=2,
                stroke={"ext": "#ed2227", "sev": "#f89821", "mod": "#f4eb13"}[
                    key
                ],  # Conditional color assignment
            )
            .encode(y="y:Q")
        )
        rules.append(rule)

    # Combine the bar chart with trigger lines
    final_chart = alt.layer(bar_chart, *rules)

    return final_chart


def aux_plot_make_barchart_annotations():
    row_annotations = [
        alt.Chart(pd.DataFrame({"text": ["lt=1"]}))
        .mark_text(
            align="left",
            baseline="middle",
            fontSize=14,
            fontWeight="bold",
            dx=-190,
            dy=-90,
        )
        .encode(text="text:N")
        .properties(width=400, height=200),
        alt.Chart(pd.DataFrame({"text": ["lt=2, Sep"]}))
        .mark_text(
            align="left",
            baseline="middle",
            fontSize=14,
            fontWeight="bold",
            dx=-190,
            dy=-90,
        )
        .encode(text="text:N")
        .properties(width=400, height=200),
        alt.Chart(pd.DataFrame({"text": ["lt=3, Aug"]}))
        .mark_text(
            align="left",
            baseline="middle",
            fontSize=14,
            fontWeight="bold",
            dx=-190,
            dy=-90,
        )
        .encode(text="text:N")
        .properties(width=400, height=200),
        alt.Chart(pd.DataFrame({"text": ["lt=4, Jul"]}))
        .mark_text(
            align="left",
            baseline="middle",
            fontSize=14,
            fontWeight="bold",
            dx=-190,
            dy=-90,
        )
        .encode(text="text:N")
        .properties(width=400, height=200),
        alt.Chart(pd.DataFrame({"text": ["lt=5"]}))
        .mark_text(
            align="left",
            baseline="middle",
            fontSize=14,
            fontWeight="bold",
            dx=-190,
            dy=-90,
        )
        .encode(text="text:N")
        .properties(width=400, height=200),
    ]
    return row_annotations


def plot_decision_table(df):
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
                    ticks=False,
                ),
                scale=alt.Scale(padding=10),
                sort=None,
            ),
            alt.Y("index", type="ordinal", axis=None),
            alt.Text("value", type="nominal"),
        )
    )


def aux_mt_plot_create_month_column(df):
    new_column = []

    for _, row in df.iterrows():
        lt = row["lt"]
        cat = row["cat"]
        season = row["season"]

        if season == "MAM":
            if lt == 1:
                if cat == "mod":
                    new_column.append("mar_x")
                elif cat == "sev":
                    new_column.append("mar_y")
                elif cat == "ext":
                    new_column.append("mar_z")
            elif lt == 2:
                if cat == "mod":
                    new_column.append("feb_x")
                elif cat == "sev":
                    new_column.append("feb_y")
                elif cat == "ext":
                    new_column.append("feb_z")
            elif lt == 3:
                if cat == "mod":
                    new_column.append("jan_x")
                elif cat == "sev":
                    new_column.append("jan_y")
                elif cat == "ext":
                    new_column.append("jan_z")
            elif lt == 4:
                if cat == "mod":
                    new_column.append("dec_x")
                elif cat == "sev":
                    new_column.append("dec_y")
                elif cat == "ext":
                    new_column.append("dec_z")
            elif lt == 5:
                if cat == "mod":
                    new_column.append("nov_x")
                elif cat == "sev":
                    new_column.append("nov_y")
                elif cat == "ext":
                    new_column.append("nov_z")
        elif season == "OND":
            if lt == 1:
                if cat == "mod":
                    new_column.append("oct_x")
                elif cat == "sev":
                    new_column.append("oct_y")
                elif cat == "ext":
                    new_column.append("oct_z")
            elif lt == 2:
                if cat == "mod":
                    new_column.append("sep_x")
                elif cat == "sev":
                    new_column.append("sep_y")
                elif cat == "ext":
                    new_column.append("sep_z")
            elif lt == 3:
                if cat == "mod":
                    new_column.append("aug_x")
                elif cat == "sev":
                    new_column.append("aug_y")
                elif cat == "ext":
                    new_column.append("aug_z")
            elif lt == 4:
                if cat == "mod":
                    new_column.append("jul_x")
                elif cat == "sev":
                    new_column.append("jul_y")
                elif cat == "ext":
                    new_column.append("jul_z")
            elif lt == 5:
                if cat == "mod":
                    new_column.append("jun_x")
                elif cat == "sev":
                    new_column.append("jun_y")
                elif cat == "ext":
                    new_column.append("jun_z")
        elif season == "JJAS":
            if lt == 2:
                if cat == "mod":
                    new_column.append("jun_x")
                elif cat == "sev":
                    new_column.append("jun_y")
                elif cat == "ext":
                    new_column.append("jun_z")
            elif lt == 3:
                if cat == "mod":
                    new_column.append("may_x")
                elif cat == "sev":
                    new_column.append("may_y")
                elif cat == "ext":
                    new_column.append("may_z")
            elif lt == 4:
                if cat == "mod":
                    new_column.append("apr_x")
                elif cat == "sev":
                    new_column.append("apr_y")
                elif cat == "ext":
                    new_column.append("apr_z")
            elif lt == 5:
                if cat == "mod":
                    new_column.append("mar_x")
                elif cat == "sev":
                    new_column.append("mar_y")
                elif cat == "ext":
                    new_column.append("mar_z")
        else:
            new_column.append("")

    #df["new_column"] = new_column
    df.insert(loc=0, column='new_column', value=new_column)
    return df


def aux_mt_plot_replace_with_list(x):
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


def aux_mt_plot_round_list(lst, decimal_places):
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


# %% table plot matplotlib

### Define the picture size and remove the ticks


### functions for whole column, row editing
def aux_mt_plot_legend_maker(text1, color_list, legend_title):
    square6 = plt.Rectangle((0.4, 0.1), 0.15, 0.25, color=color_list[0], clip_on=False)
    text1.add_artist(square6)
    square5 = plt.Rectangle((0.55, 0.1), 0.15, 0.25, color=color_list[1], clip_on=False)
    text1.add_artist(square5)
    square5 = plt.Rectangle((0.7, 0.1), 0.15, 0.25, color=color_list[2], clip_on=False)
    text1.add_artist(square5)
    square5 = plt.Rectangle((0.85, 0.1), 0.15, 0.25, color=color_list[3], clip_on=False)
    text1.add_artist(square5)
    square5 = plt.Rectangle((1.0, 0.1), 0.15, 0.25, color=color_list[4], clip_on=False)
    text1.add_artist(square5)
    plt.text(
        0.6,
        0.4,
        legend_title,
        horizontalalignment="left",
        fontsize=6,
        fontweight="bold",
        color="k",
        verticalalignment="center",
        transform=text1.transAxes,
    )
    plt.text(
        0.42,
        0.05,
        "<20",
        horizontalalignment="left",
        fontsize=6,
        fontweight="bold",
        color="k",
        verticalalignment="center",
        transform=text1.transAxes,
    )
    plt.text(
        0.57,
        0.05,
        "20-40",
        horizontalalignment="left",
        fontsize=6,
        fontweight="bold",
        color="k",
        verticalalignment="center",
        transform=text1.transAxes,
    )
    plt.text(
        0.72,
        0.05,
        "40-60",
        horizontalalignment="left",
        fontsize=6,
        fontweight="bold",
        color="k",
        verticalalignment="center",
        transform=text1.transAxes,
    )
    plt.text(
        0.87,
        0.05,
        "60-80",
        horizontalalignment="left",
        fontsize=6,
        fontweight="bold",
        color="k",
        verticalalignment="center",
        transform=text1.transAxes,
    )
    plt.text(
        1.05,
        0.05,
        "80<",
        horizontalalignment="left",
        fontsize=6,
        fontweight="bold",
        color="k",
        verticalalignment="center",
        transform=text1.transAxes,
    )


def aux_mt_plot_set_align_for_column(table, col, align="left"):
    cells = [key for key in table._cells if key[1] == col]
    for cell in cells:
        table._cells[cell]._loc = align


def aux_mt_plot_set_width_for_column(table, col, width):
    cells = [key for key in table._cells if key[1] == col]
    for cell in cells:
        table._cells[cell]._width = width


def aux_mt_plot_set_height_for_row(table, row, height):
    cells = [key for key in table._cells if key[0] == row]
    for cell in cells:
        table._cells[cell]._height = height


def aux_mt_plot_colorcell(tablerows, tablecols, cellDict, color_list):
    allcells = [(x, y) for x in tablerows[1:] for y in tablecols[2:]]
    for alcls in allcells:
        cell_value0 = json.loads(cellDict[alcls]._text.get_text())[0]
        if cell_value0 == -999.0:
            cellDict[alcls].set_facecolor("#FFFFFF")
        else:
            if float(cell_value0) <= 0.2:
                cellDict[alcls].set_facecolor(color_list[0])
            elif 0.2 < float(cell_value0) <= 0.4:
                cellDict[alcls].set_facecolor(color_list[1])
            elif 0.4 < float(cell_value0) <= 0.6:
                cellDict[alcls].set_facecolor(color_list[2])
            elif 0.6 < float(cell_value0) <= 0.8:
                cellDict[alcls].set_facecolor(color_list[3])
            elif 0.8 < float(cell_value0) <= 1.0:
                cellDict[alcls].set_facecolor(color_list[4])
            else:
                cellDict[alcls].set_facecolor("#FFFFFF")


def aux_mt_plot_remove_value(tablerows, tablecols, mpl_table):
    allcells = [(x, y) for x in tablerows[1:] for y in tablecols[2:]]
    for alcls in allcells:
        mpl_table._cells[alcls]._text.set_text("")


def aux_mt_plot_add_certain_value(tablerows, tablecols, mpl_table, cellDict):
    allcells = [(x, y) for x in tablerows[1:] for y in tablecols[2:]]
    for alcls in allcells:
        # print(cellDict[alcls]._text.get_text())
        cell_value0 = json.loads(cellDict[alcls]._text.get_text())[1]
        mpl_table._cells[alcls]._text.set_text("")
        # cell_value0=(cellDict[alcls]._text.get_text())
        if cell_value0 == -999.0:
            mpl_table._cells[alcls]._text.set_text("")
        elif cell_value0 == 999.0:
            mpl_table._cells[alcls]._text.set_text("")
        else:
            ncl = "%.1f" % cell_value0
            mpl_table._cells[alcls]._text.set_text(ncl)


def aux_mt_plot_aset_height_for_row_except_head(table, rowlist, height):
    cells_list = []
    for row in rowlist:
        cells = [key for key in table._cells if key[0] == row]
        cells_list.append(cells)
    for cells in cells_list:
        for cell in cells:
            table._cells[cell]._height = height


def aux_mt_plot_bset_height_for_row_except_head(table, rowlist, height):
    for row in rowlist:
        for col in range(len(table[row])):
            cell = table[row, col]
            cell._height = height


def aux_mt_plot_cset_height_for_row_except_head(table, row_height):
    """chatGPT function"""
    for i, cell in six.iteritems(table._cells):
        if i[0] == 0:  # Skip header row
            continue
        cell.set_height(row_height)


def aux_mt_plot_set_height_for_row_except_head(cellDict, header_row_count, height):
    for cell_key, cell in cellDict.items():
        row, col = cell_key
        if row < header_row_count:
            continue  # skip header rows
        cell.set_height(height)


def aux_mt_plot_table_header_colour(tablerows, tablecols, cellDict, mpl_table):
    allcells = [(x, y) for x in tablerows[0:1] for y in tablecols]
    header_list = [
        "Region",
        "SPI",
        "Jul",
        "Aug",
        "Sep",
        "",
        "Jul",
        "Aug",
        "Sep",
        "",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "",
        "Nov",
        "Dec",
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "",
        "Nov",
        "Dec",
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "",
    ]
    for idx, alcls in enumerate(allcells):
        cellDict[alcls].set_facecolor("#FFFFFF")
        print(header_list[idx])
        text = header_list[idx]
        mpl_table._cells[alcls]._text.set_text(text)


### funciton for table creation
def aux_mt_plot_render_mpl_table(
    data,
    color_list,
    col_width=1.0,
    row_height=0.425,
    font_size=5,
    header_color="#40466e",
    row_colors=["#f1f1f2", "w"],
    edge_color="w",
    bbox=[0, 0, 1, 1],
    header_columns=0,
    ax=None,
    **kwargs,
):
    """
    Renders a matplotlib table from a pandas DataFrame, allowing for customization of various aesthetic parameters.

    Parameters:
    - data (pandas.DataFrame): The data to display in the table.
    - color_list (list): A list of colors to use for cell background coloring based on cell values.
    - col_width (float): The width of the columns. Default is 1.0.
    - row_height (float): The height of the rows. Default is 0.625.
    - font_size (int): Font size for the cell texts. Default is 5.
    - header_color (str): Color code or name for the table header's background. Default is '#40466e'.
    - row_colors (list): A list containing color codes for alternating row colors. Default is ['#f1f1f2', 'w'].
    - edge_color (str): Color code or name for the cell edge lines. Default is 'w' (white).
    - bbox (list): A 4-element list defining the bounding box of the table within the plot. Default is [0, 0, 1, 1].
    - header_columns (int): The number of initial columns considered as header columns. Default is 0.
    - ax (matplotlib.axes.Axes): The matplotlib axes object where the table will be rendered. If None, a new one will be created.

    Returns:
    - ax (matplotlib.axes.Axes): The matplotlib axes object with the rendered table.

    This function creates a visual representation of a DataFrame as a static table in a matplotlib figure. It allows for
    significant customization, including cell coloring based on values, flexible sizing, font adjustments, and more. The
    function is particularly useful for creating detailed reports or visual summaries of data within a matplotlib figure.

    Additional keyword arguments (**kwargs) are passed directly to the `matplotlib.axes.Axes.table` method.
    """
    mpl_table = ax.table(
        cellText=data.values, bbox=bbox, colLabels=[""] * 42, cellLoc="center", **kwargs
    )
    set_align_for_column(mpl_table, col=0, align="left")
    set_width_for_column(mpl_table, 0, 0.6)
    set_width_for_column(mpl_table, 1, 0.5)
    for idx in range(2, 42):
        set_width_for_column(mpl_table, idx, 0.2)
    set_height_for_row(mpl_table, 0, 0.01)
    # set_height_for_row_except_head(mpl_table, np.arange(1, len(data.index)), 0.06)
    # set_height_for_row_except_head(mpl_table, row_height=0.03)
    cellDict = mpl_table.get_celld()
    set_height_for_row_except_head(cellDict, header_row_count=1, height=0.03)
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)
    cellDict = mpl_table.get_celld()
    tablerows = np.arange(0, len(data.index) + 1)
    tablecols = np.arange(0, len(data.columns))
    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight="bold", color="black")
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])
    colorcell(tablerows, tablecols, cellDict, color_list)
    headings = data.columns
    plt.text(
        0.245,
        1.08,
        "Mild",
        fontsize=10,
        fontweight="bold",
        color="black",
        ha="left",
        va="center",
        transform=ax.transAxes,
    )
    plt.text(
        0.545,
        1.08,
        "Moderate",
        fontsize=10,
        fontweight="bold",
        color="black",
        ha="left",
        va="center",
        transform=ax.transAxes,
    )
    plt.text(
        0.845,
        1.08,
        "Severe",
        fontsize=10,
        fontweight="bold",
        color="black",
        ha="left",
        va="center",
        transform=ax.transAxes,
    )
    table_header_colour(tablerows, tablecols, cellDict, mpl_table)
    add_certain_value(tablerows, tablecols, mpl_table, cellDict)
    return ax


def aux_mt_plot_plot_data_table(latex_path,data_table, stat_var, req_list,region_id):
    # Width and height of A4 portrait with 1-inch margins
    width = 3.67 - 2  # one inch margin on each side
    height = 11.69 - 2  # one inch margin on the top and bottom
    fig = plt.figure()
    fig.set_size_inches(height, width)
    # [left, bottom, width, height]
    table = fig.add_axes([0.04, 0.15, 0.93, 0.75], frame_on=False)
    table.xaxis.set_ticks_position("none")
    table.yaxis.set_ticks_position("none")
    table.set_xticklabels("")
    table.set_yticklabels("")
    #######
    laxes = fig.add_axes([0.22, 0.01, 0.4, 0.3], frame_on=False, zorder=0)
    laxes.xaxis.set_ticks_position("none")
    laxes.yaxis.set_ticks_position("none")
    laxes.set_xticklabels("")
    laxes.set_yticklabels("")
    if stat_var == "FAR":
        legend_title = "False Alarm Ratio %"
        color_list = ["#009600", "#64C800", "#ffff00", "#ff7800", "#ff0000"]
    else:
        legend_title = "Hit Rate %"
        color_list = ["#ff0000", "#ff7800", "#ffff00", "#64C800", "#009600"]
    legend_maker(laxes, color_list, legend_title)
    #####
    data1 = data_table[req_list]
    print(data1.info())
    mpl_table = render_mpl_table(
        data1, color_list, header_columns=0, col_width=0.2, ax=table
    )
    # cellDict = mpl_table.get_celld()
    # set_height_for_row_except_head(cellDict, header_row_count=1, height=0.06)
    # set_height_for_row_except_head(mpl_table, row_height=0.125)
    var = stat_var.lower()
    plt.show
    plt.savefig(f"{latex_path}{var}_{region_id}_prob_v20240703.jpg", dpi=300)


def aux_mt_plot_pass_month_get_colnames(months):
    original_list = [
        "region_x",
        "season",
        "nov_x",
        "dec_x",
        "jan_x",
        "feb_x",
        "mar_x",
        "apr_x",
        "may_x",
        "jun_x",
        "jul_x",
        "aug_x",
        "sep_x",
        "oct_x",
        "empty1",
        "nov_y",
        "dec_y",
        "jan_y",
        "feb_y",
        "mar_y",
        "apr_y",
        "may_y",
        "jun_y",
        "jul_y",
        "aug_y",
        "sep_y",
        "oct_y",
        "empty2",
        "nov_z",
        "dec_z",
        "jan_z",
        "feb_z",
        "mar_z",
        "apr_z",
        "may_z",
        "jun_z",
        "jul_z",
        "aug_z",
        "sep_z",
        "oct_z",
    ]
    # months = ['jul', 'aug', 'sep']
    suffixes = ["_x", "_y", "_z"]

    organized_list = [
        "region_x",
        "season",
    ]

    for suffix in suffixes:
        for month in months:
            item = month + suffix
            if item in original_list:
                organized_list.append(item)

        if suffix == "_x":
            organized_list.append("empty1")
        elif suffix == "_y":
            organized_list.append("empty2")
    return organized_list


def aux_mt_plot_table_df(tab_df, stat_var):
    tab_df_a = tab_df.rename(columns={"lead_time": "lt"})
    tab_df_m = create_month_column(tab_df_a)
    tab_df_m["pod_v"] = tab_df_m.apply(
        lambda x: [x["hit_rates"], x["trigger_values"]], axis=1
    )
    tab_df_m["far_v"] = tab_df_m.apply(
        lambda x: [x["false_alarm_ratios"], x["trigger_values"]], axis=1
    )
    tab_df_m["pod_v"] = tab_df_m["pod_v"].apply(lambda x: round_list(x, 2))
    tab_df_m["far_v"] = tab_df_m["far_v"].apply(lambda x: round_list(x, 2))
    mapping_dict = {0: "Karamoja", 1: "Marsabit", 2: "Wajir"}
    tab_df_m["region_x"] = tab_df_m["region"].replace(mapping_dict)
    p = tab_df_m.pivot_table(
        index=["region_x", "season"],
        columns="new_column",
        values="pod_v",
        aggfunc="first",
    )
    pf = p.reset_index()
    # Apply the custom function to each cell in the DataFrame
    pf1 = pf.applymap(replace_with_list)
    pf1.columns.name = None
    pf1["empty1"] = [[-999.0, -999.0]] * len(pf1)
    pf1["empty2"] = [[-999.0, -999.0]] * len(pf1)
    months = ["jul", "aug", "sep"]
    organized_list = pass_month_get_colnames(months)
    pf2 = pf1[organized_list]
    mask = (pf2["region_x"].isin(["Marsabit", "Wajir"])) & (pf2["season"] == "OND")
    pf3 = pf2[mask]
    plot_data_table(pf3, stat_var, organized_list)
