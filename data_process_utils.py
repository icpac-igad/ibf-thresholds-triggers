
import os
import ntpath

import xarray as xr
import cfgrib
from datetime import datetime

import pandas as pd
#based on https://ecmwf-projects.github.io/copernicus-training-c3s/sf-anomalies.html
from dateutil.relativedelta import relativedelta
from calendar import monthrange
import xesmf as xe
import numpy as np

from typing import Dict
from climate_indices.indices import spi, Distribution
from climate_indices.compute import Periodicity
import numpy as np
import pandas as pd
import requests
import xarray as xr




def seas5_grib_processor(path_location):
    """
    the input grib file download from cds
    https://cds.climate.copernicus.eu/cdsapp#!/dataset/seasonal-monthly-single-levels?tab=form
    but the different version are needed to avoid the missing forecast months, such as for 2022. 05 month
    
    This function combines all the grib files into one single dataset
    """
    p1_input_data=f'{path_location}seas5_v51_1981-2016_m26_lt6_tp.grib'
    ds_p1 = xr.open_dataset(p1_input_data, engine='cfgrib', backend_kwargs=dict(time_dims=('forecastMonth', 'time')))
    p2_input_data=f'{path_location}seas5_v5_2017-2021_m51_lt6_tp.grib'
    ds_p2 = xr.open_dataset(p2_input_data, engine='cfgrib', backend_kwargs=dict(time_dims=('forecastMonth', 'time')))
    p3_input_data=f'{path_location}seas5_v51_2022_m51_lt6_tp.grib'
    ds_p3 = xr.open_dataset(p3_input_data, engine='cfgrib', backend_kwargs=dict(time_dims=('forecastMonth', 'time')))
    p4_input_data=f'{path_location}seas5_v51_2023_m51_lt6_tp.grib'
    ds_p4 = xr.open_dataset(p4_input_data, engine='cfgrib', backend_kwargs=dict(time_dims=('forecastMonth', 'time')))
    p5_input_data=f'{path_location}seas5_v5_2022_m51_lt6_tp_may.grib'
    ds_p5 = xr.open_dataset(p5_input_data, engine='cfgrib', backend_kwargs=dict(time_dims=('forecastMonth', 'time')))
    ds_p5a=ds_p5.sel(time='2022-05-01T00:00:00.000000000')
    ds_2022=xr.concat([ds_p3,ds_p5a],dim='time')
    ds_p=xr.concat([ds_p1,ds_p2,ds_2022,ds_p4],dim='time')
    return ds_p




def seas5_regridder(grib_array):
    """
    

    Parameters
    ----------
    grib_array : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    dates = pd.date_range('1981-01-01', '2023-05-01', freq='MS')
    for fm in [1,2,3,4,5,6]:
        ds_p_m1=grib_array.sel(forecastMonth=fm)
        ds_out = xr.Dataset(
              {"lat": (["lat"], np.arange(0.0, 5.0, 0.25), {"units": "degrees_north"}),
              "lon": (["lon"], np.arange(31.0, 36.0, 0.25), {"units": "degrees_east"}),})
        gd2=ds_p_m1.rename({'longitude':'lon','latitude':'lat'})
        agd = gd2["tprate"]
        regridder = xe.Regridder(gd2, ds_out, "bilinear")
        dr_out = regridder(agd, keep_attrs=True)
        ds2=dr_out.to_dataset()
        #monthname=mnl.lower().split('.')[0]
        ds2.to_netcdf(f'kmj_25km_lt_month_{fm}.nc')
    
   
   
def seas5_tpm_creator():
    """
    

    Returns
    -------
    None.

    """
    for mnth in [1,2,3,4,5,6]:
        db=xr.open_dataset(f'kmj_25km_lt_month_{mnth}.nc')
        aa=pd.to_datetime(db.time.values)
        cont_vdt=[]
        for itu in aa:
            vd_t=itu+relativedelta(months=mnth)
            cont_vdt.append(vd_t)
        db = db.assign_coords(valid_time=('time',cont_vdt))
        numdays = [monthrange(dd.year,dd.month)[1] for dd in cont_vdt]
        db = db.assign_coords(numdays=('time',numdays))
        db = db * db.numdays * 24 * 60 * 60 * 1000
        db.attrs['units'] = 'mm/month'
        db.attrs['long_name'] = 'Total precipitation' 
        db.to_netcdf(f'tp_kmj_25km_lt_month_{mnth}.nc')
        
        
        
def lead_month_wise_df_create():
    """
    

    Returns
    -------
    None.

    """
    dates = pd.date_range('1981-01-01', '2023-10-01', freq='MS')
    months_needed=[1,2,3,4,5,11,12]
    for mnthn in months_needed:
        mnth_dates=dates[dates.month.isin([mnthn])]
        jd1=mnth_dates.strftime('%Y-%m-%dT00:00:00.000000000')
        cont_mdb=[]
        for jd in jd1:
            db_m1=xr.open_dataset('tp_kmj_25km_lt_month_1.nc')
            db1_m1=db_m1.sel(time=jd)
            db_m2=xr.open_dataset('tp_kmj_25km_lt_month_2.nc')
            db1_m2=db_m2.sel(time=jd)
            db_m3=xr.open_dataset('tp_kmj_25km_lt_month_3.nc')
            db1_m3=db_m3.sel(time=jd)
            db_m4=xr.open_dataset('tp_kmj_25km_lt_month_4.nc')
            db1_m4=db_m4.sel(time=jd)
            db_m5=xr.open_dataset('tp_kmj_25km_lt_month_5.nc')
            db1_m5=db_m5.sel(time=jd)
            db_m6=xr.open_dataset('tp_kmj_25km_lt_month_6.nc')
            db1_m6=db_m6.sel(time=jd)
            db1_m=xr.concat([db1_m1,db1_m2,db1_m3,db1_m4,db1_m5,db1_m6],dim='time')
            cont_mdb.append(db1_m)
        erf1=xr.concat(cont_mdb,dim='time')
        erf1 = erf1.assign_coords(time=('time',erf1.valid_time.values))
        month_dateformat=datetime.strptime('6' , '%m')
        month_str=month_dateformat.strftime('%b').lower()
        erf1.to_netcdf(f'{month_str}_tp_kmj_25km_6m_forecasted_1981.nc')
    
    
    
    
    
def spi_wrapper(
    obj: xr.DataArray,
    precip_var: str,
    scale: int,
    distribution: Distribution,
    data_start_year: int,
    calibration_year_initial: int,
    calibration_year_final: int,
    periodicity: Periodicity,
    fitting_params: Dict = None,
) -> xr.DataArray:
    # compute SPI for this timeseries
    spi_data = spi(
        values=obj[precip_var].to_numpy(), #TODO find why we need to use the variable name rather than already using the variables's DataArray (i.e. why is obj a Dataset?)
        scale=scale,
        distribution=distribution,
        data_start_year=data_start_year,
        calibration_year_initial=calibration_year_initial,
        calibration_year_final=calibration_year_final,
        periodicity=periodicity,
        fitting_params=fitting_params,
    )
    #TODO for some reason this is necessary for the nClimGrid low-resolution example NetCDFs
    #TODO find out why
    spi_data = spi_data.flatten()
    #TODO for some reason this is necessary for the NCO-modified nClimGrid normal-resolution example NetCDFs
    #TODO find out why
    #spi_data = spi_data.reshape(spi_data.size, 1)
    # create the return DataArray (copy of input object's geospatial dims/coords plus SPI data)
    da_spi = xr.DataArray(
        dims   = obj[precip_var].dims,
        coords = obj[precip_var].coords,
        attrs  = {
            'description': 'SPI computed by the climate_indices Python package',
            'references': 'https://github.com/monocongo/climate_indices',
            'valid_min': -3.09, # this should mirror climate_indices.indices._FITTED_INDEX_VALID_MIN
            'valid_max':  3.09, # this should mirror climate_indices.indices._FITTED_INDEX_VALID_MAX
        },
        data = spi_data,
    )
    return da_spi
        


def path_leaf(path):
    """
    Get the name of a file without any extension from given path

    Parameters
    ----------
    path : file full path with extension
    
    Returns
    -------
    str
       filename in the path without extension

    """
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def foldercreator(path):
   """
    creates a folder

    Parameters
    ----------
    path : folder path
            
    Returns
    -------
    creates a folder
    """
   if not os.path.exists(path):
        os.makedirs(path)