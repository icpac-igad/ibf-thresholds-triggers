
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

from utils import spi3_prod_name_creator
from utils import spi4_prod_name_creator
from utils import spi6_prod_name_creator




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


def seas5_grib_processor(input_path_location):
    """
    the input grib file download from cds
    https://cds.climate.copernicus.eu/cdsapp#!/dataset/seasonal-monthly-single-levels?tab=form
    but the different version are needed to avoid the missing forecast months, such as for 2022. 05 month
    
    This function combines all the grib files into one single dataset
    """
    p1_input_data=f'{input_path_location}seas5_v51_1981-2016_m26_lt6_tp.grib'
    ds_p1 = xr.open_dataset(p1_input_data, engine='cfgrib', backend_kwargs=dict(time_dims=('forecastMonth', 'time')))
    p2_input_data=f'{input_path_location}seas5_v5_2017-2021_m51_lt6_tp.grib'
    ds_p2 = xr.open_dataset(p2_input_data, engine='cfgrib', backend_kwargs=dict(time_dims=('forecastMonth', 'time')))
    p3_input_data=f'{input_path_location}seas5_v51_2022_m51_lt6_tp.grib'
    ds_p3 = xr.open_dataset(p3_input_data, engine='cfgrib', backend_kwargs=dict(time_dims=('forecastMonth', 'time')))
    p4_input_data=f'{input_path_location}seas5_v51_2023_m51_lt6_tp.grib'
    ds_p4 = xr.open_dataset(p4_input_data, engine='cfgrib', backend_kwargs=dict(time_dims=('forecastMonth', 'time')))
    p5_input_data=f'{input_path_location}seas5_v5_2022_m51_lt6_tp_may.grib'
    ds_p5 = xr.open_dataset(p5_input_data, engine='cfgrib', backend_kwargs=dict(time_dims=('forecastMonth', 'time')))
    ds_p5a=ds_p5.sel(time='2022-05-01T00:00:00.000000000')
    ds_2022=xr.concat([ds_p3,ds_p5a],dim='time')
    ds_p=xr.concat([ds_p1,ds_p2,ds_2022,ds_p4],dim='time')
    return ds_p




def seas5_regridder(grib_array,output_path_location):
    """
    

    Parameters
    ----------
    grib_array : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
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
        ds2.to_netcdf(f'{output_path_location}kmj_25km_lt_month_{fm}.nc')
    
   
   
def seas5_tpm_creator(output_path_location):
    """
    

    Returns
    -------
    None.

    """
    for mnth in [1,2,3,4,5,6]:
        db=xr.open_dataset(f'{output_path_location}kmj_25km_lt_month_{mnth}.nc')
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
        db.to_netcdf(f'{output_path_location}tp_kmj_25km_lt_month_{mnth}.nc')
        
        
        
def lead_month_wise_df_create(output_path_location):
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
            db_m1=xr.open_dataset(f'{output_path_location}tp_kmj_25km_lt_month_1.nc')
            db1_m1=db_m1.sel(time=jd)
            db_m2=xr.open_dataset(f'{output_path_location}tp_kmj_25km_lt_month_2.nc')
            db1_m2=db_m2.sel(time=jd)
            db_m3=xr.open_dataset(f'{output_path_location}tp_kmj_25km_lt_month_3.nc')
            db1_m3=db_m3.sel(time=jd)
            db_m4=xr.open_dataset(f'{output_path_location}tp_kmj_25km_lt_month_4.nc')
            db1_m4=db_m4.sel(time=jd)
            db_m5=xr.open_dataset(f'{output_path_location}tp_kmj_25km_lt_month_5.nc')
            db1_m5=db_m5.sel(time=jd)
            db_m6=xr.open_dataset(f'{output_path_location}tp_kmj_25km_lt_month_6.nc')
            db1_m6=db_m6.sel(time=jd)
            db1_m=xr.concat([db1_m1,db1_m2,db1_m3,db1_m4,db1_m5,db1_m6],dim='time')
            cont_mdb.append(db1_m)
        erf1=xr.concat(cont_mdb,dim='time')
        erf1 = erf1.assign_coords(time=('time',erf1.valid_time.values))
        month_dateformat=datetime.strptime(str(mnthn) , '%m')
        month_str=month_dateformat.strftime('%b').lower()
        erf1.to_netcdf(f'{output_path_location}{month_str}_tp_kmj_25km_6m_fcstd_1981.nc')
    
    
    
    
    
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
        


def three_months_spi_creator(output_path_location):
    """
    MAM SPI only using the lead time from months
    'jan','feb','nov','dec'

    Parameters
    ----------
    output_path_location : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    sdb_list=['jan_tp_kmj_25km_6m_fcstd_1981.nc',
              'feb_tp_kmj_25km_6m_fcstd_1981.nc',
              'nov_tp_kmj_25km_6m_fcstd_1981.nc',
              'dec_tp_kmj_25km_6m_fcstd_1981.nc']
    spi_prod='spi3'
    for sdbl in sdb_list:
        sdb=xr.open_dataset(f'{output_path_location}{sdbl}')
        member_list=np.arange(0,51,1)    
        for meml in member_list:
            sdb1=sdb.sel(number=meml)
            sc = sdb1.stack(grid_cells=('lat', 'lon',))
            spi_ds = sc.groupby('grid_cells').apply(
                spi_wrapper,
                precip_var='tprate',
                scale=3,
                distribution=Distribution.gamma,
                data_start_year=1981,
                calibration_year_initial=1981,
                calibration_year_final=2018,
                periodicity=Periodicity.monthly).unstack('grid_cells')
            spi_ds1=spi_ds.to_dataset(name='tprate')
            sdbl1=sdbl.split('.')[0]
            ens_output_path0=f'{output_path_location}{spi_prod}'
            foldercreator(ens_output_path0)
            ens_output_path=f'{output_path_location}{spi_prod}/{sdbl1}'
            foldercreator(ens_output_path)
            #spi_ds2.to_csv(f'{output_path}month6_{meml}.csv')
            spi_ds1.to_netcdf(f'{ens_output_path}/{meml}.nc')
            
            
            
def four_months_spi_creator(output_path_location):
    """
    MAM SPI only using the lead time from months
    'mar','apr','may'

    Parameters
    ----------
    output_path_location : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    sdb_list=['mar_tp_kmj_25km_6m_fcstd_1981.nc',
              'apr_tp_kmj_25km_6m_fcstd_1981.nc',
              'may_tp_kmj_25km_6m_fcstd_1981.nc']
    spi_prod='spi4'
    for sdbl in sdb_list:
        sdb=xr.open_dataset(f'{output_path_location}{sdbl}')
        member_list=np.arange(0,51,1)    
        for meml in member_list:
            sdb1=sdb.sel(number=meml)
            sc = sdb1.stack(grid_cells=('lat', 'lon',))
            spi_ds = sc.groupby('grid_cells').apply(
                spi_wrapper,
                precip_var='tprate',
                scale=4,
                distribution=Distribution.gamma,
                data_start_year=1981,
                calibration_year_initial=1981,
                calibration_year_final=2018,
                periodicity=Periodicity.monthly).unstack('grid_cells')
            spi_ds1=spi_ds.to_dataset(name='tprate')
            sdbl1=sdbl.split('.')[0]
            ens_output_path0=f'{output_path_location}{spi_prod}'
            foldercreator(ens_output_path0)
            ens_output_path=f'{output_path_location}{spi_prod}/{sdbl1}'
            foldercreator(ens_output_path)
            #spi_ds2.to_csv(f'{output_path}month6_{meml}.csv')
            spi_ds1.to_netcdf(f'{ens_output_path}/{meml}.nc')
            
            
def six_months_spi_creator(output_path_location):
    """
    MAM SPI only using the lead time from months
    'feb','mar'

    Parameters
    ----------
    output_path_location : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    sdb_list=['feb_tp_kmj_25km_6m_fcstd_1981.nc',
              'mar_tp_kmj_25km_6m_fcstd_1981.nc']
    spi_prod='spi6'
    for sdbl in sdb_list:
        sdb=xr.open_dataset(f'{output_path_location}{sdbl}')
        member_list=np.arange(0,51,1)    
        for meml in member_list:
            sdb1=sdb.sel(number=meml)
            sc = sdb1.stack(grid_cells=('lat', 'lon',))
            spi_ds = sc.groupby('grid_cells').apply(
                spi_wrapper,
                precip_var='tprate',
                scale=6,
                distribution=Distribution.gamma,
                data_start_year=1981,
                calibration_year_initial=1981,
                calibration_year_final=2018,
                periodicity=Periodicity.monthly).unstack('grid_cells')
            spi_ds1=spi_ds.to_dataset(name='tprate')
            sdbl1=sdbl.split('.')[0]
            ens_output_path0=f'{output_path_location}{spi_prod}'
            foldercreator(ens_output_path0)
            ens_output_path=f'{output_path_location}{spi_prod}/{sdbl1}'
            foldercreator(ens_output_path)
            #spi_ds2.to_csv(f'{output_path}month6_{meml}.csv')
            spi_ds1.to_netcdf(f'{ens_output_path}/{meml}.nc')
            

def chrips_data_regridder():
    db=xr.open_dataset('data/chirps-v2.0.monthly.nc')
    ea_db = db.sel(latitude=slice(0.0,5.0), longitude=slice(31.0,36.0))
    local_path_nc='data/kmj_chirps-v2.0.monthly.nc'
    ea_db.to_netcdf(local_path_nc)
    ds=xr.open_dataset(local_path_nc)
    ds1=ds.rename({'longitude':'lon','latitude':'lat'})
    dr = ds1["precip"] 
    ds_out = xr.Dataset(
          {"lat": (["lat"], np.arange(0.0, 5.0, 0.25), {"units": "degrees_north"}),
          "lon": (["lon"], np.arange(31.0, 36.0, 0.25), {"units": "degrees_east"}),})
    regridder = xe.Regridder(ds1, ds_out, "bilinear")
    regridder  # print basic regridder information.
    dr_out = regridder(dr, keep_attrs=True)
    ds2=dr_out.to_dataset()
    ds2.to_netcdf('data/kmj_km25_chirps-v2.0.monthly.nc')
    
    
def chrips_spi_mam_creator():
    cdb=xr.open_dataset('data/kmj_km25_chirps-v2.0.monthly.nc')
    cdb_sc = cdb.stack(grid_cells=('lat', 'lon',))
    spi_cdb = cdb_sc.groupby('grid_cells').apply(
        spi_wrapper,
        precip_var='precip',
        scale=3,
        distribution=Distribution.gamma,
        data_start_year=1981,
        calibration_year_initial=1981,
        calibration_year_final=2018,
        periodicity=Periodicity.monthly,
    ).unstack('grid_cells')
    spi_cdb1=spi_cdb.to_dataset(name='spi')
    spi_prod_list=spi3_prod_name_creator(spi_cdb1)
    spi_cdb2 = spi_cdb1.assign_coords(spi_prod=('time',spi_prod_list))
    spi_cdb3=spi_cdb2.where(spi_cdb2.spi_prod=='MAM', drop=True)
    spi_cdb3.to_netcdf('output/obs/mam_kmj_km25_chirps-v2.0.monthly.nc')
    
    
def chrips_spi_jjas_creator():
    cdb=xr.open_dataset('data/kmj_km25_chirps-v2.0.monthly.nc')
    cdb_sc = cdb.stack(grid_cells=('lat', 'lon',))
    spi_cdb = cdb_sc.groupby('grid_cells').apply(
        spi_wrapper,
        precip_var='precip',
        scale=4,
        distribution=Distribution.gamma,
        data_start_year=1981,
        calibration_year_initial=1981,
        calibration_year_final=2018,
        periodicity=Periodicity.monthly,
    ).unstack('grid_cells')
    spi_cdb1=spi_cdb.to_dataset(name='spi')
    spi_prod_list=spi4_prod_name_creator(spi_cdb1)
    spi_cdb2 = spi_cdb1.assign_coords(spi_prod=('time',spi_prod_list))
    spi_cdb3=spi_cdb2.where(spi_cdb2.spi_prod=='JJAS', drop=True)
    spi_cdb3.to_netcdf('output/obs/jjas_kmj_km25_chirps-v2.0.monthly.nc')
    
    
    
def chrips_spi_mamjja_creator():
    cdb=xr.open_dataset('data/kmj_km25_chirps-v2.0.monthly.nc')
    cdb_sc = cdb.stack(grid_cells=('lat', 'lon',))
    spi_cdb = cdb_sc.groupby('grid_cells').apply(
        spi_wrapper,
        precip_var='precip',
        scale=6,
        distribution=Distribution.gamma,
        data_start_year=1981,
        calibration_year_initial=1981,
        calibration_year_final=2018,
        periodicity=Periodicity.monthly,
    ).unstack('grid_cells')
    spi_cdb1=spi_cdb.to_dataset(name='spi')
    spi_prod_list=spi4_prod_name_creator(spi_cdb1)
    spi_cdb2 = spi_cdb1.assign_coords(spi_prod=('time',spi_prod_list))
    spi_cdb3=spi_cdb2.where(spi_cdb2.spi_prod=='MAMJJA', drop=True)
    print(spi_cdb2)
    spi_cdb3.to_netcdf('output/obs/mamjja_kmj_km25_chirps-v2.0.monthly.nc')
    
    
def chrips_spi_amjjas_creator():
    cdb=xr.open_dataset('data/kmj_km25_chirps-v2.0.monthly.nc')
    cdb_sc = cdb.stack(grid_cells=('lat', 'lon',))
    spi_cdb = cdb_sc.groupby('grid_cells').apply(
        spi_wrapper,
        precip_var='precip',
        scale=6,
        distribution=Distribution.gamma,
        data_start_year=1981,
        calibration_year_initial=1981,
        calibration_year_final=2018,
        periodicity=Periodicity.monthly,
    ).unstack('grid_cells')
    spi_cdb1=spi_cdb.to_dataset(name='spi')
    spi_prod_list=spi6_prod_name_creator(spi_cdb1)
    spi_cdb2 = spi_cdb1.assign_coords(spi_prod=('time',spi_prod_list))
    spi_cdb3=spi_cdb2.where(spi_cdb2.spi_prod=='AMJJAS', drop=True)
    spi_cdb3.to_netcdf('output/obs/amjjas_kmj_km25_chirps-v2.0.monthly.nc')
    
def chrips_spi_amjjas_creator_a():
    cdb=xr.open_dataset('data/kmj_km25_chirps-v2.0.monthly.nc')
    cdb_sc = cdb.stack(grid_cells=('lat', 'lon',))
    spi_cdb = cdb_sc.groupby('grid_cells').apply(
        spi_wrapper,
        precip_var='precip',
        scale=6,
        distribution=Distribution.gamma,
        data_start_year=1981,
        calibration_year_initial=1981,
        calibration_year_final=2018,
        periodicity=Periodicity.monthly,
    ).unstack('grid_cells')
    spi_cdb1=spi_cdb.to_dataset(name='spi')
    spi_prod_list=spi6_prod_name_creator(spi_cdb1)
    spi_cdb2 = spi_cdb1.assign_coords(spi_prod=('time',spi_prod_list))
    spi_cdb3=spi_cdb2.where(spi_cdb2.spi_prod=='MAMJJA', drop=True)
    spi_cdb3.to_netcdf('output/obs/mamjja_kmj_km25_chirps-v2.0.monthly.nc')
    
    
    
    