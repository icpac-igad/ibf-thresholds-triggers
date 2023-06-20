
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

#from utils import spi3_prod_name_creator
#from utils import spi4_prod_name_creator
#from utils import spi6_prod_name_creator




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

#%% seas5 data processing utils

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
    
    
#%% spi generate utils            
    
    
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
           
#%% CHRIPS data processor utils 

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
    
    
#%% spi probablity ncfile creator

def excprob(X, X_thr, ignore_nan=False):
    """
    For a given forecast ensemble field, compute exceedance probabilities
    for the given intensity thresholds.
    Parameters
    ----------
    X: array_like
        Array of shape (k, m, n, ...) containing an k-member ensemble of
        forecasts with shape (m, n, ...).
    X_thr: float or a sequence of floats
        Intensity threshold(s) for which the exceedance probabilities are
        computed.
    ignore_nan: bool
        If True, ignore nan values.
    Returns
    -------
    out: ndarray
        Array of shape (len(X_thr), m, n) containing the exceedance
        probabilities for the given intensity thresholds.
        If len(X_thr)=1, the first dimension is dropped.
        
    https://github.com/pySTEPS/pysteps/blob/master/pysteps/postprocessing/ensemblestats.py
    """
    #  Checks
    X = np.asanyarray(X)
    X_ndim = X.ndim

    if X_ndim < 3:
        raise Exception(
            f"Number of dimensions of X should be 3 or more. It was: {X_ndim}"
        )

    P = []

    if np.isscalar(X_thr):
        X_thr = [X_thr]
        scalar_thr = True
    else:
        scalar_thr = False

    for x in X_thr:
        X_ = X.copy()
        #original
        #X_[X >= x] = 1.0
        #X_[X < x] = 0.0
        #changes to make less than threnshold
        #based on MOZ paper method 
        X_[X <= x] = 1.0
        X_[X > x] = 0.0
        X_[~np.isfinite(X)] = np.nan

        if ignore_nan:
            P.append(np.nanmean(X_, axis=0))
        else:
            P.append(np.mean(X_, axis=0))

    if not scalar_thr:
        return np.stack(P)
    else:
        return P[0]




def ens_mem_combiner(input_path):
    """
    

    Parameters
    ----------
    input_path : TYPE
        DESCRIPTION.

    Returns
    -------
    ds_ens : TYPE
        DESCRIPTION.

    """
    member_list=np.arange(0,51,1)
    mem_data_arrays = []
    for mem in member_list:
        mem_file_path=f'{input_path}{mem}.nc'
        ds=xr.open_dataset(mem_file_path)
        mem_data_arrays.append(ds)
        #ds=[]
        # concatenate list of data arrays along new dimension into xarray dataset
    ds_ens = xr.concat(mem_data_arrays, dim='ens_mem')
    return ds_ens
    
    
    
def spi3_prod_name_creator(ds_ens):
    """
    

    Parameters
    ----------
    ds_ens : TYPE
        DESCRIPTION.

    Returns
    -------
    spi_prod_list : TYPE
        DESCRIPTION.

    """
    db=pd.DataFrame()
    db['dt']=ds_ens['time'].values
    db['month']=db['dt'].dt.strftime('%b').astype(str).str[0]
    db['year']=db['dt'].dt.strftime('%Y')
    db['spi_prod'] = db.groupby('year')['month'].shift(2)+db.groupby('year')['month'].shift(1) + db.groupby('year')['month'].shift(0)
    spi_prod_list=db['spi_prod'].tolist()
    return spi_prod_list

def spi4_prod_name_creator(ds_ens):
    """
    

    Parameters
    ----------
    ds_ens : TYPE
        DESCRIPTION.

    Returns
    -------
    spi_prod_list : TYPE
        DESCRIPTION.

    """
    db=pd.DataFrame()
    db['dt']=ds_ens['time'].values
    db['month']=db['dt'].dt.strftime('%b').astype(str).str[0]
    db['year']=db['dt'].dt.strftime('%Y')
    db['spi_prod'] = db.groupby('year')['month'].shift(3)+db.groupby('year')['month'].shift(2)+db.groupby('year')['month'].shift(1) + db.groupby('year')['month'].shift(0)
    spi_prod_list=db['spi_prod'].tolist()
    return spi_prod_list



def spi6_prod_name_creator(ds_ens):
    """
    

    Parameters
    ----------
    ds_ens : TYPE
        DESCRIPTION.

    Returns
    -------
    spi_prod_list : TYPE
        DESCRIPTION.

    """
    db=pd.DataFrame()
    db['dt']=ds_ens['time'].values
    db['month']=db['dt'].dt.strftime('%b').astype(str).str[0]
    db['year']=db['dt'].dt.strftime('%Y')
    db['spi_prod'] = db.groupby('year')['month'].shift(5)+db.groupby('year')['month'].shift(4)+db.groupby('year')['month'].shift(3)+db.groupby('year')['month'].shift(2)+db.groupby('year')['month'].shift(1) + db.groupby('year')['month'].shift(0)
    spi_prod_list=db['spi_prod'].tolist()
    return spi_prod_list

    
    

def ds_emprical_prbablity_creator(spi_ds_ens,month,threshold):
    """
    

    Parameters
    ----------
    spi_ds_ens : TYPE
        DESCRIPTION.
    month : TYPE
        DESCRIPTION.
    threshold : TYPE
        DESCRIPTION.

    Returns
    -------
    cont_data_xds_mild1 : TYPE
        DESCRIPTION.
    cont_data_xds_moderate1 : TYPE
        DESCRIPTION.
    cont_data_xds_severe1 : TYPE
        DESCRIPTION.

    """
    start_dt=spi_ds_ens['valid_time'].values[0]
    end_dt=spi_ds_ens['valid_time'].values[-1]
    dates = pd.date_range(start_dt, end_dt, freq='MS')
    mam_dates=dates[dates.month.isin([month])]
    cont_data_xds_mild=[]
    cont_data_xds_moderate=[]
    cont_data_xds_severe=[]
    for dat in mam_dates:
        jd1=dat.strftime('%Y-%m-%dT00:00:00.000000000')
        year=dat.strftime('%Y')
        spi_ds_ens_kmj_st=spi_ds_ens.sel(time=jd1)
        spi_array=spi_ds_ens_kmj_st.to_array()
        pro_ds=excprob(spi_array, threshold, ignore_nan=False)
        data_xr = xr.DataArray(pro_ds,coords={'prob':['low','mid','high'],'ens_mem':spi_ds_ens.ens_mem.values,
            'lat': spi_ds_ens.lat.values,'lon': spi_ds_ens.lon.values,'time':year}, 
        dims=["prob", "ens_mem", "lat","lon"])
        data_xds=data_xr.to_dataset(name='prob_exced')
        data_xds_mild=data_xds.sel(prob='low')
        data_xds_mild1 = data_xds_mild.where(data_xds_mild.prob_exced >=1).sum("ens_mem")/data_xds_mild.where(data_xds_mild.prob_exced.notnull()).count("ens_mem")
        data_xds_moderate=data_xds.sel(prob='mid')
        data_xds_moderate1 = data_xds_moderate.where(data_xds_moderate.prob_exced >=1).sum("ens_mem")/data_xds_moderate.where(data_xds_moderate.prob_exced.notnull()).count("ens_mem")
        data_xds_severe=data_xds.sel(prob='high')
        data_xds_severe1 = data_xds_severe.where(data_xds_severe.prob_exced >=1).sum("ens_mem")/data_xds_severe.where(data_xds_severe.prob_exced.notnull()).count("ens_mem")
        #year_data_xds=xr.concat([data_xds_mild1,data_xds_moderate1,data_xds_severe1],dim='time')
        cont_data_xds_mild.append(data_xds_mild1)
        cont_data_xds_moderate.append(data_xds_moderate1)
        cont_data_xds_severe.append(data_xds_severe1)
    cont_data_xds_mild1=xr.concat(cont_data_xds_mild,dim='time')
    cont_data_xds_moderate1=xr.concat(cont_data_xds_moderate,dim='time')
    cont_data_xds_severe1=xr.concat(cont_data_xds_severe,dim='time')
    return cont_data_xds_mild1, cont_data_xds_moderate1, cont_data_xds_severe1
    
    

def spi3_prob_ncfile_creator(output_path):
    spi_prod='mam'
    lt_month=['nov','dec','jan','feb']
    threshold=[-0.03, -0.56,-0.99]
    spi_month=5
    spi_name='MAM'
    for ltm in lt_month:
        input_path=f'output/spi3/{ltm}_tp_kmj_25km_6m_fcstd_1981/'
        ds_ens=ens_mem_combiner(input_path)
        print(ds_ens)
        spi_prod_list=spi3_prod_name_creator(ds_ens)
        print(spi_prod_list)
        ds_ens1 = ds_ens.assign_coords(spi_prod=('time',spi_prod_list))
        spi_ds_ens=ds_ens1.where(ds_ens1.spi_prod==spi_name, drop=True)
        ds_mild, ds_mod, ds_sev=ds_emprical_prbablity_creator(spi_ds_ens,spi_month,threshold)
        ds_mild.to_netcdf(f'{output_path}{spi_prod}_{ltm}_low.nc')
        ds_mod.to_netcdf(f'{output_path}{spi_prod}_{ltm}_mid.nc')
        ds_sev.to_netcdf(f'{output_path}{spi_prod}_{ltm}_high.nc')
        
        
        
def spi4_prob_ncfile_creator(output_path):
    spi_prod='jjas'
    lt_month=['mar','apr','may']
    threshold=[-0.01, -0.41,-0.99]
    spi_month=9
    spi_name='JJAS'
    for ltm in lt_month:
        input_path=f'output/spi4/{ltm}_tp_kmj_25km_6m_fcstd_1981/'
        ds_ens=ens_mem_combiner(input_path)
        print(ds_ens)
        spi_prod_list=spi4_prod_name_creator(ds_ens)
        print(spi_prod_list)
        ds_ens1 = ds_ens.assign_coords(spi_prod=('time',spi_prod_list))
        spi_ds_ens=ds_ens1.where(ds_ens1.spi_prod==spi_name, drop=True)
        ds_mild, ds_mod, ds_sev=ds_emprical_prbablity_creator(spi_ds_ens,spi_month,threshold)
        ds_mild.to_netcdf(f'{output_path}{spi_prod}_{ltm}_low.nc')
        ds_mod.to_netcdf(f'{output_path}{spi_prod}_{ltm}_mid.nc')
        ds_sev.to_netcdf(f'{output_path}{spi_prod}_{ltm}_high.nc')
        
        
def spi6_prob_ncfile_creator_a(output_path):
    spi_prod='mamjja'
    lt_month=['feb']
    threshold=[-0.02, -0.38,-1.01]
    spi_month=8
    spi_name='MAMJJA'
    for ltm in lt_month:
        input_path=f'output/spi6/{ltm}_tp_kmj_25km_6m_fcstd_1981/'
        ds_ens=ens_mem_combiner(input_path)
        print(ds_ens)
        spi_prod_list=spi6_prod_name_creator(ds_ens)
        print(spi_prod_list)
        ds_ens1 = ds_ens.assign_coords(spi_prod=('time',spi_prod_list))
        spi_ds_ens=ds_ens1.where(ds_ens1.spi_prod==spi_name, drop=True)
        ds_mild, ds_mod, ds_sev=ds_emprical_prbablity_creator(spi_ds_ens,spi_month,threshold)
        ds_mild.to_netcdf(f'{output_path}{spi_prod}_{ltm}_low.nc')
        ds_mod.to_netcdf(f'{output_path}{spi_prod}_{ltm}_mid.nc')
        ds_sev.to_netcdf(f'{output_path}{spi_prod}_{ltm}_high.nc')
        
        
def spi6_prob_ncfile_creator_b(output_path):
    spi_prod='amjjas'
    lt_month=['mar']
    threshold=[-0.02, -0.38,-1.01]
    spi_month=9
    spi_name='AMJJAS'
    for ltm in lt_month:
        input_path=f'output/spi6/{ltm}_tp_kmj_25km_6m_fcstd_1981/'
        ds_ens=ens_mem_combiner(input_path)
        print(ds_ens)
        spi_prod_list=spi6_prod_name_creator(ds_ens)
        print(spi_prod_list)
        ds_ens1 = ds_ens.assign_coords(spi_prod=('time',spi_prod_list))
        spi_ds_ens=ds_ens1.where(ds_ens1.spi_prod==spi_name, drop=True)
        ds_mild, ds_mod, ds_sev=ds_emprical_prbablity_creator(spi_ds_ens,spi_month,threshold)
        ds_mild.to_netcdf(f'{output_path}{spi_prod}_{ltm}_low.nc')
        ds_mod.to_netcdf(f'{output_path}{spi_prod}_{ltm}_mid.nc')
        ds_sev.to_netcdf(f'{output_path}{spi_prod}_{ltm}_high.nc')
        

#%% spi mean ncfile creator

def ds_spi_mean_emprical_prbablity_creator(spi_ds_ens,month,threshold):
    """
    

    Parameters
    ----------
    spi_ds_ens : TYPE
        DESCRIPTION.
    month : TYPE
        DESCRIPTION.
    threshold : TYPE
        DESCRIPTION.

    Returns
    -------
    cont_data_xds_mild1 : TYPE
        DESCRIPTION.
    cont_data_xds_moderate1 : TYPE
        DESCRIPTION.
    cont_data_xds_severe1 : TYPE
        DESCRIPTION.

    """
    start_dt=spi_ds_ens['valid_time'].values[0]
    end_dt=spi_ds_ens['valid_time'].values[-1]
    dates = pd.date_range(start_dt, end_dt, freq='MS')
    mam_dates=dates[dates.month.isin([month])]
    cont_data_xds_low=[]
    cont_data_xds_mid=[]
    cont_data_xds_high=[]
    for dat in mam_dates:
        jd1=dat.strftime('%Y-%m-%dT00:00:00.000000000')
        year=dat.strftime('%Y')
        spi_ds_ens_kmj_st=spi_ds_ens.sel(time=jd1)
        spi_array=spi_ds_ens_kmj_st.to_array()
        pro_ds=excprob(spi_array, threshold, ignore_nan=False)
        data_xr = xr.DataArray(pro_ds,coords={'prob':['low','mid','high'],'ens_mem':spi_ds_ens.ens_mem.values,
            'lat': spi_ds_ens.lat.values,'lon': spi_ds_ens.lon.values,'time':year}, 
        dims=["prob", "ens_mem", "lat","lon"])
        data_xds=data_xr.to_dataset(name='prob_exced')
        data_xds_low=data_xds.sel(prob='low')
        data_xds_low1 = data_xds_low.where(data_xds_low.prob_exced >=1)
        data_xds_low2=data_xds_low1.dropna(dim="ens_mem", how="all")
        spi_ds_low=spi_ds_ens_kmj_st.sel(ens_mem=data_xds_low2['ens_mem'].values)
        spi_ds_low1=spi_ds_low['tprate'].mean(dim='ens_mem')
        spi_ds_low2=spi_ds_low1.to_dataset(name='spi_ema')
        ####################
        data_xds_mid=data_xds.sel(prob='mid')
        data_xds_mid1 = data_xds_mid.where(data_xds_mid.prob_exced >=1)
        data_xds_mid2=data_xds_mid1.dropna(dim="ens_mem", how="all")
        spi_ds_mid=spi_ds_ens_kmj_st.sel(ens_mem=data_xds_mid2['ens_mem'].values)
        spi_ds_mid1=spi_ds_mid['tprate'].mean(dim='ens_mem')
        spi_ds_mid2=spi_ds_mid1.to_dataset(name='spi_ema')
        ####################
        data_xds_high=data_xds.sel(prob='high')
        data_xds_high1 = data_xds_high.where(data_xds_high.prob_exced >=1)
        data_xds_high2=data_xds_high1.dropna(dim="ens_mem", how="all")
        spi_ds_high=spi_ds_ens_kmj_st.sel(ens_mem=data_xds_high2['ens_mem'].values)
        spi_ds_high1=spi_ds_high['tprate'].mean(dim='ens_mem')
        spi_ds_high2=spi_ds_high1.to_dataset(name='spi_ema')
        cont_data_xds_low.append(spi_ds_low2)
        cont_data_xds_mid.append(spi_ds_mid2)
        cont_data_xds_high.append(spi_ds_high2)
    cont_data_xds_low1=xr.concat(cont_data_xds_low,dim='time')
    cont_data_xds_mid1=xr.concat(cont_data_xds_mid,dim='time')
    cont_data_xds_high1=xr.concat(cont_data_xds_high,dim='time')
    return cont_data_xds_low1, cont_data_xds_mid1, cont_data_xds_high1
    
    

def spi3_mean_ncfile_creator(output_path):
    spi_prod='mam'
    lt_month=['nov','dec','jan','feb']
    threshold=[-0.03, -0.56,-0.99]
    spi_month=5
    spi_name='MAM'
    for ltm in lt_month:
        input_path=f'output/spi3/{ltm}_tp_kmj_25km_6m_fcstd_1981/'
        ds_ens=ens_mem_combiner(input_path)
        print(ds_ens)
        spi_prod_list=spi3_prod_name_creator(ds_ens)
        print(spi_prod_list)
        ds_ens1 = ds_ens.assign_coords(spi_prod=('time',spi_prod_list))
        spi_ds_ens=ds_ens1.where(ds_ens1.spi_prod==spi_name, drop=True)
        ds_mild, ds_mod, ds_sev=ds_spi_mean_emprical_prbablity_creator(spi_ds_ens,spi_month,threshold)
        ds_mild.to_netcdf(f'{output_path}{spi_prod}_{ltm}_low.nc')
        ds_mod.to_netcdf(f'{output_path}{spi_prod}_{ltm}_mid.nc')
        ds_sev.to_netcdf(f'{output_path}{spi_prod}_{ltm}_high.nc')
        
        
        
def spi4_mean_ncfile_creator(output_path):
    spi_prod='jjas'
    lt_month=['mar','apr','may']
    threshold=[-0.01, -0.41,-0.99]
    spi_month=9
    spi_name='JJAS'
    for ltm in lt_month:
        input_path=f'output/spi4/{ltm}_tp_kmj_25km_6m_fcstd_1981/'
        ds_ens=ens_mem_combiner(input_path)
        print(ds_ens)
        spi_prod_list=spi4_prod_name_creator(ds_ens)
        print(spi_prod_list)
        ds_ens1 = ds_ens.assign_coords(spi_prod=('time',spi_prod_list))
        spi_ds_ens=ds_ens1.where(ds_ens1.spi_prod==spi_name, drop=True)
        ds_mild, ds_mod, ds_sev=ds_spi_mean_emprical_prbablity_creator(spi_ds_ens,spi_month,threshold)
        ds_mild.to_netcdf(f'{output_path}{spi_prod}_{ltm}_low.nc')
        ds_mod.to_netcdf(f'{output_path}{spi_prod}_{ltm}_mid.nc')
        ds_sev.to_netcdf(f'{output_path}{spi_prod}_{ltm}_high.nc')
        
        
def spi6_mean_ncfile_creator_a(output_path):
    spi_prod='mamjja'
    lt_month=['feb']
    threshold=[-0.02, -0.38,-1.01]
    spi_month=8
    spi_name='MAMJJA'
    for ltm in lt_month:
        input_path=f'output/spi6/{ltm}_tp_kmj_25km_6m_fcstd_1981/'
        ds_ens=ens_mem_combiner(input_path)
        print(ds_ens)
        spi_prod_list=spi6_prod_name_creator(ds_ens)
        print(spi_prod_list)
        ds_ens1 = ds_ens.assign_coords(spi_prod=('time',spi_prod_list))
        spi_ds_ens=ds_ens1.where(ds_ens1.spi_prod==spi_name, drop=True)
        ds_mild, ds_mod, ds_sev=ds_spi_mean_emprical_prbablity_creator(spi_ds_ens,spi_month,threshold)
        ds_mild.to_netcdf(f'{output_path}{spi_prod}_{ltm}_low.nc')
        ds_mod.to_netcdf(f'{output_path}{spi_prod}_{ltm}_mid.nc')
        ds_sev.to_netcdf(f'{output_path}{spi_prod}_{ltm}_high.nc')
        
        
def spi6_mean_ncfile_creator_b(output_path):
    spi_prod='amjjas'
    lt_month=['mar']
    threshold=[-0.02, -0.38,-1.01]
    spi_month=9
    spi_name='AMJJAS'
    for ltm in lt_month:
        input_path=f'output/spi6/{ltm}_tp_kmj_25km_6m_fcstd_1981/'
        ds_ens=ens_mem_combiner(input_path)
        print(ds_ens)
        spi_prod_list=spi6_prod_name_creator(ds_ens)
        print(spi_prod_list)
        ds_ens1 = ds_ens.assign_coords(spi_prod=('time',spi_prod_list))
        spi_ds_ens=ds_ens1.where(ds_ens1.spi_prod==spi_name, drop=True)
        ds_mild, ds_mod, ds_sev=ds_spi_mean_emprical_prbablity_creator(spi_ds_ens,spi_month,threshold)
        ds_mild.to_netcdf(f'{output_path}{spi_prod}_{ltm}_low.nc')
        ds_mod.to_netcdf(f'{output_path}{spi_prod}_{ltm}_mid.nc')
        ds_sev.to_netcdf(f'{output_path}{spi_prod}_{ltm}_high.nc')