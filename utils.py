import regionmask
import numpy as np
import cartopy.crs as ccrs
import geopandas as gp
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd

import numpy as np
import xarray as xr

import pandas as pd



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
    X = np.asanyarray,(X)
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
    
    
    
def spi_prod_name_creator_mam(ds_ens):
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

def jjas_spi_prod_name_creator(ds_ens):
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
        pro_ds=excprob(spi_ds_ens_kmj_st.to_array(), threshold, ignore_nan=False)
        data_xr = xr.DataArray(pro_ds,coords={'prob':['mild','moderate','severe'],'ens_mem':spi_ds_ens.ens_mem.values,
            'lat': spi_ds_ens.lat.values,'lon': spi_ds_ens.lon.values,'time':year}, 
        dims=["prob", "ens_mem", "lat","lon"])
        data_xds=data_xr.to_dataset(name='prob_exced')
        data_xds_mild=data_xds.sel(prob='mild')
        data_xds_mild1 = data_xds_mild.where(data_xds_mild.prob_exced >=1).sum("ens_mem")/data_xds_mild.where(data_xds_mild.prob_exced.notnull()).count("ens_mem")
        data_xds_moderate=data_xds.sel(prob='moderate')
        data_xds_moderate1 = data_xds_moderate.where(data_xds_moderate.prob_exced >=1).sum("ens_mem")/data_xds_moderate.where(data_xds_moderate.prob_exced.notnull()).count("ens_mem")
        data_xds_severe=data_xds.sel(prob='severe')
        data_xds_severe1 = data_xds_severe.where(data_xds_severe.prob_exced >=1).sum("ens_mem")/data_xds_severe.where(data_xds_severe.prob_exced.notnull()).count("ens_mem")
        year_data_xds=xr.concat([data_xds_mild1,data_xds_moderate1,data_xds_severe1],dim='time')
        cont_data_xds_mild.append(data_xds_mild1)
        cont_data_xds_moderate.append(data_xds_moderate1)
        cont_data_xds_severe.append(data_xds_severe1)
    cont_data_xds_mild1=xr.concat(cont_data_xds_mild,dim='time')
    cont_data_xds_moderate1=xr.concat(cont_data_xds_moderate,dim='time')
    cont_data_xds_severe1=xr.concat(cont_data_xds_severe,dim='time')
    return cont_data_xds_mild1, cont_data_xds_moderate1, cont_data_xds_severe1
    
    

