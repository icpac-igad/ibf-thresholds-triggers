import regionmask
import numpy as np
import cartopy.crs as ccrs
import geopandas as gp
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

import numpy as np
import xarray as xr

import pandas as pd

def kmj_mask_creator():
    dis=gp.read_file('data/Karamoja_9_districts.shp')
    reg=gp.read_file('data/Karamoja_boundary_dissolved.shp')
    mds=pd.concat([dis,reg])
    mds['region']=[0,1,2,3,4,5,6,7,8,9]
    region_list=mds['admin2Name'].tolist()
    region_list[9]='Karamoja'
    mds['region_name']=region_list
    rl_dict=dict(zip(mds.region, mds.region_name))
    the_mask = regionmask.from_geopandas(mds,numbers='region',overlap=True)
    return the_mask, rl_dict

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
        year_data_xds=xr.concat([data_xds_mild1,data_xds_moderate1,data_xds_severe1],dim='time')
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
        
        
        
def prob_exceed_year_plot(ncfile_path,spi_prod,lt_month,the_mask,region_idx,rl_dict):
    low_ds_w=xr.open_dataset(f'{ncfile_path}{spi_prod}_{lt_month}_low.nc')
    maskd_low_ds=the_mask.mask_3D(low_ds_w)
    query_the_mask_low = maskd_low_ds.sel(region=region_idx)
    low_ds = low_ds_w.where(query_the_mask_low)
    low_ds1 = low_ds.reset_coords(drop=True).to_dataframe()
    low_ds2=low_ds1.reset_index()
    low_ds3 = low_ds2.groupby(['time'])['prob_exced'].mean().reset_index()
    ###############
    mid_ds_w=xr.open_dataset(f'{ncfile_path}{spi_prod}_{lt_month}_mid.nc')
    maskd_mid_ds=the_mask.mask_3D(mid_ds_w)
    query_the_mask_mid = maskd_mid_ds.sel(region=region_idx)
    mid_ds = mid_ds_w.where(query_the_mask_mid)
    mid_ds1= mid_ds.reset_coords(drop=True).to_dataframe()
    mid_ds2= mid_ds1.reset_index()
    mid_ds3= mid_ds2.groupby(['time'])['prob_exced'].mean().reset_index()
    ##############
    high_ds_w=xr.open_dataset(f'{ncfile_path}{spi_prod}_{lt_month}_high.nc')
    maskd_high_ds=the_mask.mask_3D(high_ds_w)
    query_the_mask_high = maskd_high_ds.sel(region=region_idx)
    high_ds = high_ds_w.where(query_the_mask_high)
    high_ds1= high_ds.reset_coords(drop=True).to_dataframe()
    high_ds2= high_ds1.reset_index()
    high_ds3= high_ds2.groupby(['time'])['prob_exced'].mean().reset_index()
    ##############
    year = low_ds.time.values
    prob_values = {
        'moderate': [i * 100 for i in low_ds3['prob_exced'].tolist()],
        'extreme': [i * 100 for i in mid_ds3['prob_exced'].tolist()],
        'severe': [i * 100 for i in high_ds3['prob_exced'].tolist()]
    }
    fig, ax = plt.subplots()
    #ax.stackplot(year, population_by_continent.values(),
    #             labels=population_by_continent.keys(), alpha=0.8,baseline='weighted_wiggle')
    ###################
    ax.margins(x=0)
    ax.margins(y=0)
    ax.set_axisbelow(True)
    ax.grid(zorder=0,color='gray', linestyle='dashed')
    ax.plot(year,prob_values['moderate'],color='#ffff00',lw=4)
    ax.plot(year,prob_values['extreme'],color='#ffa500',lw=4)
    ax.plot(year,prob_values['severe'],color='#8b0000',lw=4)
    ###################
    ax.fill_between(year, [0]*len(year),prob_values['moderate'],color='#ffff00',alpha=1,zorder=2)
    ax.fill_between(year, [0]*len(year),prob_values['extreme'],color='#ffa500',alpha=1,zorder=5)
    ax.fill_between(year, [0]*len(year),prob_values['severe'],color='#8b0000',alpha=1,zorder=10)
    ###################
    #ax.legend(['moderate','extreme','severe'],loc='upper left')
    #spi_prod_t=spi_prod.title()
    #lt_month_t=lt_month.title()
    ax.set_title(rl_dict[region_idx],fontsize=18,fontweight='bold')
    #ax.set_xlabel('Year')
    #ax.set_ylabel('Probablity(%)')
    #plt.xticks(np.arange(1981, 2023+1, 5.0),rotation=90)
    top_idx=[0,8,6,7]
    if region_idx==3:
        start, end = ax.get_xlim()
        ax.xaxis.set_ticks(np.arange(start, end, 5))
        start, end = ax.get_ylim()
        ax.yaxis.set_ticks(np.arange(start, 110, 10))
    elif region_idx==9:
        start, end = ax.get_xlim()
        ax.xaxis.set_ticks(np.arange(start, end, 5))
        start, end = ax.get_ylim()
        ax.yaxis.set_ticks(np.arange(start, 110, 10))
        ax.tick_params(labelbottom=False)
    elif region_idx in top_idx:
        start, end = ax.get_xlim()
        ax.xaxis.set_ticks(np.arange(start, end, 5))
        start, end = ax.get_ylim()
        ax.yaxis.set_ticks(np.arange(start, 110, 10))
        ax.tick_params(labelbottom=False)
        ax.tick_params(labelleft=False)
    else:
        start, end = ax.get_ylim()
        ax.yaxis.set_ticks(np.arange(start, 110, 10))
        start, end = ax.get_xlim()
        ax.xaxis.set_ticks(np.arange(start, end, 5))
        ax.tick_params(labelleft=False)
    plt.savefig(f'output/prob_plot/{region_idx}_{spi_prod}_{lt_month}.png',bbox_inches='tight')