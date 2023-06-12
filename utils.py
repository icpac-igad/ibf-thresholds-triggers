
import os
import regionmask
import numpy as np
import cartopy.crs as ccrs
import geopandas as gp
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np

import numpy as np
import xarray as xr

import pandas as pd
import xskillscore as xs

import collections

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
    """
    https://stackoverflow.com/questions/29766827/matplotlib-make-axis-ticks-label-for-dates-bold

    Parameters
    ----------
    ncfile_path : TYPE
        DESCRIPTION.
    spi_prod : TYPE
        DESCRIPTION.
    lt_month : TYPE
        DESCRIPTION.
    the_mask : TYPE
        DESCRIPTION.
    region_idx : TYPE
        DESCRIPTION.
    rl_dict : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
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
    ###
    # prob_values['year']=year
    # db=pd.DataFrame(prob_values)
    # db['region']=rl_dict[region_idx]
    # db.to_csv(f'output/prob_csv/{region_idx}_{spi_prod}_{lt_month}.csv')
    ###
    left_idx=[9,6,4]
    bot_idx=[2,5]
    if region_idx in left_idx:
        start, end = ax.get_xlim()
        ax.xaxis.set_ticks(np.arange(start, end, 5))
        start, end = ax.get_ylim()
        ax.yaxis.set_ticks(np.arange(start, 110, 10))
        ax.xaxis.set_tick_params(labelsize=13)
        ax.yaxis.set_tick_params(labelsize=13)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontweight('bold') for label in labels]
        ax.tick_params(labelbottom=False)
    elif region_idx in bot_idx:
        start, end = ax.get_xlim()
        ax.xaxis.set_ticks(np.arange(start, end, 5))
        start, end = ax.get_ylim()
        ax.yaxis.set_ticks(np.arange(start, 110, 10))
        ax.xaxis.set_tick_params(labelsize=13)
        ax.yaxis.set_tick_params(labelsize=13)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontweight('bold') for label in labels]
        ax.tick_params(labelleft=False)
    elif region_idx==1:
        start, end = ax.get_xlim()
        ax.xaxis.set_ticks(np.arange(start, end, 5))
        start, end = ax.get_ylim()
        ax.yaxis.set_ticks(np.arange(start, 110, 10))
        ax.xaxis.set_tick_params(labelsize=13)
        ax.yaxis.set_tick_params(labelsize=13)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontweight('bold') for label in labels]
    else:
        start, end = ax.get_ylim()
        ax.yaxis.set_ticks(np.arange(start, 110, 10))
        start, end = ax.get_xlim()
        ax.xaxis.set_ticks(np.arange(start, end, 5))
        ax.tick_params(labelleft=False)
        ax.tick_params(labelbottom=False)
    plt.savefig(f'output/prob_plot/{region_idx}_{spi_prod}_{lt_month}.png',bbox_inches='tight')
    
    
    
def stitch_plots(image_folder,spi_prod,lt_month):
    """
    https://matplotlib.org/stable/gallery/axes_grid1/simple_axesgrid.html

    Parameters
    ----------
    image_folder : TYPE
        DESCRIPTION.
    spi_prod : TYPE
        DESCRIPTION.
    lt_month : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    im1 = plt.imread(f'{image_folder}9_{spi_prod}_{lt_month}.png')[:,:,:3]
    im2 = plt.imread(f'{image_folder}0_{spi_prod}_{lt_month}.png')[:,:,:3]
    im3 = plt.imread(f'{image_folder}8_{spi_prod}_{lt_month}.png')[:,:,:3]
    im4 = plt.imread(f'{image_folder}6_{spi_prod}_{lt_month}.png')[:,:,:3]
    im5 = plt.imread(f'{image_folder}7_{spi_prod}_{lt_month}.png')[:,:,:3]
    im6 = plt.imread(f'{image_folder}3_{spi_prod}_{lt_month}.png')[:,:,:3]
    im7 = plt.imread(f'{image_folder}4_{spi_prod}_{lt_month}.png')[:,:,:3]
    im8 = plt.imread(f'{image_folder}2_{spi_prod}_{lt_month}.png')[:,:,:3]
    im9 = plt.imread(f'{image_folder}5_{spi_prod}_{lt_month}.png')[:,:,:3]
    im10 = plt.imread(f'{image_folder}1_{spi_prod}_{lt_month}.png')[:,:,:3]
    fig = plt.figure(figsize=(8., 12.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(4, 3),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )
    grid[-1].remove()
    grid[-2].remove()
    for ax, im in zip(grid, [im1, im2, im3, im4,im5,im6,im7,im8,im9,im10]):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
        ax.axis('off')
    #plt.show()
    for imgno in range(0,10):
        os.remove(f'{image_folder}{imgno}_{spi_prod}_{lt_month}.png')
    plt.savefig(f'{image_folder}{spi_prod}_{lt_month}.png',bbox_inches='tight',dpi=300)
    
    
def plot_prob(spi_prod,lt_month):
    the_mask, rl_dict=kmj_mask_creator()
    ncfile_path='output/prob/'
    #spi_prod='mam'
    #lt_month='jan'
    #region_idx=9
    ridx_list=[0,1,2,3,4,5,6,7,8,9]
    for ridx in ridx_list:
        prob_exceed_year_plot(ncfile_path,spi_prod,lt_month,the_mask,ridx,rl_dict)
    image_folder='output/prob_plot/'
    stitch_plots(image_folder,spi_prod,lt_month)
    
def prob_exceed_csv(ncfile_path,spi_prod,lt_month,the_mask,region_idx,rl_dict):
    """
    https://stackoverflow.com/questions/29766827/matplotlib-make-axis-ticks-label-for-dates-bold

    Parameters
    ----------
    ncfile_path : TYPE
        DESCRIPTION.
    spi_prod : TYPE
        DESCRIPTION.
    lt_month : TYPE
        DESCRIPTION.
    the_mask : TYPE
        DESCRIPTION.
    region_idx : TYPE
        DESCRIPTION.
    rl_dict : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
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
    mol_dict={'mam':'05-01','jjas':'09-01','mamjja':'08-01','amjjas':'09-01'}
    mdc=mol_dict[spi_prod]
    year = low_ds.time.values
    year1=[f'{yr}-{mdc}' for yr in year]
    prob_values = {
        'moderate': [i * 100 for i in low_ds3['prob_exced'].tolist()],
        'extreme': [i * 100 for i in mid_ds3['prob_exced'].tolist()],
        'severe': [i * 100 for i in high_ds3['prob_exced'].tolist()]
    }
    ###
    prob_values['year']=year1
    db1=pd.DataFrame(prob_values['moderate'])
    db1['year']=year1
    db1.columns=['prob','year']
    db1['thre']='low'
    db2=pd.DataFrame(prob_values['extreme'])
    db2['year']=year1
    db2.columns=['prob','year']
    db2['thre']='mid'
    db3=pd.DataFrame(prob_values['severe'])
    db3['year']=year1
    db3.columns=['prob','year']
    db3['thre']='high'
    dba=pd.concat([db1,db2,db3])
    dba['region']=rl_dict[region_idx]
    dba['prod']=f'{spi_prod}_{lt_month}_'+dba['thre']+'_'+dba['year']
    return dba
    
    
def csv_prob(spi_prod,lt_month):
    the_mask, rl_dict=kmj_mask_creator()
    ncfile_path='output/prob/'
    #spi_prod='mam'
    #lt_month='jan'
    #region_idx=9
    ridx_list=[0,1,2,3,4,5,6,7,8,9]
    cont=[]
    for ridx in ridx_list:
        db=prob_exceed_csv(ncfile_path,spi_prod,lt_month,the_mask,ridx,rl_dict)
        cont.append(db)
    cont1=pd.concat(cont)
    cont1.to_csv(f'output/prob_csv/{spi_prod}_{lt_month}.csv',index=False)
    
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
        
def det_cat_fct_init(thr, axis=None):
    """
    Initialize a contingency table object.
    Parameters
    ----------
    thr: float
        threshold that is applied to predictions and observations in order
        to define events vs no events (yes/no).
    axis: None or int or tuple of ints, optional
        Axis or axes along which a score is integrated. The default, axis=None,
        will integrate all of the elements of the input arrays.\n
        If axis is -1 (or any negative integer),
        the integration is not performed
        and scores are computed on all of the elements in the input arrays.\n
        If axis is a tuple of ints, the integration is performed on all of the
        axes specified in the tuple.
    Returns
    -------
    out: dict
      The contingency table object.
    """
    contab = {}
    # catch case of axis passed as integer
    def get_iterable(x):
        if x is None or (
            isinstance(x, collections.abc.Iterable) and not isinstance(x, int)
        ):
            return x
        else:
            return (x,)
    contab["thr"] = thr
    contab["axis"] = get_iterable(axis)
    contab["hits"] = None
    contab["false_alarms"] = None
    contab["misses"] = None
    contab["correct_negatives"] = None
    return contab


def det_cat_fct_accum(contab, pred, obs):
    """
    https://github.com/pySTEPS/pysteps/blob/master/pysteps/verification/detcatscores.py
    Accumulate the frequency of "yes" and "no" forecasts and observations
    in the contingency table.
    Parameters
    ----------
    contab: dict
      A contingency table object initialized with
      pysteps.verification.detcatscores.det_cat_fct_init.
    pred: array_like
        Array of predictions. NaNs are ignored.
    obs: array_like
        Array of verifying observations. NaNs are ignored.
    """
    pred = np.asarray(pred.copy())
    obs = np.asarray(obs.copy())
    axis = tuple(range(pred.ndim)) if contab["axis"] is None else contab["axis"]
    # checks
    if pred.shape != obs.shape:
        raise ValueError(
            "the shape of pred does not match the shape of obs %s!=%s"
            % (pred.shape, obs.shape)
        )
    if pred.ndim <= np.max(axis):
        raise ValueError(
            "axis %d is out of bounds for array of dimension %d"
            % (np.max(axis), len(pred.shape))
        )
    idims = [dim not in axis for dim in range(pred.ndim)]
    nshape = tuple(np.array(pred.shape)[np.array(idims)])
    if contab["hits"] is None:
        # initialize the count arrays in the contingency table
        contab["hits"] = np.zeros(nshape, dtype=int)
        contab["false_alarms"] = np.zeros(nshape, dtype=int)
        contab["misses"] = np.zeros(nshape, dtype=int)
        contab["correct_negatives"] = np.zeros(nshape, dtype=int)
    else:
        # check dimensions
        if contab["hits"].shape != nshape:
            raise ValueError(
                "the shape of the input arrays does not match "
                + "the shape of the "
                + "contingency table %s!=%s" % (nshape, contab["hits"].shape)
            )
    # add dummy axis in case integration is not required
    if np.max(axis) < 0:
        pred = pred[None, :]
        obs = obs[None, :]
        axis = (0,)
    axis = tuple([a for a in axis if a >= 0])
    # apply threshold
    predb = pred > contab["thr"]
    obsb = obs > contab["thr"]
    # calculate hits, misses, false positives, correct rejects
    H_idx = np.logical_and(predb == 1, obsb == 1)
    F_idx = np.logical_and(predb == 1, obsb == 0)
    M_idx = np.logical_and(predb == 0, obsb == 1)
    R_idx = np.logical_and(predb == 0, obsb == 0)
    # accumulate in the contingency table
    contab["hits"] += np.nansum(H_idx.astype(int), axis=axis)
    contab["misses"] += np.nansum(M_idx.astype(int), axis=axis)
    contab["false_alarms"] += np.nansum(F_idx.astype(int), axis=axis)
    contab["correct_negatives"] += np.nansum(R_idx.astype(int), axis=axis)
    return contab

    
def get_iterable(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        else:
            return (x,)    

    
def det_cat_fct_compute(contab, scores=""):
    """
    https://github.com/pySTEPS/pysteps/blob/master/pysteps/verification/detcatscores.py
    Compute simple and skill scores for deterministic categorical
    (dichotomous) forecasts from a contingency table object.
    Parameters
    ----------
    contab: dict
      A contingency table object initialized with
      pysteps.verification.detcatscores.det_cat_fct_init and populated with
      pysteps.verification.detcatscores.det_cat_fct_accum.
    scores: {string, list of strings}, optional
        The name(s) of the scores. The default, scores="", will compute all
        available scores.
        The available score names a
        .. tabularcolumns:: |p{2cm}|L|
        +------------+--------------------------------------------------------+
        | Name       | Description                                            |
        +============+========================================================+
        |  ACC       | accuracy (proportion correct)                          |
        +------------+--------------------------------------------------------+
        |  BIAS      | frequency bias                                         |
        +------------+--------------------------------------------------------+
        |  CSI       | critical success index (threat score)                  |
        +------------+--------------------------------------------------------+
        |  ETS       | equitable threat score                                 |
        +------------+--------------------------------------------------------+
        |  F1        | the harmonic mean of precision and sensitivity         |
        +------------+--------------------------------------------------------+
        |  FA        | false alarm rate (prob. of false detection, fall-out,  |
        |            | false positive rate)                                   |
        +------------+--------------------------------------------------------+
        |  FAR       | false alarm ratio (false discovery rate)               |
        +------------+--------------------------------------------------------+
        |  GSS       | Gilbert skill score (equitable threat score)           |
        +------------+--------------------------------------------------------+
        |  HK        | Hanssen-Kuipers discriminant (Pierce skill score)      |
        +------------+--------------------------------------------------------+
        |  HSS       | Heidke skill score                                     |
        +------------+--------------------------------------------------------+
        |  MCC       | Matthews correlation coefficient                       |
        +------------+--------------------------------------------------------+
        |  POD       | probability of detection (hit rate, sensitivity,       |
        |            | recall, true positive rate)                            |
        +------------+--------------------------------------------------------+
        |  SEDI      | symmetric extremal dependency index                    |
        +------------+--------------------------------------------------------+
    Returns
    -------
    result: dict
        Dictionary containing the verification results.
    """
    # catch case of single score passed as string
    scores = get_iterable(scores)
    H = 1.0 * contab["hits"]  # true positives
    M = 1.0 * contab["misses"]  # false negatives
    F = 1.0 * contab["false_alarms"]  # false positives
    R = 1.0 * contab["correct_negatives"]  # true negatives
    result = {}
    for score in scores:
        # catch None passed as score
        if score is None:
            continue
        score_ = score.lower()
        # simple scores
        POD = H / (H + M)
        FAR = F / (H + F)
        FA = F / (F + R)
        s = (H + M) / (H + M + F + R)
        if score_ in ["pod", ""]:
            # probability of detection
            result["POD"] = POD
        if score_ in ["far", ""]:
            # false alarm ratio
            result["FAR"] = FAR
        if score_ in ["fa", ""]:
            # false alarm rate (prob of false detection)
            result["FA"] = FA
        if score_ in ["acc", ""]:
            # accuracy (fraction correct)
            ACC = (H + R) / (H + M + F + R)
            result["ACC"] = ACC
        if score_ in ["csi", ""]:
            # critical success index
            CSI = H / (H + M + F)
            result["CSI"] = CSI
        if score_ in ["bias", ""]:
            # frequency bias
            B = (H + F) / (H + M)
            result["BIAS"] = B
        # skill scores
        if score_ in ["hss", ""]:
            # Heidke Skill Score (-1 < HSS < 1) < 0 implies no skill
            HSS = 2 * (H * R - F * M) / ((H + M) * (M + R) + (H + F) * (F + R))
            result["HSS"] = HSS
        if score_ in ["hk", ""]:
            # Hanssen-Kuipers Discriminant
            HK = POD - FA
            result["HK"] = HK
        if score_ in ["gss", "ets", ""]:
            # Gilbert Skill Score
            GSS = (POD - FA) / ((1 - s * POD) / (1 - s) + FA * (1 - s) / s)
            if score_ == "ets":
                result["ETS"] = GSS
            else:
                result["GSS"] = GSS
        if score_ in ["sedi", ""]:
            # Symmetric extremal dependence index
            SEDI = (np.log(FA) - np.log(POD) + np.log(1 - POD) - np.log(1 - FA)) / (
                np.log(FA) + np.log(POD) + np.log(1 - POD) + np.log(1 - FA)
            )
            result["SEDI"] = SEDI
        if score_ in ["mcc", ""]:
            # Matthews correlation coefficient
            MCC = (H * R - F * M) / np.sqrt((H + F) * (H + M) * (R + F) * (R + M))
            result["MCC"] = MCC
        if score_ in ["f1", ""]:
            # F1 score
            F1 = 2 * H / (2 * H + F + M)
            result["F1"] = F1
    return result


def verif_cont_stats(spi_prod,lt_month,thr_str,thr_val,the_mask,region_idx,rl_dict):
    obs_w=xr.open_dataset(f'output/obs/{spi_prod}_kmj_km25_chirps-v2.0.monthly.nc')
    end_dt=obs_w['time'].values[-1]    
    fct_w=xr.open_dataset(f'output/mean_spi/{spi_prod}_{lt_month}_{thr_str}.nc')
    start_dt=fct_w['time'].values[0]
    obs_w1=obs_w.sel(time=slice(start_dt, end_dt))
    fct_w1=fct_w.sel(time=slice(start_dt, end_dt))
    #############
    m_fct=the_mask.mask_3D(fct_w1)
    q_mfct = m_fct.sel(region=region_idx)
    fct_ds = fct_w1.where(q_mfct)
    fct_ds1 = fct_ds.reset_coords(drop=True).to_dataframe()
    fct_ds2=fct_ds1.reset_index()
    ############
    m_obs=the_mask.mask_3D(obs_w1)
    q_mobs = m_obs.sel(region=region_idx)
    obs_ds = obs_w1.where(q_mobs)
    obs_ds1 = obs_ds.reset_coords(drop=True).to_dataframe()
    obs_ds2=obs_ds1.reset_index()
    #############    
    fct_ds3=fct_ds2.drop_duplicates('time')
    date_list=fct_ds3['time'].tolist()
    contab=det_cat_fct_init(thr_val)
    metr_cont=[]
    date_list1=[x.strftime('%Y-%m-%d') for x in date_list]
    for datey in date_list1:
        if datey=='1985-09-01':
            pass
        else:
            y_fct=fct_ds2[fct_ds2['time']==datey]
            y_fct1 = y_fct[y_fct['spi_ema'].notnull()]
            y_obs=obs_ds2[obs_ds2['time']==datey]
            y_obs1 = y_obs[y_obs['spi'].notnull()]
            fct_spi=y_fct1['spi_ema'].tolist()
            obs_spi=y_obs1['spi'].tolist()
            pcontab=det_cat_fct_accum(contab, fct_spi, obs_spi)
            metr=det_cat_fct_compute(pcontab,scores=['FAR','POD','BIAS','HSS','HK'])
            metr['time']=datey
            metr_cont.append(metr)
    db=pd.DataFrame(metr_cont)
    db['region']=rl_dict[region_idx]
    return db


def metrics_csv(spi_prod,lt_month,thr_val_list):
    the_mask, rl_dict=kmj_mask_creator()
    ridx_list=[0,1,2,3,4,5,6,7,8,9]
    thr_str_list=['low','mid','high']
    for tstr,tval in zip(thr_str_list[2:3],thr_val_list):
        print(tstr)
        cont_db=[]
        for ridx in ridx_list[1:]:
            db=verif_cont_stats(spi_prod,lt_month,tstr,tval,the_mask,ridx,rl_dict)
            cont_db.append(db)
            print(ridx)
        db1=pd.concat(cont_db)
        db1.to_csv(f'output/metrics/{spi_prod}_{lt_month}_{tstr}.csv')