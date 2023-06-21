#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from mpl_toolkits.axes_grid1 import ImageGrid
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import geopandas as gp

import xskillscore as xs
import regionmask
import collections

#%% common functions

def kmj_mask_creator():
    """
    

    Returns
    -------
    the_mask : TYPE
        DESCRIPTION.
    rl_dict : TYPE
        DESCRIPTION.

    """
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


#%% probablity exceedance plots utilitiy functions
        
        
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
    
def legend_maker(laxes):
    """
    

    Parameters
    ----------
    laxes : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    square6=plt.Rectangle((0.4, 0.1), 0.25, 0.15,color='#ffff00', clip_on=False)
    laxes.add_artist(square6)
    square5=plt.Rectangle((0.75, 0.1), 0.25, 0.15,color='#ffa500', clip_on=False)
    laxes.add_artist(square5)
    square5=plt.Rectangle((1.05, 0.1), 0.25, 0.15,color='#8b0000', clip_on=False)
    laxes.add_artist(square5)
    plt.text(0.7, 0.28,'SPI Category', horizontalalignment='left',fontsize=6,fontweight='bold',color='k', verticalalignment='center', transform =laxes.transAxes)
    plt.text(0.4, 0.05,'Moderate', horizontalalignment='left',fontsize=6,fontweight='bold',color='k', verticalalignment='center', transform =laxes.transAxes)
    plt.text(0.8, 0.05,'Extreme', horizontalalignment='left',fontsize=6,fontweight='bold',color='k', verticalalignment='center', transform =laxes.transAxes)
    plt.text(1.1, 0.05,'Severe', horizontalalignment='left',fontsize=6,fontweight='bold',color='k', verticalalignment='center', transform =laxes.transAxes)
    
    
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
    laxes=fig.add_axes([0.5,0.25, 0.3, 0.1], frame_on=False,zorder=0)
    laxes.xaxis.set_ticks_position('none')
    laxes.yaxis.set_ticks_position('none') 
    laxes.set_xticklabels('')
    laxes.set_yticklabels('')
    legend_maker(laxes)
    for ax, im in zip(grid, [im1, im2, im3, im4,im5,im6,im7,im8,im9,im10]):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
        ax.axis('off')
    #plt.show()
    for imgno in range(0,10):
        os.remove(f'{image_folder}{imgno}_{spi_prod}_{lt_month}.png')
    plt.savefig(f'{image_folder}{spi_prod}_{lt_month}.png',bbox_inches='tight',dpi=300)
    
    
def plot_prob(spi_prod,lt_month):
    """
    

    Parameters
    ----------
    spi_prod : TYPE
        DESCRIPTION.
    lt_month : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
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
    
#%% csv file creator for probablity of exceedance     

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
    """
    

    Parameters
    ----------
    spi_prod : TYPE
        DESCRIPTION.
    lt_month : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
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
    

#%% utilitiy function for stat metrices, create contigency table, apply statt metrices   
        
def det_cat_fct_init(thr, axis=None):
    """
    function from https://github.com/pySTEPS/pysteps/blob/master/pysteps/verification/detcatscores.py
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
    """
    Helper function for stat metrics, copied from 
    https://github.com/pySTEPS/pysteps/blob/master/pysteps/verification/detcatscores.py

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    x : TYPE
        DESCRIPTION.

    """
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
    """
    Funtion to do the stat metrics, supply the functin with name of spi product,
    lead month, threshold string to open the forcast netcdf file and obseration
    netcdf file. 

    Parameters
    ----------
    spi_prod : TYPE
        DESCRIPTION.
    lt_month : TYPE
        DESCRIPTION.
    thr_str : TYPE
        DESCRIPTION.
    thr_val : TYPE
        DESCRIPTION.
    the_mask : TYPE
        DESCRIPTION.
    region_idx : TYPE
        DESCRIPTION.
    rl_dict : TYPE
        DESCRIPTION.

    Returns
    -------
    db : TYPE
        DESCRIPTION.

    """
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
    """
    

    Parameters
    ----------
    spi_prod : TYPE
        DESCRIPTION.
    lt_month : TYPE
        DESCRIPTION.
    thr_val_list : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
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