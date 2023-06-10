
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr


import sys
from PIL import Image
import numpy as np

filename="data/Karamoja_9_districts.shp"

shape_feature = ShapelyFeature(Reader(filename).geometries(), ccrs.PlateCarree(), 
                               linewidth = 1, facecolor = (1, 1, 1, 0), 
                               edgecolor = 'black')

flatui = ["#8000ff", "#0d5a7f", "#00ea12", "#6bff00", "#e9ff00", "#facf00", "#ff5200", "#ff0000", "#bf001f", "#71008a"]


prob_path='probablity/'

spi_prod='mam'

lt_month_list=['nov','dec','jan','feb']

thres_list=['mild','mod','sev']

def concat_images(imga, imgb):
    """
    Combines two color image ndarrays side-by-side.
    from https://stackoverflow.com/a/30228563/2501953
    """
    ha,wa = imga.shape[:2]
    hb,wb = imgb.shape[:2]
    max_height = np.max([ha, hb])
    total_width = wa+wb
    new_img = np.zeros(shape=(max_height, total_width, 3))
    new_img[:ha,:wa]=imga
    new_img[:hb,wa:wa+wb]=imgb
    return new_img

def concat_n_images(image_path_list,plot_title):
    """
    Combines N color images from a list of image paths.
    from https://stackoverflow.com/a/30228563/2501953
    """
    plt.axis('off')
    plt.title(plot_title, y=-0.01,fontsize=18,fontweight='bold')
    output = None
    for i, img_path in enumerate(image_path_list):
        img = plt.imread(img_path)[:,:,:3]
        if i==0:
            output = img
        else:
            output = concat_images(output, img)
    return output

year='2023'


for lt_mon in lt_month_list:
    db=xr.open_dataset(f'{prob_path}{spi_prod}_{lt_mon}_mild.nc')
    db1=db['prob_exced'].sel(time=year)
    db2=db1*100
    db3 = db2.to_dataset(name = 'prob_exced')
    #fig.add_feature(shape_feature)
    fig = plt.figure(figsize=(6,8))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax.set_extent([33, 35.5, 1, 4.5])
    ax.add_feature(shape_feature, facecolor='None')
    ax.set_yticks([1,1.5,2.0,2.5,3.0,3.5,4.0, 4.5], crs=ccrs.PlateCarree())
    ax.set_xticks([33,33.5,34.0,34.5,35.0, 35.5], crs=ccrs.PlateCarree())
    plt.title('Probability \n Moderate -0.03 to -0.59', fontsize=18,fontweight='bold')
    db3['prob_exced'].plot(ax=ax,levels=[0,10, 20, 30, 40, 50, 60,70, 80, 90,100],colors=flatui,add_labels=False,add_colorbar=True)
    plt.savefig(f'{prob_path}{spi_prod}_{lt_mon}_mild.png')
    ###############
    ###############
    db=xr.open_dataset(f'{prob_path}{spi_prod}_{lt_mon}_mod.nc')
    db1=db['prob_exced'].sel(time=year)
    db2=db1*100
    db3 = db2.to_dataset(name = 'prob_exced')
    #fig.add_feature(shape_feature)
    fig = plt.figure(figsize=(6,8))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax.set_extent([33, 35.5, 1, 4.5])
    ax.add_feature(shape_feature, facecolor='None')
    ax.set_yticks([1,1.5,2.0,2.5,3.0,3.5,4.0, 4.5], crs=ccrs.PlateCarree())
    ax.set_xticks([33,33.5,34.0,34.5,35.0, 35.5], crs=ccrs.PlateCarree())
    plt.title('Probability \n Severe -0.6 to -0.89', fontsize=18,fontweight='bold')
    db3['prob_exced'].plot(ax=ax,levels=[0,10, 20, 30, 40, 50, 60,70, 80, 90,100],colors=flatui,add_labels=False,add_colorbar=True)
    plt.savefig(f'{prob_path}{spi_prod}_{lt_mon}_mod.png')
    ###############
    ###############
    db=xr.open_dataset(f'{prob_path}{spi_prod}_{lt_mon}_sev.nc')
    db1=db['prob_exced'].sel(time=year)
    db2=db1*100
    db3 = db2.to_dataset(name = 'prob_exced')
    #fig.add_feature(shape_feature)
    fig = plt.figure(figsize=(6,8))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax.set_extent([33, 35.5, 1, 4.5])
    ax.add_feature(shape_feature, facecolor='None')
    ax.set_yticks([1,1.5,2.0,2.5,3.0,3.5,4.0, 4.5], crs=ccrs.PlateCarree())
    ax.set_xticks([33,33.5,34.0,34.5,35.0, 35.5], crs=ccrs.PlateCarree())
    plt.title('Probability \n Extreme -0.9 and below', fontsize=18,fontweight='bold')
    db3['prob_exced'].plot(ax=ax,levels=[0,10, 20, 30, 40, 50, 60,70, 80, 90,100],colors=flatui,add_labels=False,add_colorbar=True)
    plt.savefig(f'{prob_path}{spi_prod}_{lt_mon}_sev.png')
    images=[f'{prob_path}{spi_prod}_{lt_mon}_mild.png',f'{prob_path}{spi_prod}_{lt_mon}_mod.png',
           f'{prob_path}{spi_prod}_{lt_mon}_sev.png']
    spi_prod1=spi_prod.upper()
    lt_mon1=lt_mon.title()
    plot_title=f'{spi_prod1} forcast on {year} with lead time from {lt_mon1}'
    output = concat_n_images(images, plot_title)
    fig = plt.figure(figsize=(14,8))
    plt.axis('off')
    plt.title(plot_title, y=-0.01,fontsize=18,fontweight='bold')
    plt.imshow(output)
    plt.savefig(f'{prob_path}prob_{spi_prod}_{lt_mon}_{year}.png',bbox_inches='tight')