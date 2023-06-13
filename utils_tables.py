#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 06:05:04 2023

@author: bulbul
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import six
from datetime import datetime
import textwrap as tw

#%% table plot matplotlib

### Define the picture size and remove the ticks
fig = plt.figure()
fig.set_size_inches(6.7,18)
table=fig.add_axes([0.08, 0.02, 0.55, 0.9], frame_on=False) 
table.xaxis.set_ticks_position('none')
table.yaxis.set_ticks_position('none') 
table.set_xticklabels('')
table.set_yticklabels('')

### functions for whole column, row editing
def set_align_for_column(table, col, align="left"):
    cells = [key for key in table._cells if key[1] == col]
    for cell in cells:
        table._cells[cell]._loc = align


def set_width_for_column(table, col, width):
    cells = [key for key in table._cells if key[1] == col]
    for cell in cells:
        table._cells[cell]._width = width
        
def set_height_for_row(table, row, height):
    cells = [key for key in table._cells if key[0] == row]
    for cell in cells:
        table._cells[cell]._height = height

def colorcell(tablerows,tablecols,cellDict):
    allcells=[(x,y) for x in tablerows[1:] for y in tablecols[1:]]
    for alcls in allcells:
        if float(cellDict[alcls]._text.get_text()) <=30:
            cellDict[alcls].set_facecolor('#009600')
        elif 30 < float(cellDict[alcls]._text.get_text()) <= 60:
            cellDict[alcls].set_facecolor('#64C800')
        elif 60 < float(cellDict[alcls]._text.get_text()) <= 90:
            cellDict[alcls].set_facecolor('#ffff00')
        elif 90 < float(cellDict[alcls]._text.get_text()) <= 120:
            cellDict[alcls].set_facecolor('#ff7800')
        elif 120 < float(cellDict[alcls]._text.get_text()) <= 250:
            cellDict[alcls].set_facecolor('#ff0000')
        else:
            cellDict[alcls].set_facecolor('#961414')


def remove_cell_value(tablerows,tablecols,mpl_table):
    allcells=[(x,y) for x in tablerows[1:] for y in tablecols[1:]]
    for alcls in allcells:
        mpl_table._cells[alcls]._text.set_text('')

def set_height_for_row_except_head(table, rowlist, height):
    cells_list=[]
    for row in rowlist:
        cells = [key for key in table._cells if key[0] == row]
        cells_list.append(cells)
    for cells in cells_list:
        for cell in cells:
             table._cells[cell]._height = height


### funciton for table creation
def render_mpl_table(data,df3, col_width=1.0, row_height=1.625, font_size=12,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=['','','','',''],cellLoc='center', **kwargs)
    set_align_for_column(mpl_table, col=0, align="left")
    set_width_for_column(mpl_table, 0, 0.1)
    set_width_for_column(mpl_table, 1, 0.02)
    set_width_for_column(mpl_table, 2, 0.02)
    set_width_for_column(mpl_table, 3, 0.02)
    set_width_for_column(mpl_table, 4, 0.02)
    set_height_for_row(mpl_table, 0, 0.05)
    set_height_for_row_except_head(mpl_table, np.arange(1,len(data.index)), 0.012)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)
    cellDict=mpl_table.get_celld()
    tablerows=np.arange(0,len(df3.index)+1)
    tablecols=np.arange(0,len(df3.columns))
    for k, cell in  six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    colorcell(tablerows,tablecols,cellDict)
    headings=data.columns
    plt.text(0.070, 0.950,'City', fontsize=16, fontweight='bold', color='w', ha='left', va='center', transform =ax.transAxes)
    plt.text(0.600, 0.965,headings[1], fontsize=10, fontweight='bold',color='w', ha='left', va ='center',rotation=90, transform =ax.transAxes)
    plt.text(0.715, 0.965,headings[2], fontsize=10, fontweight='bold',color='w', ha='left', va ='center',rotation=90, transform =ax.transAxes)
    plt.text(0.815, 0.965,headings[3], fontsize=10, fontweight='bold',color='w', ha='left', va ='center',rotation=90, transform =ax.transAxes)
    plt.text(0.925, 0.965,headings[4], fontsize=10, fontweight='bold',color='w', ha='left', va ='center',rotation=90, transform =ax.transAxes)
    removecellvalue(tablerows,tablecols,mpl_table)
    
    
#%% df csv creator, pivot for plot  
def metric_db():
    spi_prod_list=['mam','jjas','mamjja','amjjas']
    spi_prod={'mam':['nov','dec','jan','feb'],'jjas':['mar','apr','may'],'mamjja':['feb'],'amjjas':['mar']}
    inpt_path='output/metrics/'
    fl_name=[]
    db_cont1=[]
    for sppd in spi_prod_list:
        lt_month=spi_prod[sppd]
        db_cont=[]
        for ltm in lt_month:
            db1=pd.read_csv(f'{inpt_path}{sppd}_{ltm}_low.csv')
            db1['prod']=f'{sppd}_{ltm}_low_'+db1['time']
            db2=pd.read_csv(f'{inpt_path}{sppd}_{ltm}_mid.csv')
            db2['prod']=f'{sppd}_{ltm}_mid_'+db2['time']
            db3=pd.read_csv(f'{inpt_path}{sppd}_{ltm}_high.csv')
            db3['prod']=f'{sppd}_{ltm}_high_'+db3['time']
            fl_name.append(f'{inpt_path}{sppd}_{ltm}_low.csv')
            fl_name.append(f'{inpt_path}{sppd}_{ltm}_mid.csv')
            fl_name.append(f'{inpt_path}{sppd}_{ltm}_high.csv')
            db=pd.concat([db1,db2,db3])
            db_cont.append(db)
        dbc=pd.concat(db_cont)
        db_cont1.append(dbc)
    dbf=pd.concat(db_cont1)
    dbf['prod']=dbf['prod']+'_'+dbf['region']
    return dbf


def prob_db():
    spi_prod_list=['mam','jjas','mamjja','amjjas']
    spi_prod={'mam':['nov','dec','jan','feb'],'jjas':['mar','apr','may'],'mamjja':['feb'],'amjjas':['mar']}
    inpt_path='output/prob_csv/'
    _=[]
    for sppd in spi_prod_list:
        lt_month=spi_prod[sppd]
        for ltm in lt_month:
            db=pd.read_csv(f'{inpt_path}{sppd}_{ltm}.csv')
            _.append(db)
    db1=pd.concat(_)
    db1['prod']=db1['prod']+'_'+db1['region']
    return db1


def subset_db(row,str_list):
    df_row=row['spi_prod']+'_'+row['lt_month']+'_'+row['region_y']+'_'+row['thre']
    given_row=str_list[0]+'_'+str_list[1]+'_'+str_list[2]+'_'+str_list[3]
    if given_row==df_row:
        mask=1
    else:
        mask=0
    return mask

def bias_filter(db):
    #https://stackoverflow.com/a/12141207/2501953
    bf=min(db['BIAS'].tolist(), key=lambda x:abs(x-1))
    db1=db[db['BIAS']==bf]
    db2=db1[['BIAS','FAR','POD','prob']]
    db2.columns=['BIAS','bb_FAR','bb_POD','bb_prob']
    db3=db2.reset_index()
    db4=db3[0:1]
    return db4
    
    
def hss_filter(db):
    db1=db.loc[db["HSS"].idxmax()]
    a=db1.reset_index()
    a1=a.T
    a1.columns = a1.iloc[0]
    a2=a1.drop(index=['index'])
    db2=a2[['HSS','FAR','POD','prob']]
    db2.columns=['HSS','bh_FAR','bh_POD','bh_prob']
    db3=db2.reset_index()
    return db3

def hk_filter(db):
    db1=db.loc[db["HK"].idxmax()]
    a=db1.reset_index()
    a1=a.T
    a1.columns = a1.iloc[0]
    a2=a1.drop(index=['index'])
    db2=a2[['HK','FAR','POD','prob']]
    db2.columns=['HK','bk_FAR','bk_POD','bk_prob']
    db3=db2.reset_index()
    return db3



def final_des(db):
    #taking minumum probablity
    db1=db.T
    #db[["bb_prob", "bh_prob","bk_prob"]].max(axis=1)
    db2=db1.reset_index()
    db2.columns=['title','value']
    db3=db2[db2['title'].isin(['bb_prob','bh_prob','bk_prob'])]
    #db4=db3.loc[db3.idxmax()]
    aa=db3.min()
    #aa=db3.max()
    head=aa['title'].split('_')[0]
    head_col=[f'{head}_FAR',f'{head}_POD',f'{head}_prob']
    dbf=db[head_col]
    #db.columns
    dbf.columns=['FAR','POD','prob']
    return dbf


def all_var_except_thre_string_maker():
    spi_prod_list=['mam','jjas','mamjja','amjjas']
    spi_prod={'mam':['nov','dec','jan','feb'],'jjas':['mar','apr','may'],'mamjja':['feb'],'amjjas':['mar']}
    # fl_name=[]
    regions=['Abim','Napak','Nabilatuk','Kotido', 'Moroto', 'Nakapiripirit', 'Kaabong', 'Karenga', 'Amudat', 'Karamoja']
    db_cont1=[]
    for sppd in spi_prod_list:
        lt_month=spi_prod[sppd]
        db_cont=[]
        for ltm in lt_month:
             for reg in regions:
                 #odb1=odb[odb]
                 ndc={}
                 ndc['sp']=sppd
                 ndc['ltm']=ltm
                 ndc['reg']=reg
                 db_cont1.append(ndc)
    db=pd.DataFrame(db_cont1)
    return db


def fdf_except_thre_db_maker(thre):
    str_db=all_var_except_thre_string_maker()
    odb=pd.read_csv('output/tables/all_metrices_thres.csv')
    odb['spi_prod'] = odb['prod'].str.split('_').str[0]
    odb['lt_month'] = odb['prod'].str.split('_').str[1]
    fdb_cont=[]
    for idx,drow in str_db.iterrows():
        str_list=[drow['sp'],drow['ltm'],drow['reg'],thre]
        odb['mask'] = odb.apply(lambda row: subset_db(row,str_list), axis = 1)
        odb_maskd=odb[odb['mask']==1]
        if odb_maskd.empty:
            print(str_list)
        else:
            bdb=bias_filter(odb_maskd)
            hdb=hss_filter(odb_maskd)
            kdb=hk_filter(odb_maskd)
            db=pd.concat([bdb,hdb,kdb],axis=1)
            f_db=final_des(db)
            f_db.loc[0,'spi_prod']=drow['sp']
            f_db.loc[0,'lt_month']=drow['ltm']
            f_db.loc[0,'thre']=thre
            f_db.loc[0,'region']=drow['reg']
            fdb_cont.append(f_db)
    fdb_cont1=pd.concat(fdb_cont)
    return fdb_cont1


def thres_db_maker():
    threl=['low','mid','high']
    f_c=[]
    for thr in threl:
        db=fdf_except_thre_db_maker(thr)
        #f_c.append(db)
        db.to_csv(f'output/tables/final_db_{thr}.csv',index=False) 
    #fc1=pd.concat(f_c)
    

def df_far_table_maker(thre,regions_list):
    db=pd.read_csv(f'output/tables/final_db_{thre}.csv')
    db1=db[['FAR','prob','spi_prod','lt_month','region']]
    empty_df_cols=['spi_prod','region','prob nov','FAR nov','prob dec','FAR dec',
                   'prob jan','FAR jan','prob feb','FAR feb','prob mar','FAR mar',
                   'prob apr','FAR apr','prob may','FAR may']
    _=[]
    for region_str in regions_list:
        db2=db1[db1['region']==region_str]
        for sp in ['mam','jjas','mamjja','amjjas']:
            db3=db2[db2['spi_prod']==sp]
            db4=db3.pivot(index=['spi_prod','region'], columns="lt_month", values=["prob",'FAR'])
            db4.reset_index(inplace=True)
            db4.columns = [' '.join(col).strip() for col in db4.columns.values]
            df = pd.DataFrame(columns=empty_df_cols)
            df1=pd.concat([df,db4])
            _.append(df1)
    df_far=pd.concat(_)
    return df_far


def df_pod_table_maker(thre,regions_list):
    db=pd.read_csv(f'output/tables/final_db_{thre}.csv')
    db1=db[['POD','prob','spi_prod','lt_month','region']]
    empty_df_cols=['spi_prod','region','prob nov','POD nov','prob dec','POD dec',
                   'prob jan','POD jan','prob feb','POD feb','prob mar','POD mar',
                   'prob apr','POD apr','prob may','POD may']
    _=[]
    for region_str in regions_list:
        db2=db1[db1['region']==region_str]
        for sp in ['mam','jjas','mamjja','amjjas']:
            db3=db2[db2['spi_prod']==sp]
            db4=db3.pivot(index=['spi_prod','region'], columns="lt_month", values=["prob",'POD'])
            db4.reset_index(inplace=True)
            db4.columns = [' '.join(col).strip() for col in db4.columns.values]
            df = pd.DataFrame(columns=empty_df_cols)
            df1=pd.concat([df,db4])
            _.append(df1)
    df_pod=pd.concat(_)
    return df_pod


def far_pod_table_all(thre):
    regions_list=['Abim','Napak','Nabilatuk','Kotido', 'Moroto',
             'Nakapiripirit', 'Kaabong', 'Karenga', 'Amudat',
             'Karamoja']
    if thre=='low':
        fdb_low=df_far_table_maker('low',regions_list)
        pdb_low=df_pod_table_maker('low',regions_list)
        return fdb_low,pdb_low
    elif thre=='mid':
        fdb_mid=df_far_table_maker('mid',regions_list)
        pdb_mid=df_pod_table_maker('mid',regions_list)
        return fdb_mid,pdb_mid
    else:
        fdb_high=df_far_table_maker('high',regions_list)
        pdb_high=df_pod_table_maker('high',regions_list)
        return fdb_high,pdb_high
    