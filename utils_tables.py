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
from functools import reduce
import json
#%% table plot matplotlib

### Define the picture size and remove the ticks

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
    allcells=[(x,y) for x in tablerows[1:] for y in tablecols[2:]]
    for alcls in allcells:
        cell_value0=json.loads(cellDict[alcls]._text.get_text())[0]
        if cell_value0==-999.0:
            cellDict[alcls].set_facecolor('#FFFFFF')
        else:
            if float(cell_value0) <=0.2:
                cellDict[alcls].set_facecolor('#009600')
            elif 0.2 < float(cell_value0) <= 0.4:
                cellDict[alcls].set_facecolor('#64C800')
            elif 0.4 < float(cell_value0) <= 0.6:
                cellDict[alcls].set_facecolor('#ffff00')
            elif 0.6 < float(cell_value0) <= 0.8:
                cellDict[alcls].set_facecolor('#ff7800')
            elif 0.8 < float(cell_value0) <= 1.0:
                cellDict[alcls].set_facecolor('#ff0000')
            else:
                cellDict[alcls].set_facecolor('#FFFFFF')


def remove_value(tablerows,tablecols,mpl_table):
    allcells=[(x,y) for x in tablerows[1:] for y in tablecols[2:]]
    for alcls in allcells:
        mpl_table._cells[alcls]._text.set_text('')
        
def add_certain_value(tablerows,tablecols,mpl_table,cellDict):
    allcells=[(x,y) for x in tablerows[1:] for y in tablecols[2:]]
    for alcls in allcells:
        #print(cellDict[alcls]._text.get_text())
        cell_value0=json.loads(cellDict[alcls]._text.get_text())[1]
        mpl_table._cells[alcls]._text.set_text('')
        #cell_value0=(cellDict[alcls]._text.get_text())
        if cell_value0==-999.0:
            mpl_table._cells[alcls]._text.set_text('')
        elif cell_value0==999.0:
            mpl_table._cells[alcls]._text.set_text('')
        else:
            ncl="%.1f" % cell_value0
            mpl_table._cells[alcls]._text.set_text(ncl)

def set_height_for_row_except_head(table, rowlist, height):
    cells_list=[]
    for row in rowlist:
        cells = [key for key in table._cells if key[0] == row]
        cells_list.append(cells)
    for cells in cells_list:
        for cell in cells:
             table._cells[cell]._height = height
                
                
def table_header_colour(tablerows,tablecols,cellDict,mpl_table):
    allcells=[(x,y) for x in tablerows[0:1] for y in tablecols]
    header_list=['SPI','District','Nov','Dec','Jan','Feb','Mar','Apr','May','',
                 'Nov','Dec','Jan','Feb','Mar','Apr','May','','Nov','Dec','Jan','Feb','Mar','Apr','May','']
    for idx,alcls in enumerate(allcells):
        cellDict[alcls].set_facecolor('#FFFFFF')
        print(header_list[idx])
        text=header_list[idx]
        mpl_table._cells[alcls]._text.set_text('text')
        
    

### funciton for table creation
def render_mpl_table(data, col_width=1.0, row_height=1.625, font_size=8,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=['']*25,cellLoc='center', **kwargs)
    set_align_for_column(mpl_table, col=0, align="left")
    set_width_for_column(mpl_table, 0, 0.3)
    set_width_for_column(mpl_table, 1, 0.5)
    for idx in range(2,26):
        set_width_for_column(mpl_table, idx, 0.2)
    set_height_for_row(mpl_table, 0, 0.01)
    set_height_for_row_except_head(mpl_table, np.arange(1,len(data.index)), 0.012)
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)
    cellDict=mpl_table.get_celld()
    tablerows=np.arange(0,len(data.index)+1)
    tablecols=np.arange(0,len(data.columns))
    for k, cell in  six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    colorcell(tablerows,tablecols,cellDict)
    headings=data.columns
    table_header_colour(tablerows,tablecols,cellDict,mpl_table)
    add_certain_value(tablerows,tablecols,mpl_table,cellDict)
    return ax
    

def plot_data_table(data_table):
    fig = plt.figure()
    fig.set_size_inches(18,18)
    table=fig.add_axes([0.08, 0.02, 0.55, 0.9], frame_on=False) 
    table.xaxis.set_ticks_position('none')
    table.yaxis.set_ticks_position('none') 
    table.set_xticklabels('')
    table.set_yticklabels('')
    req_list=['spi_prod_x', 'region_x', 'nov_x', 'dec_x', 'jan_x', 'feb_x', 'mar_x',
           'apr_x', 'may_x','empty1', 'nov_y', 'dec_y','jan_y', 'feb_y', 'mar_y', 'apr_y', 'may_y',
            'empty2','nov', 'dec', 'jan', 'feb', 'mar', 'apr', 'may']
    
    data1=data_table[req_list]
    render_mpl_table(data1, header_columns=0, col_width=0.2,ax=table)
    plt.savefig('tables/far_prob.jpg', dpi=150, alpha=True)

    
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
    
    
def table_plot_df_maker(thre,rdict,sdict):
    db=pd.read_csv(f'output/tables/final_db_{thre}.csv')
    dbv=db.pivot_table(index=['spi_prod','region'], columns=['lt_month'], values=['prob','FAR',])
    dbv1=dbv.replace(np.nan,-999)
    db2 = dbv1.reorder_levels([1, 0], axis=1)
    #https://stackoverflow.com/a/67818894/2501953
    db2['nov_fp'] = db2["nov"].values.tolist()
    #https://stackoverflow.com/a/43898233/2501953
    db2['dec_fp'] = db2["dec"].values.tolist()
    db2['jan_fp'] = db2["jan"].values.tolist()
    db2['feb_fp'] = db2["feb"].values.tolist()
    db2['mar_fp'] = db2["mar"].values.tolist()
    db2['apr_fp'] = db2["apr"].values.tolist()
    db2['may_fp'] = db2["may"].values.tolist()
    db3=db2[['nov_fp','dec_fp','jan_fp','feb_fp','mar_fp','apr_fp','may_fp']]
    #db4=db3['region'].sort_values()
    db4=db3.reset_index()
    db5=db4.sort_values('region')
    db5['r_order'] = db5['region'].map(rdict)
    db5['s_order'] = db5['spi_prod'].map(sdict)
    db6=db5.sort_values(['r_order','s_order'])
    db6.columns = db6.columns.droplevel(1)
    db6 = db6.rename_axis(None, axis=1)
    db7=db6.reset_index()
    db8=db7[['spi_prod','region','nov_fp','dec_fp','jan_fp','feb_fp','mar_fp','apr_fp','may_fp']]
    db8.columns=['spi_prod', 'region', 'nov', 'dec', 'jan', 'feb', 'mar','apr', 'may']
    db8['idx']=db8['spi_prod']+'_'+db8['region']
    return db8


def thre_df_table_plot():
    regions_list=['Abim','Napak','Nabilatuk','Kotido', 
                  'Moroto','Nakapiripirit', 'Kaabong', 'Karenga',
                  'Amudat','Karamoja']
    rorder_list=[1,2,3,4,5,6,7,8,9,0]
    spi_prodlist=['mam','jjas','mamjja','amjjas']
    rorder_splist=[0,1,2,3]
    sdict=dict(zip(spi_prodlist, rorder_splist))
    rdict = dict(zip(regions_list, rorder_list))
    adb=table_plot_df_maker('low',rdict,sdict)
    bdb=table_plot_df_maker('mid',rdict,sdict)
    cdb=table_plot_df_maker('high',rdict,sdict)
    data_frames = [adb, bdb, cdb]
    df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['idx'],
                                            how='outer'), data_frames)
    df_merged['empty1']='[999.0,999.0]'
    df_merged['empty2']='[999.0,999.0]'
    req_list=['spi_prod_x', 'region_x', 'nov_x', 'dec_x', 'jan_x', 'feb_x', 'mar_x',
       'apr_x', 'may_x','empty1', 'nov_y', 'dec_y','jan_y', 'feb_y', 'mar_y', 'apr_y', 'may_y',
        'empty2','nov', 'dec', 'jan', 'feb', 'mar', 'apr', 'may']
    df_merged1=df_merged[req_list]
    return df_merged1