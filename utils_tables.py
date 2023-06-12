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


def removecellvalue(tablerows,tablecols,mpl_table):
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