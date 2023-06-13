#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 08:30:16 2023

@author: bulbul
"""

from utils_tables import metric_db
from utils_tables import prob_db
import pandas as pd
from utils_tables import thres_db_maker

from utils_tables import far_pod_table_all
# mdb=metric_db()

# pdb=prob_db()

# dfg=pd.merge(pdb,mdb,on='prod')

# dfg.to_csv('output/tables/all_metrices_thres.csv',index=False)

#thres_db_maker()
thre='low'
fdb_low,pdb_low=far_pod_table_all(thre)

thre='mid'
fdb_mid,pdb_mid=far_pod_table_all(thre)

thre='high'
fdb_high,pdb_high=far_pod_table_all(thre)

