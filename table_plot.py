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


# mdb=metric_db()

# pdb=prob_db()

# dfg=pd.merge(pdb,mdb,on='prod')

# dfg.to_csv('output/tables/all_metrices_thres.csv',index=False)

thres_db_maker()


