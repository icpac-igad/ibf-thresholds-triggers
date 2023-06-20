
from utils import metrics_csv


# the_mask, rl_dict=kmj_mask_creator()

# ncfile_path='output/prob/'
# spi_prod='mam'
# lt_month='jan'
# region_idx=9
# ridx_list=[0,1,2,3,4,5,6,7,8,9]

# for ridx in ridx_list:
#     prob_exceed_year_plot(ncfile_path,spi_prod,lt_month,the_mask,ridx,rl_dict)

# spi_prod='mam'
# lt_month='nov'
# thr_val_list=[-0.03,-0.56,-0.99]
# metrics_csv(spi_prod, lt_month, thr_val_list)

# spi_prod='mam'
# lt_month='dec'
# thr_val_list=[-0.03,-0.56,-0.99]
# metrics_csv(spi_prod, lt_month, thr_val_list)

# spi_prod='mam'
# lt_month='jan'
# thr_val_list=[-0.03,-0.56,-0.99]
# metrics_csv(spi_prod, lt_month, thr_val_list)

# spi_prod='mam'
# lt_month='feb'
# thr_val_list=[-0.03,-0.56,-0.99]
# metrics_csv(spi_prod, lt_month, thr_val_list)

# spi_prod='jjas'
# lt_month='mar'
# thr_val_list=[-0.01,-0.41,-0.99]
# metrics_csv(spi_prod, lt_month, thr_val_list)

# spi_prod='jjas'
# lt_month='apr'
# thr_val_list=[-0.01,-0.41,-0.99]
# metrics_csv(spi_prod, lt_month, thr_val_list)

spi_prod='jjas'
lt_month='may'
thr_val_list=[-0.01,-0.41,-0.99]
metrics_csv(spi_prod, lt_month, thr_val_list)

# spi_prod='mamjja'
# lt_month='feb'
# thr_val_list=[-0.02,-0.38,-1.01]
# metrics_csv(spi_prod, lt_month, thr_val_list)

# spi_prod='amjjas'
# lt_month='mar'
# thr_val_list=[-0.02,-0.38,-1.01]
# metrics_csv(spi_prod, lt_month, thr_val_list)
