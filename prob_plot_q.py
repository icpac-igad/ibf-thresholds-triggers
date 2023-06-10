from utils import kmj_mask_creator
from utils import prob_exceed_year_plot


the_mask, rl_dict=kmj_mask_creator()

ncfile_path='output/prob/'
spi_prod='mam'
lt_month='jan'
region_idx=9
ridx_list=[0,1,2,3,4,5,6,7,8,9]

for ridx in ridx_list:
    prob_exceed_year_plot(ncfile_path,spi_prod,lt_month,the_mask,ridx,rl_dict)
