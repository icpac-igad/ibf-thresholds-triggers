


from utils import prob_exceed_csv

from utils import csv_prob


# the_mask, rl_dict=kmj_mask_creator()

# ncfile_path='output/prob/'
# spi_prod='mam'
# lt_month='jan'
# region_idx=9
# ridx_list=[0,1,2,3,4,5,6,7,8,9]

# for ridx in ridx_list:
#     prob_exceed_year_plot(ncfile_path,spi_prod,lt_month,the_mask,ridx,rl_dict)

spi_prod='mam'
lt_month='nov'
csv_prob(spi_prod,lt_month)

spi_prod='mam'
lt_month='dec'
csv_prob(spi_prod,lt_month)

spi_prod='mam'
lt_month='jan'
csv_prob(spi_prod,lt_month)

spi_prod='mam'
lt_month='feb'
csv_prob(spi_prod,lt_month)

spi_prod='jjas'
lt_month='mar'
csv_prob(spi_prod,lt_month)

spi_prod='jjas'
lt_month='apr'
csv_prob(spi_prod,lt_month)

spi_prod='jjas'
lt_month='may'
csv_prob(spi_prod,lt_month)

spi_prod='mamjja'
lt_month='feb'
csv_prob(spi_prod,lt_month)

spi_prod='amjjas'
lt_month='mar'
csv_prob(spi_prod,lt_month)
