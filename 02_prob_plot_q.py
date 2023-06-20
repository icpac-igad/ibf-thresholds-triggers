

from utils import kmj_mask_creator
from utils import prob_exceed_year_plot

from utils import plot_prob


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
plot_prob(spi_prod,lt_month)

spi_prod='mam'
lt_month='dec'
plot_prob(spi_prod,lt_month)

spi_prod='mam'
lt_month='jan'
plot_prob(spi_prod,lt_month)

spi_prod='mam'
lt_month='feb'
plot_prob(spi_prod,lt_month)

spi_prod='jjas'
lt_month='mar'
plot_prob(spi_prod,lt_month)

spi_prod='jjas'
lt_month='apr'
plot_prob(spi_prod,lt_month)

spi_prod='jjas'
lt_month='may'
plot_prob(spi_prod,lt_month)

spi_prod='mamjja'
lt_month='feb'
plot_prob(spi_prod,lt_month)

spi_prod='amjjas'
lt_month='mar'
plot_prob(spi_prod,lt_month)
