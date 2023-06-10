#regridding and save as forecast months
#add total precentipation per month value 
#

from data_process_utils import foldercreator
from data_process_utils import seas5_grib_processor
from data_process_utils import seas5_regridder
from data_process_utils import seas5_tpm_creator
from data_process_utils import lead_month_wise_df_create

from data_process_utils import three_months_spi_creator
from data_process_utils import four_months_spi_creator
from data_process_utils import six_months_spi_creator
#######
from utils import spi3_prob_ncfile_creator
from utils import spi4_prob_ncfile_creator
from utils import spi6_prob_ncfile_creator_a
from utils import spi6_prob_ncfile_creator_b

from utils import prob_exceed_year_plot

output_path_location='output/'

foldercreator(output_path_location)


input_path_location='data/'

grib_array=seas5_grib_processor(input_path_location)

seas5_regridder(grib_array,output_path_location)

seas5_tpm_creator(output_path_location)

lead_month_wise_df_create(output_path_location)


three_months_spi_creator(output_path_location)

four_months_spi_creator(output_path_location)

six_months_spi_creator(output_path_location)


###########
###########
output_path='output/prob_at/'

spi3_prob_ncfile_creator(output_path)
spi4_prob_ncfile_creator(output_path)
spi6_prob_ncfile_creator_a(output_path)
spi6_prob_ncfile_creator_b(output_path)


ncfile_path='output/prob_at/'
spi_prod='mam'
lt_month='jan'
prob_exceed_year_plot(ncfile_path,spi_prod,lt_month)