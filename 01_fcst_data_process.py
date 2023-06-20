#regridding and save as forecast months
#add total precentipation per month value 
#

from utils_data_process import foldercreator
from utils_data_process import chrips_data_regridder
from utils_data_process import chrips_spi_mam_creator
from utils_data_process import chrips_spi_jjas_creator
from utils_data_process import chrips_spi_mamjja_creator
from utils_data_process import chrips_spi_amjjas_creator

########

from utils_data_process import seas5_grib_processor
from utils_data_process import seas5_regridder
from utils_data_process import seas5_tpm_creator
from utils_data_process import lead_month_wise_df_create

from utils_data_process import three_months_spi_creator
from utils_data_process import four_months_spi_creator
from utils_data_process import six_months_spi_creator

######
from utils_data_process import spi3_prob_ncfile_creator
from utils_data_process import spi4_prob_ncfile_creator
from utils_data_process import spi6_prob_ncfile_creator_a
from utils_data_process import spi6_prob_ncfile_creator_b

#from utils import kmj_mask_creator
#from utils import prob_exceed_year_plot

from utils_data_process import spi3_mean_ncfile_creator
from utils_data_process import spi4_mean_ncfile_creator
from utils_data_process import spi6_mean_ncfile_creator_a
from utils_data_process import spi6_mean_ncfile_creator_b





output_path_location='output/'

foldercreator(output_path_location)


chrips_data_regridder()

chrips_spi_mam_creator()    
chrips_spi_jjas_creator()
chrips_spi_mamjja_creator()
chrips_spi_amjjas_creator()

#########
#########

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
output_path='output/prob/'

spi3_prob_ncfile_creator(output_path)
spi4_prob_ncfile_creator(output_path)
spi6_prob_ncfile_creator_a(output_path)
spi6_prob_ncfile_creator_b(output_path)
###########
###########
###########
output_path='output/mean_spi/'

spi3_mean_ncfile_creator(output_path)
spi4_mean_ncfile_creator(output_path)
spi6_mean_ncfile_creator_a(output_path)
spi6_mean_ncfile_creator_b(output_path)


