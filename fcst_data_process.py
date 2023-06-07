#regridding and save as forecast months
#add total precentipation per month value 
#

from data_process_utils import foldercreator
from data_process_utils import seas5_grib_processor
from data_process_utils import seas5_regridder
from data_process_utils import seas5_tpm_creator
from data_process_utils import lead_month_wise_df_create

output_path_location='output/'

foldercreator(output_path_location)


input_path_location='data/'

grib_array=seas5_grib_processor(input_path_location)

seas5_regridder(output_path_location,grib_array)

seas5_tpm_creator(output_path_location)

lead_month_wise_df_create(output_path_location)


