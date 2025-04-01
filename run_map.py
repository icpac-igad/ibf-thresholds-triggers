import os
from dotenv import load_dotenv

from vthree_utils import BinCreateParams
from utils_plots import run_map_plot

load_dotenv()
ea_input_path = os.getenv("ea_input_path")
sa_file = os.getenv("sa_file")
polygon_pq_uri = os.getenv("polygon_pq_uri")


params = BinCreateParams(
    region_id=0,
    season_str="MAM",
    lead_int=2,
    level="mod",
    region_name_dict={0: "Karamoja", 1: "Marsabit", 2: "Wajir"}, # this will be populated later
    spi_prod_name="spi3",
    data_path=ea_input_path,
    output_path=os.path.join(os.getcwd(), "output"),
    spi4_data_path="",
    obs_netcdf_file=os.path.join(ea_input_path, "kn_obs_spi3_20240717.nc"),
    fct_netcdf_file=os.path.join(ea_input_path, "kn_fct_spi3_20240717.nc"),
    service_account_json=sa_file,
    gcs_file_url=polygon_pq_uri,
    region_filter="kmj",
)

# run_xhist2d(params)
run_map_plot(params)

"""
params.lead_int = 3
# run_xhist2d(params)
run_map_plot(params)

params.lead_int = 4
# run_xhist2d(params)
run_map_plot(params)

params.season_str = "JJA"
params.sc_season_str = "jja"
params.spi_prod_name = "spi3"
#params.data_path = params.spi4_data_path
params.lead_int = 2
# run_xhist2d(params)
run_map_plot(params)


params.lead_int = 3
run_map_plot(params)

params.lead_int = 4
run_map_plot(params)


#########################
#########################
params = BinCreateParams(
    region_id=1,
    region_name_dict={0: "Karamoja", 1: "Marsabit", 2: "Wajir"},
    season_str="MAM",
    lead_int=2,
    level="mod",
    spi_prod_name="spi3",
    data_path=os.getenv("ea_input_path"),
    spi4_data_path=os.getenv("data_path"),
    output_path=os.getenv("output_path"),
)

run_map_plot(params)

params.lead_int = 3
run_map_plot(params)

params.lead_int = 4
run_map_plot(params)

params.season_str = "OND"
params.sc_season_str = "ond"
params.lead_int = 2
run_map_plot(params)


params.lead_int = 3
run_map_plot(params)

params.lead_int = 4
run_map_plot(params)

##########################
##########################

params.season_str = "MAM"
params.sc_season_str = "mam"
params.spi_prod_name = "spi3"
params.region_id = 2
params.lead_int = 2
run_map_plot(params)


params.lead_int = 3
run_map_plot(params)

params.lead_int = 4
run_map_plot(params)

params.season_str = "OND"
params.sc_season_str = "ond"
params.spi_prod_name = "spi3"
params.lead_int = 2
run_map_plot(params)"""


# params.lead_int = 3
# run_map_plot(params)

# params.lead_int = 4
# run_map_plot(params)
