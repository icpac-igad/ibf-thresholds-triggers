import os
from dotenv import load_dotenv

from vthree_utils import BinCreateParams

from utils_plots import run_bar_plot

load_dotenv()

# Get environment variables
ea_input_path = './'
#ea_input_path = os.getenv("ea_input_path")
sa_file = os.getenv("sa_file")
polygon_pq_uri = os.getenv("polygon_pq_uri")


# Now create the BinCreateParams object
params = BinCreateParams(
    region_id=0,
    season_str="MAM",
    lead_int=1,
    level="mod",
    region_name_dict={0: "Karamoja", 1: "Marsabit", 2: "Wajir"},
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

#run_bar_plot(params)

params.season_str = "JJA"
params.sc_season_str = "jja"
params.spi_prod_name = "spi3"
#params.data_path = params.spi4_data_path

run_bar_plot(params)

