import os
from dotenv import load_dotenv

from vthree_utils import BinCreateParams

from utils_plots import run_heatmap_plot

load_dotenv()

params = BinCreateParams(
    region_id=0,
    region_name_dict={0: "Karamoja", 1: "Marsabit", 2: "Wajir"},
    season_str="MAM",
    lead_int=2,
    level="mod",
    spi_prod_name="spi3",
    data_path=os.getenv("ea_input_path"),
    spi4_data_path=os.getenv("data_path"),
    output_path=os.getenv("output_path"),
)

run_heatmap_plot(params)

params.season_str = "JJAS"
params.sc_season_str = "jjas"
params.spi_prod_name = "spi4"
params.data_path = params.spi4_data_path

run_heatmap_plot(params)
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

run_heatmap_plot(params)

params.season_str = "OND"
params.sc_season_str = "ond"

run_heatmap_plot(params)

##########################
##########################

params.season_str = "MAM"
params.sc_season_str = "mam"
params.spi_prod_name = "spi3"
params.region_id = 2

run_heatmap_plot(params)

params.season_str = "OND"
params.sc_season_str = "ond"
params.spi_prod_name = "spi3"
run_heatmap_plot(params)
