import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import altair as alt
from io import StringIO
import xarray as xr

from dask.distributed import Client
from utils import get_threshold

from vthree_utils import BinCreateParams
from vthree_utils import ken_mask_creator
from vthree_utils import make_obs_fct_dataset
from vthree_utils import get_threshold
from vthree_utils import seas51_patch_empirical_probability

# from utils import prepare_data_for_concat
from utils_plots import helper_stamp_plot
from utils_plots import plot_allrows
from utils_plots import merge_png_files
from utils_plots import run_map_plot

load_dotenv()

params = BinCreateParams(
    region_id=0,
    season_str="MAM",
    lead_int=2,
    level="mod",
    region_name_dict={0: "Karamjoa", 1: "Marsabit", 2: "Wajir"},
    spi_prod_name="spi3",
    data_path=os.getenv("ea_input_path"),
    spi4_data_path=os.getenv("data_path"),
    output_path=os.getenv("output_path"),
)


# params.data_path = params.spi4_data_path

threshold_dict = get_threshold(params.region_id, params.sc_season_str)


obs_data, ens_data = make_obs_fct_dataset(
    params.data_path, params.region_id, params.season_str, params.lead_int
)
fct_mod, fct_sev, fct_ext = seas51_patch_empirical_probability(ens_data, threshold_dict)

dstree = helper_stamp_plot(ens_data, obs_data, fct_mod, fct_sev, fct_ext)
plot_allrows(dstree, params)

merge_png_files(
    input_dir=f"{params.output_path}map_{params.region_id}_{params.sc_season_str}_lt{params.lead_int}",
    output_file="merged_stamp_plots1.png",
    delete_originals=False,
)
