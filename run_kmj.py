import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import altair as alt
from io import StringIO
import xarray as xr
from sklearn.metrics import roc_auc_score
from xbootstrap import block_bootstrap
import xskillscore as xs


from dask.distributed import Client
from vthree_utils import get_threshold

from vthree_utils import ken_mask_creator
from vthree_utils import make_obs_fct_dataset
from vthree_utils import get_threshold
from vthree_utils import seas51_patch_empirical_probability
from vthree_utils import xhist_metrics_2d
from vthree_utils import BinCreateParams
from vthree_utils import run_xhist2d
from vthree_utils import run_xhist1d


load_dotenv()

params = BinCreateParams(
    region_id=0,
    season_str="MAM",
    lead_int=2,
    level="mod",
    spi_prod_name="spi3",
    data_path=os.getenv("ea_input_path"),
    spi4_data_path=os.getenv("data_path"),
    output_path=os.getenv("output_path"),
)

# run_xhist2d(params)
run_xhist1d(params)


params.lead_int = 3
# run_xhist2d(params)
run_xhist1d(params)

params.lead_int = 4
# run_xhist2d(params)
run_xhist1d(params)

params.season_str = "JJAS"
params.sc_season_str = "jjas"
params.spi_prod_name = "spi4"
params.data_path = params.spi4_data_path
params.lead_int = 2
# run_xhist2d(params)
run_xhist1d(params)


params.lead_int = 3
# run_xhist2d(params)
run_xhist1d(params)

params.lead_int = 4
# run_xhist2d(params)
run_xhist1d(params)
