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

from vthree_utils import BinCreateParams
from vthree_utils import get_threshold
from vthree_utils import generate_trigger_dict
from vthree_utils import run_bar_plot_df

from utils_plots import aux_plot_make_barchart_annotation
from utils_plots import plot_obs_chart_with_triggers
from utils_plots import bar_stitch_plot

load_dotenv()

params = BinCreateParams(
    region_id=0,
    season_str="JJAS",
    lead_int=2,
    level="mod",
    spi_prod_name="spi4",
    data_path=os.getenv("ea_input_path"),
    spi4_data_path=os.getenv("data_path"),
    output_path=os.getenv("output_path"),
)
params.data_path = params.spi4_data_path

#######################
threshold_dict = get_threshold(params.region_id, params.sc_season_str)
obs_df, plot_dflt2 = run_bar_plot_df(params, is_obs_df=True)

# run_xhist2d(params)
row_annotation = aux_plot_make_barchart_annotation(params)
decision_dict, dec_dflt2 = generate_trigger_dict(params)

obs_plot = plot_obs_chart_with_triggers(
    "obs", obs_df, "year", "spi3", threshold_dict, row_annotation
)
lt2_plot = plot_obs_chart_with_triggers(
    "fct", plot_dflt2, "year", "ep_pb", decision_dict, row_annotation
)
#######################
params.lead_int = 3
plot_dflt3 = run_bar_plot_df(params, is_obs_df=False)

# run_xhist2d(params)
row_annotation = aux_plot_make_barchart_annotation(params)
decision_dict, dec_dflt3 = generate_trigger_dict(params)

lt3_plot = plot_obs_chart_with_triggers(
    "fct", plot_dflt2, "year", "ep_pb", decision_dict, row_annotation
)
#######################
#######################
params.lead_int = 4
plot_dflt4 = run_bar_plot_df(params, is_obs_df=False)

# run_xhist2d(params)
row_annotation = aux_plot_make_barchart_annotation(params)
decision_dict, dec_dflt4 = generate_trigger_dict(params)

lt4_plot = plot_obs_chart_with_triggers(
    "fct", plot_dflt4, "year", "ep_pb", decision_dict, row_annotation
)
#######################

bar_stitch_config = {
    "obs_plot": obs_plot,
    "lt2_plot": lt2_plot,
    "lt3_plot": lt3_plot,
    "lt4_plot": lt4_plot,
    "dec_dflt2": dec_dflt2,
    "dec_dflt3": dec_dflt3,
    "dec_dflt4": dec_dflt4,
}

bar_stitch_plot(params, bar_stitch_config)


"""
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
run_xhist1d(params)"""
