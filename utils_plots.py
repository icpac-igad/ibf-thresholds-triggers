from io import StringIO
import os
from dotenv import load_dotenv
import logging
from pathlib import Path

import climpred
from sqlalchemy import False_
import xarray as xr
import xesmf as xe
import numpy as np
import pandas as pd
import regionmask
import geopandas as gp
from climpred import HindcastEnsemble
from datetime import datetime
from datatree import DataTree

import xhistogram.xarray as xhist
from sklearn.metrics import roc_auc_score

import xskillscore as xs
from xbootstrap import block_bootstrap
from dask.distributed import Client

# matplotlib.use("Agg")
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm
from datetime import datetime
from dateutil.relativedelta import relativedelta
from calendar import month_abbr
import cartopy.crs as ccrs
import six
import textwrap as tw
from functools import reduce
import json
from dateutil.relativedelta import relativedelta
from calendar import monthrange
from PIL import Image


from vthree_utils import ken_mask_creator
from vthree_utils import make_obs_fct_dataset
from vthree_utils import get_threshold
from vthree_utils import seas51_patch_empirical_probability

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_subset(dfa, cat_str):
    # Filter out rows with null values in 'hit_rate' and 'false_alarm_ratio'
    # df = df.dropna(subset=['hit_rate', 'false_alarm_ratio'])
    df = dfa[dfa["cat"] == cat_str]
    # Sort the DataFrame by 'peirce_score' in descending order
    df = df.sort_values(by="hanssen_kuipers_scores", ascending=False)

    # Get the row with the maximum 'peirce_score'
    max_peirce_row = df.iloc[0]

    # Sort the DataFrame by 'bias_score' in descending order, and filter for 'bias_score' < 1.0
    df = df.loc[df["bias_scores"] < 1.0].sort_values(by="bias_scores", ascending=False)

    # Get the row with the maximum 'bias_score' < 1.0
    max_bias_row = df.iloc[0]

    # Sort the DataFrame by 'heidke_score' in descending order
    df = df.sort_values(by="heidke_skill_scores", ascending=False)

    # Get the row with the maximum 'heidke_score'
    max_heidke_row = df.iloc[0]

    # Combine the three rows into a subset
    subset = pd.concat(
        [
            pd.DataFrame([max_peirce_row]),
            pd.DataFrame([max_bias_row]),
            pd.DataFrame([max_heidke_row]),
        ],
        ignore_index=True,
    )

    return subset


def trigger_decision_dict(df0):
    df = df0[df0["auroc_scores"] >= 0.5]
    df_mod = get_subset(df, "mod")
    mod_max_cn = df_mod["CN"].max()
    mod_df_max_cn = df_mod[df_mod["CN"] == mod_max_cn]
    mod_max_hits = mod_df_max_cn["hits"].max()
    mod_df_max_hits = mod_df_max_cn[mod_df_max_cn["hits"] == mod_max_hits]

    df_sev = get_subset(df, "sev")
    sev_max_cn = df_sev["CN"].max()
    sev_df_max_cn = df_sev[df_sev["CN"] == sev_max_cn]
    sev_max_hits = sev_df_max_cn["hits"].max()
    sev_df_max_hits = sev_df_max_cn[sev_df_max_cn["hits"] == sev_max_hits]

    df_ext = get_subset(df, "ext")
    ext_max_cn = df_ext["CN"].max()
    ext_df_max_cn = df_ext[df_ext["CN"] == ext_max_cn]
    ext_max_hits = ext_df_max_cn["hits"].max()
    ext_df_max_hits = ext_df_max_cn[ext_df_max_cn["hits"] == ext_max_hits]
    tri_dict = {
        "mod": mod_df_max_hits["trigger_values"].values[0],
        "sev": sev_df_max_hits["trigger_values"].values[0],
        "ext": ext_df_max_hits["trigger_values"].values[0],
    }
    df0 = pd.concat([mod_df_max_hits, sev_df_max_hits, ext_df_max_hits])
    return tri_dict, df0


def helper_stamp_plot(ens_data, obs_data, fct_mod, fct_sev, fct_ext):
    """

    DEPRECATED to replaced it with xarray datatree
    Helper function to prepare and combine data for stamp plot.

    Parameters:
    ens_data (xarray.Dataset): Ensemble data.
    obs_data (xarray.Dataset): Observation data.
    fct_mod (xarray.Dataset): Moderate forecast data.
    fct_sev (xarray.Dataset): Severe forecast data.
    fct_ext (xarray.Dataset): Extreme forecast data.

    Returns:
    xarray.Dataset: Combined dataset for stamp plot.
    """
    try:
        logger.info("Starting datatree with helper_stamp_plot function")
        seas51tree = DataTree()
        for member in ens_data.member:
            member_data = ens_data.sel(member=member)
            seas51tree[f"ensemble/member_{int(member)}"] = DataTree(
                name=f"member_{int(member)}", data=member_data
            )

        seas51tree["observation"] = DataTree(name="observation", data=obs_data)
        seas51tree["fct_mod"] = DataTree(name="fct_mod", data=fct_mod)
        seas51tree["fct_sev"] = DataTree(name="fct_sev", data=fct_sev)
        seas51tree["fct_ext"] = DataTree(name="fct_ext", data=fct_ext)
        logger.info(f"made the combined_data as xarray datatree {seas51tree}")
        logger.info("helper_stamp_plot function completed successfully")
        return seas51tree

    except KeyError as e:
        logger.error(f"KeyError in helper_stamp_plot: {str(e)}")
        raise
    except ValueError as e:
        logger.error(f"ValueError in helper_stamp_plot: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in helper_stamp_plot: {str(e)}")
        raise


def create_single_row_plot(tree, init, params, region_geom):
    logging.info(
        f"Creating single row plot for {init} with variable {params.spi_prod_name}"
    )
    try:

        members = list(tree["ensemble"].children.keys())
        valid_times = tree["ensemble/member_0"].ds.valid_time.values
        lats = tree["ensemble/member_0"].ds.lat.values
        lons = tree["ensemble/member_0"].ds.lon.values
        num_members = len(members)
        num_additional_plots = 4  # Obs, mod, sev, ext
        total_plots = num_members + num_additional_plots
        cbar_figsize = (2 * total_plots, 2)
        # Create the figure and axes for plotting
        fig, axs = plt.subplots(
            1,
            total_plots,
            figsize=(2 * total_plots, 2),
            subplot_kw={"projection": ccrs.PlateCarree()},
        )

        ensemble_cmap_range = (-4, 4)
        fct_cmap_range = (0.0, 1.0)

        valid_time = valid_times[
            np.where(tree["ensemble/member_0"].ds.init.values == init)[0][0]
        ]

        # Plot ensemble members
        for j, member_key in enumerate(members):
            overlay_shapefile = (
                j == 0
            )  # Only overlay shapefile for the first member (member_0)
            _plot_ensemble_member(
                tree,
                member_key,
                params.spi_prod_name,
                init,
                axs[j],
                ensemble_cmap_range,
                region_geom,
                shape_overlay=True if overlay_shapefile else False,
            )

        # Add the observation and additional models as the last plots
        plot_titles = ["Obs", "mod", "sev", "ext"]
        plot_keys = ["observation", "fct_mod", "fct_sev", "fct_ext"]
        for k, (title, key) in enumerate(zip(plot_titles, plot_keys)):
            _plot_additional_data(
                tree,
                key,
                params.spi_prod_name,
                init,
                valid_time,
                axs[num_members + k],
                title,
                ensemble_cmap_range if key == "observation" else fct_cmap_range,
            )

        # Instead of saving the plot, return the figure and axes for further modifications
        return fig, axs, cbar_figsize

    except Exception as e:
        logging.error(f"Error creating plot: {str(e)}")
        raise


def ocreate_single_row_plot(tree, init, params, region_geom):
    logging.info(f"Creating single row plot for {init} with variable {variable}")
    try:
        output_dir = f"{params.output_path}map_region{params.region_id}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Folder created: {output_dir}")
        else:
            print(f"Folder already exists: {output_dir}")
        members = list(tree["ensemble"].children.keys())
        valid_times = tree["ensemble/member_0"].ds.valid_time.values
        lats = tree["ensemble/member_0"].ds.lat.values
        lons = tree["ensemble/member_0"].ds.lon.values
        num_members = len(members)
        num_additional_plots = 4  # Obs, mod, sev, ext
        total_plots = num_members + num_additional_plots

        # Create output directory if it doesn't exist

        # Create a wide figure for a single row
        fig, axs = plt.subplots(
            1,
            total_plots,
            figsize=(2 * total_plots, 2),
            subplot_kw={"projection": ccrs.PlateCarree()},
        )

        # Define the color scale ranges
        ensemble_cmap_range = (-4, 4)
        fct_cmap_range = (0.0, 1.0)

        valid_time = valid_times[
            np.where(tree["ensemble/member_0"].ds.init.values == init)[0][0]
        ]

        # Plot ensemble members
        for j, member_key in enumerate(members):
            overlay_shapefile = j == 0
            _plot_ensemble_member(
                tree,
                member_key,
                params.spi_prod_name,
                init,
                axs[0, j],
                ensemble_cmap_range,
                region_geom,
                shape_overlay=True if overlay_shapefile else False,
            )

        # Add the observation and additional models as the last plots
        plot_titles = ["Obs", "mod", "sev", "ext"]
        plot_keys = ["observation", "fct_mod", "fct_sev", "fct_ext"]
        for k, (title, key) in enumerate(zip(plot_titles, plot_keys)):
            _plot_additional_data(
                tree,
                key,
                params.spi_prod_name,
                init,
                valid_time,
                axs[num_members + k],
                title,
                ensemble_cmap_range if key == "observation" else fct_cmap_range,
            )

        plt.tight_layout()
        output_file = f'{output_dir}/stamp_plot_{init.strftime("%Y%m%d")}.png'
        plt.savefig(output_file, dpi=100, bbox_inches="tight")
        logging.info(f"Plot saved to {output_file}")
        plt.close()

    except Exception as e:
        logging.error(f"Error creating plot: {str(e)}")
        raise


def _plot_ensemble_member(
    tree, member_key, variable, init, ax, cmap_range, geom, shape_overlay=False
):
    member_data = tree[f"ensemble/{member_key}"].ds[variable]
    data = member_data.sel(init=init).values
    time_np64 = np.array(str(init), dtype="datetime64[ns]")
    year = pd.to_datetime(time_np64).year
    ax.pcolormesh(
        tree["ensemble/member_0"].ds.lon.values,
        tree["ensemble/member_0"].ds.lat.values,
        data,
        cmap="RdBu",
        transform=ccrs.PlateCarree(),
        vmin=cmap_range[0],
        vmax=cmap_range[1],
    )
    ax.set_title(f'{year}m{member_key.split("_")[1]}', fontsize=6)
    ax.set_xticks([])
    ax.set_yticks([])

    # If shapefile is provided, overlay it
    if shape_overlay:
        ax.add_geometries(
            [geom], crs=ccrs.PlateCarree(), edgecolor="black", facecolor="none"
        )


def _plot_additional_data(tree, key, variable, init, valid_time, ax, title, cmap_range):
    dataset = tree[key].ds[variable]
    coord_key = "time" if "time" in dataset.coords else "init"
    obs_init = (
        np.datetime64(valid_time.strftime("%Y-%m-%d %H:%M:%S"))
        if coord_key == "time"
        else init
    )

    obs_data = dataset.sel({coord_key: obs_init}).values

    cmap = "RdBu" if key == "observation" else "Blues"

    ax.pcolormesh(
        tree["ensemble/member_0"].ds.lon.values,
        tree["ensemble/member_0"].ds.lat.values,
        obs_data,
        cmap=cmap,
        transform=ccrs.PlateCarree(),
        vmin=cmap_range[0],
        vmax=cmap_range[1],
    )
    ax.set_title(title, fontsize=6)
    ax.set_xticks([])
    ax.set_yticks([])


def add_colorbar_and_title(figsize, params, output_dir="output"):
    """
    This function creates a new figure just for the colorbar and title, without using the previous figure.
    """
    try:
        # Create a new figure for the colorbar and title
        fig = plt.figure(figsize=figsize)

        ensemble_cmap_range = (-4, 4)

        # Adjust colorbar size and position for the first colorbar
        cbar_ax = fig.add_axes([0.90, 0.5, 0.05, 0.1])  # Adjusted for better fit
        norm = plt.Normalize(vmin=ensemble_cmap_range[0], vmax=ensemble_cmap_range[1])
        fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap="RdBu"),
            cax=cbar_ax,
            orientation="horizontal",
            label="SPI3 (Ensemble & Obs)",
        )

        # Add second colorbar for forecast
        cbar_ax2 = fig.add_axes([0.90, 0.2, 0.05, 0.1])  # Adjusted for better fit
        fct_cmap_range = (0.0, 1.0)
        norm = plt.Normalize(vmin=fct_cmap_range[0], vmax=fct_cmap_range[1])
        fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap="Blues"),
            cax=cbar_ax2,
            orientation="horizontal",
            label="Forecasts (mod/sev/ext)",
        )
        region_name = params.region_name_dict[params.region_id]
        fig.suptitle(
            f"{region_name} SEA51-CHRIPS Observations Forecasts for 1981-2022",
            fontsize=92,
            weight="bold",
            y=0.95,
        )
        for ax in fig.axes:
            if ax != cbar_ax and ax != cbar_ax2:
                fig.delaxes(ax)
        # Save the final figure
        final_output = f"{output_dir}/final_plot_with_colorbar_and_title_new.png"
        fig.savefig(final_output, dpi=100, bbox_inches="tight")
        plt.close(fig)
        logging.info(f"New final plot with colorbar and title saved to {final_output}")
    except Exception as e:
        logging.error(f"Error adding colorbar and title: {str(e)}")
        raise


def plot_allrows(seas51tree, params):
    """
    This function loops through the initializations and creates row plots for each init.
    After all init plots are generated, it calls a separate function to add a colorbar and title.
    """
    # Example usage:
    output_dir = f"{params.output_path}map_{params.region_id}_{params.sc_season_str}_lt{params.lead_int}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Folder created: {output_dir}")
    else:
        print(f"Folder already exists: {output_dir}")

    the_mask, rl_dict, mds1 = ken_mask_creator(params.data_path)
    region_geom = mds1[mds1["region"] == params.region_id]["geometry"].values[0]

    inits = seas51tree["ensemble/member_0"].ds.init.values
    plot_files = []  # Keep track of all the generated plot files
    last_fig, last_axs = None, None  # To store the figure from the last init
    color_mappable = None

    for i, init in enumerate(inits):
        fig, axs, figsize = create_single_row_plot(
            seas51tree, init, params, region_geom
        )

        # Store the mappable object for colorbar from the first axis (or any axis with valid data)
        if color_mappable is None and axs[0].collections:
            color_mappable = axs[0].collections[0]  # First mappable object for colorbar

        output_file = f'{output_dir}/stamp_plot_{init.strftime("%Y%m%d")}.png'
        plot_files.append(output_file)  # Append the file path

        # Save the figure for each init
        fig.savefig(output_file, dpi=100, bbox_inches="tight")
        plt.close(fig)

        # Keep track of the last figure and axes
        last_fig, last_axs, cbar_figsize = fig, axs, figsize

    # Once all init plots are done, add the colorbar and title using the last figure's size and axes
    add_colorbar_and_title(
        cbar_figsize,
        params,
        output_dir=output_dir,
    )


def merge_png_files(
    input_dir="single_row_plots",
    output_file="merged_stamp_plots.png",
    delete_originals=False,
):
    """
    Merges PNG files in the input directory into a single image and
    optionally deletes the original files. Saves the merged image in the same directory.
    This version maintains the original sizes of all images.

    Args:
        input_dir (str): Path to the directory containing PNG files.
        output_file (str): Name of the merged image file.
        delete_originals (bool): Whether to delete original files after merging.
    """
    logging.info(f"Starting image merging process in {input_dir}")
    try:
        # Get all PNG files in the input directory
        png_files = sorted(Path(input_dir).glob("*.png"))
        if not png_files:
            logging.warning(f"No PNG files found in {input_dir}")
            return

        # Open all images and get their sizes
        images = []
        max_width = 0
        total_height = 0
        for png_file in png_files:
            with Image.open(png_file) as img:
                images.append(img.copy())
                max_width = max(max_width, img.width)
                total_height += img.height

        # Create a new image with the calculated dimensions
        merged_image = Image.new("RGB", (max_width, total_height), (255, 255, 255))

        # Paste each image into the merged image
        y_offset = 0
        for img in images:
            merged_image.paste(img, ((max_width - img.width) // 2, y_offset))
            y_offset += img.height

        # Save the merged image in the input directory
        output_path = os.path.join(input_dir, output_file)
        merged_image.save(output_path, dpi=(300, 300))
        logging.info(f"Merged image saved as {output_path}")

        # Delete the individual PNG files if requested
        if delete_originals:
            for png_file in png_files:
                try:
                    os.remove(png_file)
                except Exception as e:
                    logging.error(f"Error deleting {png_file}: {e}")
            logging.info(f"Individual PNG files in {input_dir} have been deleted.")

    except Exception as e:
        logging.error(f"An error occurred during the merging process: {e}")


def amerge_png_files(
    input_dir="single_row_plots",
    output_file="merged_stamp_plots.png",
    delete_originals=False,
):
    """
    Merges PNG files in the input directory into a single image and
    deletes the original files. Saves the merged image in the same directory.

    Args:
        input_dir (str): Path to the directory containing PNG files.
        output_file (r): Name of the merged image file.
    """
    logging.info(f"Starting image merging process in {input_dir}")

    try:
        # Get all PNG files in the input directory
        png_files = sorted(Path(input_dir).glob("*.png"))

        if not png_files:
            logging.warning(f"No PNG files found in {input_dir}")
            return

        # Open the first image to get dimensions
        with Image.open(png_files[0]) as img:
            row_width, row_height = img.size

        # Create a new image with the calculated dimensions
        merged_height = row_height * len(png_files)
        merged_image = Image.new("RGB", (row_width, merged_height))

        # Paste each row image into the merged image
        for i, png_file in enumerate(png_files):
            try:
                with Image.open(png_file) as img:
                    merged_image.paste(img, (0, i * row_height))
            except Exception as e:
                logging.error(f"Error processing {png_file}: {e}")
                continue  # Skip this file and continue with the rest

        # Save the merged image in the input directory
        output_path = os.path.join(input_dir, output_file)
        merged_image.save(output_path, dpi=(300, 300))
        logging.info(f"Merged image saved as {output_path}")

        # Delete the individual PNG files
        if delete_originals:
            for png_file in png_files:
                try:
                    os.remove(png_file)
                except Exception as e:
                    logging.error(f"Error deleting {png_file}: {e}")
            logging.info(f"Individual PNG files in {input_dir} have been deleted.")

    except Exception as e:
        logging.error(f"An error occurred during the merging process: {e}")


def plot_obs_chart_with_triggers(
    plot_type, df, year_column, spi_column, threshold_dict, row_annotations
):
    """
    Create an Altair chart with a bar chart overlaid by trigger lines.

    Parameters:
    df : pandas.DataFrame
        The DataFrame containing the data.
    year_column : str
        The name of the DataFrame column containing the year.
    spi_column : str
        The name of the DataFrame column containing SPI values.
    threshold_dict : dict
        A dictionary with keys as threshold names and values as threshold values.
    """

    # Bar chart
    if plot_type == "obs":
        bar_chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X(f"{year_column}:N", axis=alt.Axis(labelAngle=90)),
                y=alt.Y(
                    f"{spi_column}:Q", title=spi_column, scale=alt.Scale(domain=[-4, 4])
                ),
                color=alt.condition(
                    alt.datum[spi_column] > 0,
                    alt.value("blue"),  # Color for positive values
                    alt.value("red"),  # Color for negative values
                ),
            )
            .properties(width=400, height=200)
        )
    else:
        color_scale = alt.Scale(
            # domain=["ext", "sev", "mod"], range=["#880203", "#ffa400", "#fffe00"]
            domain=["mod", "sev", "ext"],
            range=["#f4eb13", "#f89821", "#ed2227"],
        )

        bar_chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X(f"{year_column}:N", axis=alt.Axis(labelAngle=90)),
                y=alt.Y(f"{spi_column}:Q", title="Probability (%)", stack=None),
                color=alt.Color("cat:N", scale=color_scale, sort=["sev", "mod", "ext"]),
            )
            .properties(width=400, height=200)
            + row_annotations
        )

    # Adding trigger lines
    rules = []
    for key, value in threshold_dict.items():
        rule = (
            alt.Chart(pd.DataFrame({"y": [value]}))
            .mark_rule(
                strokeWidth=2,
                stroke={"ext": "#ed2227", "sev": "#f89821", "mod": "#f4eb13"}[
                    key
                ],  # Conditional color assignment
            )
            .encode(y="y:Q")
        )
        rules.append(rule)

    # Combine the bar chart with trigger lines
    final_chart = alt.layer(bar_chart, *rules)

    return final_chart


def aux_plot_make_barchart_annotation(params):

    def get_month_abbr(date):
        return date.strftime("%b")

    # Dictionary to map season strings to their last month
    last_month_dict = {"MAM": "May", "JJAS": "September", "OND": "December"}

    # Get the last month of the season
    last_month_str = last_month_dict.get(params.season_str)
    if not last_month_str:
        raise ValueError(f"Unsupported season string: {params.season_str}")

    # Parse the last month string to a datetime object
    last_month = datetime.strptime(last_month_str, "%B")

    # Calculate the month for this lead time
    month = last_month - relativedelta(months=params.lead_int + 1)

    # Determine the text for the annotation
    text = f"lt={params.lead_int}, {get_month_abbr(month)}"

    # Create the annotation chart
    annotation = (
        alt.Chart(pd.DataFrame({"text": [text]}))
        .mark_text(
            align="left",
            baseline="middle",
            fontSize=14,
            fontWeight="bold",
            dx=-190,
            dy=-90,
        )
        .encode(text="text:N")
        .properties(width=400, height=200)
    )

    return annotation


def plot_decision_table(df):
    return (
        alt.Chart(df.reset_index())
        .mark_text()
        .transform_fold(df.columns.tolist())
        .encode(
            alt.X(
                "key",
                type="nominal",
                axis=alt.Axis(
                    # flip x labels upside down
                    orient="top",
                    # put x labels into horizontal direction
                    labelAngle=0,
                    title=None,
                    ticks=False,
                ),
                scale=alt.Scale(padding=10),
                sort=None,
            ),
            alt.Y("index", type="ordinal", axis=None),
            alt.Text("value", type="nominal"),
        )
    )


def bar_stitch_plot(params, config):
    obs_plot = config["obs_plot"]
    lt2_plot = config["lt2_plot"]
    lt3_plot = config["lt3_plot"]
    lt4_plot = config["lt4_plot"]
    dec_dflt2 = config["dec_dflt2"]
    dec_dflt3 = config["dec_dflt3"]
    dec_dflt4 = config["dec_dflt4"]
    tab_df1 = pd.concat([dec_dflt2, dec_dflt3, dec_dflt4])
    tab_df1["Trigger"] = tab_df1["Trigger"] * 100
    tab_df1["Trigger"] = tab_df1["Trigger"].apply(
        lambda x: aux_mt_plot_round_list(x, 2)
    )
    tab_df1["%hit"] = tab_df1["%hit"].apply(lambda x: aux_mt_plot_round_list([x], 2))

    tab_plot = plot_decision_table(tab_df1).properties(height=200, width=400)
    # tab_plot
    emtpy_plot = (
        alt.Chart(pd.DataFrame({"A": []}))
        .mark_text()
        .encode()
        .properties(width=400, height=200)
    )

    panels = alt.vconcat(
        alt.hconcat(obs_plot, lt2_plot),
        alt.hconcat(tab_plot, lt3_plot),
        alt.hconcat(emtpy_plot, lt4_plot),
    )

    panels.configure_view(stroke=None).configure_axisY(
        labelFontSize=12, titleFontSize=14
    ).configure_axisX(labelFontSize=10, titleFontSize=12).configure_legend(
        labelFontSize=12, titleFontSize=14
    )

    # print(" Marsabit selected trigger for OND")
    panels.save(f"{params.output_path}{params.region_id}_{params.sc_season_str}.png")


def calculate_month(season, lt):
    """
    Calculate the month corresponding to a given season and lead time.

    Parameters:
    season (str): The season string, which can be 'mam' (March-April-May),
                  'jjas' (June-July-August-September), or 'ond' (October-November-December).
    lt (int): The lead time in months to subtract from the last month of the season.

    Returns:
    str: The name of the month corresponding to the given lead time and season.

    Raises:
    ValueError: If an unsupported season string is provided.
    """
    last_month_dict = {"mam": "May", "jjas": "September", "ond": "December"}

    # Get the last month of the season
    last_month_str = last_month_dict.get(season)
    if not last_month_str:
        raise ValueError(f"Unsupported season string: {season}")

    # Parse the last month string to a datetime object
    last_month = datetime.strptime(last_month_str, "%B")

    # Calculate the month for this lead time
    month = last_month - relativedelta(months=lt + 1)

    # Return the month as a string
    return month.strftime("%B")


def create_category_dataframes(df):
    """
    Create a set of dataframes in matrix form based on categories and metrics from the input dataframe.

    The function generates and returns a dictionary where each category ('mod', 'sev', 'ext') has
    dataframes that are structured as matrices, which can be queried by seaborn for heatmap generation.

    The keys in the returned dictionary can be accessed like dt_df[cat]['annot_hr'] or dt_df[cat]['data'],
    where each dataframe is indexed by months and seasons.

    Example matrix-like structure (as a dictionary for explanation):
    {
        'April': {'jjas': nan},
        'May': {'jjas': 1717.1717},
        'June': {'jjas': 2323.2323}
    }

    Parameters:
    df (pd.DataFrame): Input dataframe containing columns such as 'lt_month', 'cat',
                       'x2d_season', 'trigger_value', 'hit_rate', and 'false_alarm_ratio'.

    Returns:
    dict: A dictionary where keys are categories ('mod', 'sev', 'ext'), and values are
          dictionaries of pandas DataFrames for each metric (data, hit rate, false alarm ratio),
          indexed by months and seasons.

    The resulting DataFrames can be visualized using seaborn heatmaps by querying like:
    dt_df[cat]['annot_hr'], dt_df[cat]['data'], etc.
    """

    # Function to get month number (for sorting)
    def get_month_num(month_name):
        return list(month_abbr).index(month_name[:3].title())

    percentage_columns = ["trigger_value", "hit_rate", "false_alarm_ratio"]
    df[percentage_columns] = df[percentage_columns].mul(100)

    # Create empty dictionaries to store our results
    categories = ["mod", "sev", "ext"]
    metrics = ["data", "annot_hr", "annot_far"]
    result = {cat: {metric: {} for metric in metrics} for cat in categories}

    # Extract unique months from the input DataFrame
    all_months = sorted(df["lt_month"].unique(), key=get_month_num)

    # Iterate through the DataFrame
    for _, row in df.iterrows():
        cat = row["cat"]
        if cat not in categories:
            continue  # Skip if category is not mod, sev, or ext

        month = row["lt_month"]
        season = row["x2d_season"]

        # Initialize all months for this category if not already done
        for m in all_months:
            if m not in result[cat]["data"]:
                for metric in metrics:
                    result[cat][metric][m] = {}

        # Update the dictionaries
        result[cat]["data"][month][season] = row["trigger_value"]
        result[cat]["annot_hr"][month][season] = row["hit_rate"]
        result[cat]["annot_far"][month][season] = row["false_alarm_ratio"]

    # Convert dictionaries to DataFrames and sort columns
    for cat in categories:
        for metric in metrics:
            df = pd.DataFrame(result[cat][metric])
            # Ensure all months are present, fill with NaN if missing
            for month in all_months:
                if month not in df.columns:
                    df[month] = pd.np.nan
            # Sort columns (months) based on calendar order
            df = df.reindex(columns=all_months)
            result[cat][metric] = df

    return result


def generate_custom_colormap(reverse_colors=False):
    """
    Generate a custom colormap and normalization based on provided color ranges.

    Parameters:
    color_ranges (list): List of dictionaries with 'min', 'max', and 'color' keys.
    reverse_colors (bool): If True, reverse the order of colors.

    Returns:
    tuple: (colormap, normalization)
    """
    # Extract colors and boundaries from the color ranges
    color_ranges = [
        {"min": 0, "max": 20, "color": "#009600"},  # Green
        {"min": 20, "max": 40, "color": "#64C800"},  # Light green
        {"min": 40, "max": 60, "color": "#ffff00"},  # Yellow
        {"min": 60, "max": 80, "color": "#ff7800"},  # Orange
        {"min": 80, "max": 100, "color": "#ff0000"},  # Red
    ]
    colors = [r["color"] for r in color_ranges]
    boundaries = [r["min"] for r in color_ranges] + [
        color_ranges[-1]["max"]
    ]  # Add the last max to close the range

    # Reverse colors if specified
    if reverse_colors:
        colors = colors[::-1]

    # Create the custom colormap and normalization
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries, cmap.N)

    return cmap, norm


def create_heatmap_subplot(dt_df, params):
    """
    Create a heatmap subplot visualizing hit rates and false alarm ratios for three categories.
    Parameters:
    dt_df (dict): A dictionary containing dataframes for three categories ('mod', 'sev', 'ext'),
                  with data for hit rates (annot_hr) and false alarm ratios (annot_far).
    Returns:
    matplotlib.figure.Figure: The generated figure with subplots containing heatmaps.
    The figure consists of six subplots, where the top row displays heatmaps for hit rates and
    the bottom row displays heatmaps for false alarm ratios for 'Moderate', 'Severe', and
    'Extreme' categories. Two colorbars are added to the figure: one for hit rate and one for
    false alarm ratio.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    categories = ["mod", "sev", "ext"]
    titles = ["Moderate", "Severe", "Extreme"]

    hr_cmap, hr_norm = generate_custom_colormap(reverse_colors=True)
    far_cmap, far_norm = generate_custom_colormap(reverse_colors=False)

    for i, (cat, title) in enumerate(zip(categories, titles)):
        # Hit Rate heatmap (top row)
        sns.heatmap(
            dt_df[cat]["annot_hr"],
            annot=dt_df[cat]["data"],
            fmt=".1f",
            cmap=hr_cmap,
            norm=hr_norm,
            cbar=False,
            linewidths=0.5,
            linecolor="black",
            ax=axes[0, i],
        )
        axes[0, i].set_title(f"{title} - Hit Rate")

        # False Alarm Ratio heatmap (bottom row)
        sns.heatmap(
            dt_df[cat]["annot_far"],
            annot=dt_df[cat]["data"],
            fmt=".1f",
            cmap=far_cmap,
            norm=far_norm,
            cbar=False,
            linewidths=0.5,
            linecolor="black",
            ax=axes[1, i],
        )
        axes[1, i].set_title(f"{title} - False Alarm Ratio")

    # Adjust layout and add a main title
    plt.tight_layout()
    fig.suptitle("Hit Rates and False Alarm Ratios by Category", fontsize=16, y=1.02)

    # Add colorbars
    cbar_ax = fig.add_axes([1.02, 0.53, 0.02, 0.35])  # [left, bottom, width, height]
    plt.colorbar(plt.cm.ScalarMappable(cmap=hr_cmap, norm=hr_norm), cax=cbar_ax)
    cbar_ax.set_title("Hit Rate")

    cbar_ax = fig.add_axes([1.02, 0.1, 0.02, 0.35])  # [left, bottom, width, height]
    plt.colorbar(plt.cm.ScalarMappable(cmap=far_cmap, norm=far_norm), cax=cbar_ax)
    cbar_ax.set_title("False Alarm Ratio", rotation=90, y=0.5)
    plt.savefig(
        f"{params.output_path}dt_{params.region_id}_{params.sc_season_str}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
    return "made plot"


def run_map_plot(params):
    threshold_dict = get_threshold(params.region_id, params.sc_season_str)
    obs_data, ens_data = make_obs_fct_dataset(
        params.data_path, params.region_id, params.season_str, params.lead_int
    )
    fct_mod, fct_sev, fct_ext = seas51_patch_empirical_probability(
        ens_data, threshold_dict
    )

    dstree = helper_stamp_plot(ens_data, obs_data, fct_mod, fct_sev, fct_ext)
    plot_allrows(dstree, params)

    merge_png_files(
        input_dir=f"{params.output_path}map_{params.region_id}_{params.sc_season_str}_lt{params.lead_int}",
        output_file="map_{params.region_id}_{params.sc_season_str}_lt{params.lead_int}.png",
        delete_originals=False,
    )
    return "merge plot"
