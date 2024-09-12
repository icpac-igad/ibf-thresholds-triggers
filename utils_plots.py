from io import StringIO
import os
from dotenv import load_dotenv
import logging
from pathlib import Path

import climpred
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
import cartopy.crs as ccrs
import six
import textwrap as tw
from functools import reduce
import json
from dateutil.relativedelta import relativedelta
from calendar import monthrange
from PIL import Image


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


def create_single_row_plot(tree, init, output_dir, variable="spi3", is_last_plot=False):
    logging.info(f"Creating single row plot for {init} with variable {variable}")
    try:
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
            _plot_ensemble_member(
                tree, member_key, variable, init, axs[j], ensemble_cmap_range
            )

        # Add the observation and additional models as the last plots
        plot_titles = ["Obs", "mod", "sev", "ext"]
        plot_keys = ["observation", "fct_mod", "fct_sev", "fct_ext"]
        for k, (title, key) in enumerate(zip(plot_titles, plot_keys)):
            _plot_additional_data(
                tree,
                key,
                variable,
                init,
                valid_time,
                axs[num_members + k],
                title,
                ensemble_cmap_range if key == "observation" else fct_cmap_range,
            )

        if is_last_plot:
            _add_colorbars(fig, axs)
        else:
            plt.tight_layout()

        plt.tight_layout()
        output_file = f'{output_dir}/stamp_plot_{init.strftime("%Y%m%d")}.png'
        plt.savefig(output_file, dpi=100, bbox_inches="tight")
        logging.info(f"Plot saved to {output_file}")
        plt.close()

    except Exception as e:
        logging.error(f"Error creating plot: {str(e)}")
        raise


def _plot_ensemble_member(tree, member_key, variable, init, ax, cmap_range):
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


def _add_colorbars(fig, axs):
    cbar_ax = fig.add_axes([0.90, 0.5, 0.05, 0.1])
    cbar = fig.colorbar(axs[0].collections[0], cax=cbar_ax, orientation="horizontal")
    cbar.set_label("SPI3 (Ensemble & Obs)")

    cbar_ax2 = fig.add_axes([0.90, 0.2, 0.05, 0.1])
    cbar2 = plt.colorbar(axs[-1].collections[0], cax=cbar_ax2, orientation="horizontal")
    cbar2.set_label("Forecasts (mod/sev/ext)")


def plot_allrows(seas51tree, output_dir):
    # Example usage:
    inits = seas51tree["ensemble/member_0"].ds.init.values
    for i, init in enumerate(inits):
        is_last_plot = i == len(inits) - 1
        create_single_row_plot(seas51tree, init, output_dir, is_last_plot=is_last_plot)


def merge_png_files(
    input_dir="single_row_plots",
    output_file="merged_stamp_plots.png",
    delete_originals=False,
):
    """
    Merges PNG files in the input directory into a single image and
    deletes the original files. Saves the merged image in the same directory.

    Args:
        input_dir (str): Path to the directory containing PNG files.
        output_file (str): Name of the merged image file.
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


def DEPRICATE_aux_plot_make_barchart_annotations():
    row_annotations = [
        alt.Chart(pd.DataFrame({"text": ["lt=1"]}))
        .mark_text(
            align="left",
            baseline="middle",
            fontSize=14,
            fontWeight="bold",
            dx=-190,
            dy=-90,
        )
        .encode(text="text:N")
        .properties(width=400, height=200),
        alt.Chart(pd.DataFrame({"text": ["lt=2, Sep"]}))
        .mark_text(
            align="left",
            baseline="middle",
            fontSize=14,
            fontWeight="bold",
            dx=-190,
            dy=-90,
        )
        .encode(text="text:N")
        .properties(width=400, height=200),
        alt.Chart(pd.DataFrame({"text": ["lt=3, Aug"]}))
        .mark_text(
            align="left",
            baseline="middle",
            fontSize=14,
            fontWeight="bold",
            dx=-190,
            dy=-90,
        )
        .encode(text="text:N")
        .properties(width=400, height=200),
        alt.Chart(pd.DataFrame({"text": ["lt=4, Jul"]}))
        .mark_text(
            align="left",
            baseline="middle",
            fontSize=14,
            fontWeight="bold",
            dx=-190,
            dy=-90,
        )
        .encode(text="text:N")
        .properties(width=400, height=200),
        alt.Chart(pd.DataFrame({"text": ["lt=5"]}))
        .mark_text(
            align="left",
            baseline="middle",
            fontSize=14,
            fontWeight="bold",
            dx=-190,
            dy=-90,
        )
        .encode(text="text:N")
        .properties(width=400, height=200),
    ]
    return row_annotations


def get_month_abbr(date):
    return date.strftime("%b")


def aux_plot_make_barchart_annotation(params):
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


def aux_mt_plot_create_month_column(df):
    new_column = []

    for _, row in df.iterrows():
        lt = row["lt"]
        cat = row["cat"]
        season = row["season"]

        if season == "MAM":
            if lt == 1:
                if cat == "mod":
                    new_column.append("mar_x")
                elif cat == "sev":
                    new_column.append("mar_y")
                elif cat == "ext":
                    new_column.append("mar_z")
            elif lt == 2:
                if cat == "mod":
                    new_column.append("feb_x")
                elif cat == "sev":
                    new_column.append("feb_y")
                elif cat == "ext":
                    new_column.append("feb_z")
            elif lt == 3:
                if cat == "mod":
                    new_column.append("jan_x")
                elif cat == "sev":
                    new_column.append("jan_y")
                elif cat == "ext":
                    new_column.append("jan_z")
            elif lt == 4:
                if cat == "mod":
                    new_column.append("dec_x")
                elif cat == "sev":
                    new_column.append("dec_y")
                elif cat == "ext":
                    new_column.append("dec_z")
            elif lt == 5:
                if cat == "mod":
                    new_column.append("nov_x")
                elif cat == "sev":
                    new_column.append("nov_y")
                elif cat == "ext":
                    new_column.append("nov_z")
        elif season == "OND":
            if lt == 1:
                if cat == "mod":
                    new_column.append("oct_x")
                elif cat == "sev":
                    new_column.append("oct_y")
                elif cat == "ext":
                    new_column.append("oct_z")
            elif lt == 2:
                if cat == "mod":
                    new_column.append("sep_x")
                elif cat == "sev":
                    new_column.append("sep_y")
                elif cat == "ext":
                    new_column.append("sep_z")
            elif lt == 3:
                if cat == "mod":
                    new_column.append("aug_x")
                elif cat == "sev":
                    new_column.append("aug_y")
                elif cat == "ext":
                    new_column.append("aug_z")
            elif lt == 4:
                if cat == "mod":
                    new_column.append("jul_x")
                elif cat == "sev":
                    new_column.append("jul_y")
                elif cat == "ext":
                    new_column.append("jul_z")
            elif lt == 5:
                if cat == "mod":
                    new_column.append("jun_x")
                elif cat == "sev":
                    new_column.append("jun_y")
                elif cat == "ext":
                    new_column.append("jun_z")
        elif season == "JJAS":
            if lt == 2:
                if cat == "mod":
                    new_column.append("jun_x")
                elif cat == "sev":
                    new_column.append("jun_y")
                elif cat == "ext":
                    new_column.append("jun_z")
            elif lt == 3:
                if cat == "mod":
                    new_column.append("may_x")
                elif cat == "sev":
                    new_column.append("may_y")
                elif cat == "ext":
                    new_column.append("may_z")
            elif lt == 4:
                if cat == "mod":
                    new_column.append("apr_x")
                elif cat == "sev":
                    new_column.append("apr_y")
                elif cat == "ext":
                    new_column.append("apr_z")
            elif lt == 5:
                if cat == "mod":
                    new_column.append("mar_x")
                elif cat == "sev":
                    new_column.append("mar_y")
                elif cat == "ext":
                    new_column.append("mar_z")
        else:
            new_column.append("")

    # df["new_column"] = new_column
    df.insert(loc=0, column="new_column", value=new_column)
    return df


def aux_mt_plot_replace_with_list(x):
    """
    Replaces NaN float values with a predefined list of replacement values.

    Parameters:
    - x (float): The input value to be checked and potentially replaced.

    Returns:
    - A list of replacement values if `x` is a float and is NaN. Otherwise, returns `x` unchanged.

    Note:
    - This function is designed to handle cases where cell values in a dataset need to be replaced with a list of values
      for indicating missing or special cases.
    """
    replacement_values = [-999.0, -999.0]
    # If x is a float and it is nan (meaning the cell was originally empty), return the replacement list
    if isinstance(x, float) and np.isnan(x):
        return replacement_values
    # Otherwise, return x as it is
    return x


def aux_mt_plot_round_list(lst, decimal_places):
    """
    Rounds each element in a list to a specified number of decimal places.

    Parameters:
    - lst (list of float): The list of numbers to be rounded.
    - decimal_places (int): The number of decimal places to round each number to.

    Returns:
    - A list containing the rounded values of the input list.

    Note:
    - This function is useful for rounding numerical values in a list to ensure consistency or to improve readability.
    """
    if isinstance(lst, list):
        return [round(x, decimal_places) for x in lst]
    else:
        # If it's not a list, assume it's a single float and round it
        return round(lst, decimal_places)


# %% table plot matplotlib

### Define the picture size and remove the ticks


### functions for whole column, row editing
def aux_mt_plot_legend_maker(text1, color_list, legend_title):
    square6 = plt.Rectangle((0.4, 0.1), 0.15, 0.25, color=color_list[0], clip_on=False)
    text1.add_artist(square6)
    square5 = plt.Rectangle((0.55, 0.1), 0.15, 0.25, color=color_list[1], clip_on=False)
    text1.add_artist(square5)
    square5 = plt.Rectangle((0.7, 0.1), 0.15, 0.25, color=color_list[2], clip_on=False)
    text1.add_artist(square5)
    square5 = plt.Rectangle((0.85, 0.1), 0.15, 0.25, color=color_list[3], clip_on=False)
    text1.add_artist(square5)
    square5 = plt.Rectangle((1.0, 0.1), 0.15, 0.25, color=color_list[4], clip_on=False)
    text1.add_artist(square5)
    plt.text(
        0.6,
        0.4,
        legend_title,
        horizontalalignment="left",
        fontsize=6,
        fontweight="bold",
        color="k",
        verticalalignment="center",
        transform=text1.transAxes,
    )
    plt.text(
        0.42,
        0.05,
        "<20",
        horizontalalignment="left",
        fontsize=6,
        fontweight="bold",
        color="k",
        verticalalignment="center",
        transform=text1.transAxes,
    )
    plt.text(
        0.57,
        0.05,
        "20-40",
        horizontalalignment="left",
        fontsize=6,
        fontweight="bold",
        color="k",
        verticalalignment="center",
        transform=text1.transAxes,
    )
    plt.text(
        0.72,
        0.05,
        "40-60",
        horizontalalignment="left",
        fontsize=6,
        fontweight="bold",
        color="k",
        verticalalignment="center",
        transform=text1.transAxes,
    )
    plt.text(
        0.87,
        0.05,
        "60-80",
        horizontalalignment="left",
        fontsize=6,
        fontweight="bold",
        color="k",
        verticalalignment="center",
        transform=text1.transAxes,
    )
    plt.text(
        1.05,
        0.05,
        "80<",
        horizontalalignment="left",
        fontsize=6,
        fontweight="bold",
        color="k",
        verticalalignment="center",
        transform=text1.transAxes,
    )


def aux_mt_plot_set_align_for_column(table, col, align="left"):
    cells = [key for key in table._cells if key[1] == col]
    for cell in cells:
        table._cells[cell]._loc = align


def aux_mt_plot_set_width_for_column(table, col, width):
    cells = [key for key in table._cells if key[1] == col]
    for cell in cells:
        table._cells[cell]._width = width


def aux_mt_plot_set_height_for_row(table, row, height):
    cells = [key for key in table._cells if key[0] == row]
    for cell in cells:
        table._cells[cell]._height = height


def aux_mt_plot_colorcell(tablerows, tablecols, cellDict, color_list):
    allcells = [(x, y) for x in tablerows[1:] for y in tablecols[2:]]
    for alcls in allcells:
        cell_value0 = json.loads(cellDict[alcls]._text.get_text())[0]
        if cell_value0 == -999.0:
            cellDict[alcls].set_facecolor("#FFFFFF")
        else:
            if float(cell_value0) <= 0.2:
                cellDict[alcls].set_facecolor(color_list[0])
            elif 0.2 < float(cell_value0) <= 0.4:
                cellDict[alcls].set_facecolor(color_list[1])
            elif 0.4 < float(cell_value0) <= 0.6:
                cellDict[alcls].set_facecolor(color_list[2])
            elif 0.6 < float(cell_value0) <= 0.8:
                cellDict[alcls].set_facecolor(color_list[3])
            elif 0.8 < float(cell_value0) <= 1.0:
                cellDict[alcls].set_facecolor(color_list[4])
            else:
                cellDict[alcls].set_facecolor("#FFFFFF")


def aux_mt_plot_remove_value(tablerows, tablecols, mpl_table):
    allcells = [(x, y) for x in tablerows[1:] for y in tablecols[2:]]
    for alcls in allcells:
        mpl_table._cells[alcls]._text.set_text("")


def aux_mt_plot_add_certain_value(tablerows, tablecols, mpl_table, cellDict):
    allcells = [(x, y) for x in tablerows[1:] for y in tablecols[2:]]
    for alcls in allcells:
        # print(cellDict[alcls]._text.get_text())
        cell_value0 = json.loads(cellDict[alcls]._text.get_text())[1]
        mpl_table._cells[alcls]._text.set_text("")
        # cell_value0=(cellDict[alcls]._text.get_text())
        if cell_value0 == -999.0:
            mpl_table._cells[alcls]._text.set_text("")
        elif cell_value0 == 999.0:
            mpl_table._cells[alcls]._text.set_text("")
        else:
            ncl = "%.1f" % cell_value0
            mpl_table._cells[alcls]._text.set_text(ncl)


def aux_mt_plot_aset_height_for_row_except_head(table, rowlist, height):
    cells_list = []
    for row in rowlist:
        cells = [key for key in table._cells if key[0] == row]
        cells_list.append(cells)
    for cells in cells_list:
        for cell in cells:
            table._cells[cell]._height = height


def aux_mt_plot_bset_height_for_row_except_head(table, rowlist, height):
    for row in rowlist:
        for col in range(len(table[row])):
            cell = table[row, col]
            cell._height = height


def aux_mt_plot_cset_height_for_row_except_head(table, row_height):
    """chatGPT function"""
    for i, cell in six.iteritems(table._cells):
        if i[0] == 0:  # Skip header row
            continue
        cell.set_height(row_height)


def aux_mt_plot_set_height_for_row_except_head(cellDict, header_row_count, height):
    for cell_key, cell in cellDict.items():
        row, col = cell_key
        if row < header_row_count:
            continue  # skip header rows
        cell.set_height(height)


def aux_mt_plot_table_header_colour(tablerows, tablecols, cellDict, mpl_table):
    allcells = [(x, y) for x in tablerows[0:1] for y in tablecols]
    header_list = [
        "Region",
        "SPI",
        "Jul",
        "Aug",
        "Sep",
        "",
        "Jul",
        "Aug",
        "Sep",
        "",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "",
        "Nov",
        "Dec",
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "",
        "Nov",
        "Dec",
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "",
    ]
    for idx, alcls in enumerate(allcells):
        cellDict[alcls].set_facecolor("#FFFFFF")
        print(header_list[idx])
        text = header_list[idx]
        mpl_table._cells[alcls]._text.set_text(text)


### funciton for table creation
def aux_mt_plot_render_mpl_table(
    data,
    color_list,
    col_width=1.0,
    row_height=0.425,
    font_size=5,
    header_color="#40466e",
    row_colors=["#f1f1f2", "w"],
    edge_color="w",
    bbox=[0, 0, 1, 1],
    header_columns=0,
    ax=None,
    **kwargs,
):
    """
    Renders a matplotlib table from a pandas DataFrame, allowing for customization of various aesthetic parameters.

    Parameters:
    - data (pandas.DataFrame): The data to display in the table.
    - color_list (list): A list of colors to use for cell background coloring based on cell values.
    - col_width (float): The width of the columns. Default is 1.0.
    - row_height (float): The height of the rows. Default is 0.625.
    - font_size (int): Font size for the cell texts. Default is 5.
    - header_color (str): Color code or name for the table header's background. Default is '#40466e'.
    - row_colors (list): A list containing color codes for alternating row colors. Default is ['#f1f1f2', 'w'].
    - edge_color (str): Color code or name for the cell edge lines. Default is 'w' (white).
    - bbox (list): A 4-element list defining the bounding box of the table within the plot. Default is [0, 0, 1, 1].
    - header_columns (int): The number of initial columns considered as header columns. Default is 0.
    - ax (matplotlib.axes.Axes): The matplotlib axes object where the table will be rendered. If None, a new one will be created.

    Returns:
    - ax (matplotlib.axes.Axes): The matplotlib axes object with the rendered table.

    This function creates a visual representation of a DataFrame as a static table in a matplotlib figure. It allows for
    significant customization, including cell coloring based on values, flexible sizing, font adjustments, and more. The
    function is particularly useful for creating detailed reports or visual summaries of data within a matplotlib figure.

    Additional keyword arguments (**kwargs) are passed directly to the `matplotlib.axes.Axes.table` method.
    """
    mpl_table = ax.table(
        cellText=data.values, bbox=bbox, colLabels=[""] * 42, cellLoc="center", **kwargs
    )
    set_align_for_column(mpl_table, col=0, align="left")
    set_width_for_column(mpl_table, 0, 0.6)
    set_width_for_column(mpl_table, 1, 0.5)
    for idx in range(2, 42):
        set_width_for_column(mpl_table, idx, 0.2)
    set_height_for_row(mpl_table, 0, 0.01)
    # set_height_for_row_except_head(mpl_table, np.arange(1, len(data.index)), 0.06)
    # set_height_for_row_except_head(mpl_table, row_height=0.03)
    cellDict = mpl_table.get_celld()
    set_height_for_row_except_head(cellDict, header_row_count=1, height=0.03)
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)
    cellDict = mpl_table.get_celld()
    tablerows = np.arange(0, len(data.index) + 1)
    tablecols = np.arange(0, len(data.columns))
    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight="bold", color="black")
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])
    colorcell(tablerows, tablecols, cellDict, color_list)
    headings = data.columns
    plt.text(
        0.245,
        1.08,
        "Mild",
        fontsize=10,
        fontweight="bold",
        color="black",
        ha="left",
        va="center",
        transform=ax.transAxes,
    )
    plt.text(
        0.545,
        1.08,
        "Moderate",
        fontsize=10,
        fontweight="bold",
        color="black",
        ha="left",
        va="center",
        transform=ax.transAxes,
    )
    plt.text(
        0.845,
        1.08,
        "Severe",
        fontsize=10,
        fontweight="bold",
        color="black",
        ha="left",
        va="center",
        transform=ax.transAxes,
    )
    table_header_colour(tablerows, tablecols, cellDict, mpl_table)
    add_certain_value(tablerows, tablecols, mpl_table, cellDict)
    return ax


def aux_mt_plot_plot_data_table(latex_path, data_table, stat_var, req_list, region_id):
    # Width and height of A4 portrait with 1-inch margins
    width = 3.67 - 2  # one inch margin on each side
    height = 11.69 - 2  # one inch margin on the top and bottom
    fig = plt.figure()
    fig.set_size_inches(height, width)
    # [left, bottom, width, height]
    table = fig.add_axes([0.04, 0.15, 0.93, 0.75], frame_on=False)
    table.xaxis.set_ticks_position("none")
    table.yaxis.set_ticks_position("none")
    table.set_xticklabels("")
    table.set_yticklabels("")
    #######
    laxes = fig.add_axes([0.22, 0.01, 0.4, 0.3], frame_on=False, zorder=0)
    laxes.xaxis.set_ticks_position("none")
    laxes.yaxis.set_ticks_position("none")
    laxes.set_xticklabels("")
    laxes.set_yticklabels("")
    if stat_var == "FAR":
        legend_title = "False Alarm Ratio %"
        color_list = ["#009600", "#64C800", "#ffff00", "#ff7800", "#ff0000"]
    else:
        legend_title = "Hit Rate %"
        color_list = ["#ff0000", "#ff7800", "#ffff00", "#64C800", "#009600"]
    legend_maker(laxes, color_list, legend_title)
    #####
    data1 = data_table[req_list]
    print(data1.info())
    mpl_table = render_mpl_table(
        data1, color_list, header_columns=0, col_width=0.2, ax=table
    )
    # cellDict = mpl_table.get_celld()
    # set_height_for_row_except_head(cellDict, header_row_count=1, height=0.06)
    # set_height_for_row_except_head(mpl_table, row_height=0.125)
    var = stat_var.lower()
    plt.show
    plt.savefig(f"{latex_path}{var}_{region_id}_prob_v20240703.jpg", dpi=300)


def aux_mt_plot_pass_month_get_colnames(months):
    original_list = [
        "region_x",
        "season",
        "nov_x",
        "dec_x",
        "jan_x",
        "feb_x",
        "mar_x",
        "apr_x",
        "may_x",
        "jun_x",
        "jul_x",
        "aug_x",
        "sep_x",
        "oct_x",
        "empty1",
        "nov_y",
        "dec_y",
        "jan_y",
        "feb_y",
        "mar_y",
        "apr_y",
        "may_y",
        "jun_y",
        "jul_y",
        "aug_y",
        "sep_y",
        "oct_y",
        "empty2",
        "nov_z",
        "dec_z",
        "jan_z",
        "feb_z",
        "mar_z",
        "apr_z",
        "may_z",
        "jun_z",
        "jul_z",
        "aug_z",
        "sep_z",
        "oct_z",
    ]
    # months = ['jul', 'aug', 'sep']
    suffixes = ["_x", "_y", "_z"]

    organized_list = [
        "region_x",
        "season",
    ]

    for suffix in suffixes:
        for month in months:
            item = month + suffix
            if item in original_list:
                organized_list.append(item)

        if suffix == "_x":
            organized_list.append("empty1")
        elif suffix == "_y":
            organized_list.append("empty2")
    return organized_list


def aux_mt_plot_table_df(tab_df, stat_var):
    tab_df_a = tab_df.rename(columns={"lead_time": "lt"})
    tab_df_m = create_month_column(tab_df_a)
    tab_df_m["pod_v"] = tab_df_m.apply(
        lambda x: [x["hit_rates"], x["trigger_values"]], axis=1
    )
    tab_df_m["far_v"] = tab_df_m.apply(
        lambda x: [x["false_alarm_ratios"], x["trigger_values"]], axis=1
    )
    tab_df_m["pod_v"] = tab_df_m["pod_v"].apply(lambda x: round_list(x, 2))
    tab_df_m["far_v"] = tab_df_m["far_v"].apply(lambda x: round_list(x, 2))
    mapping_dict = {0: "Karamoja", 1: "Marsabit", 2: "Wajir"}
    tab_df_m["region_x"] = tab_df_m["region"].replace(mapping_dict)
    p = tab_df_m.pivot_table(
        index=["region_x", "season"],
        columns="new_column",
        values="pod_v",
        aggfunc="first",
    )
    pf = p.reset_index()
    # Apply the custom function to each cell in the DataFrame
    pf1 = pf.applymap(replace_with_list)
    pf1.columns.name = None
    pf1["empty1"] = [[-999.0, -999.0]] * len(pf1)
    pf1["empty2"] = [[-999.0, -999.0]] * len(pf1)
    months = ["jul", "aug", "sep"]
    organized_list = pass_month_get_colnames(months)
    pf2 = pf1[organized_list]
    mask = (pf2["region_x"].isin(["Marsabit", "Wajir"])) & (pf2["season"] == "OND")
    pf3 = pf2[mask]
    plot_data_table(pf3, stat_var, organized_list)
