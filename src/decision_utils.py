import numpy as np
import pandas as pd

import pandas as pd
import numpy as np
import matplotlib

import os
from dotenv import load_dotenv

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import six
from datetime import datetime
import textwrap as tw
from functools import reduce
import json


def create_new_column(df):
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
                    new_column.append("may_x")
                elif cat == "sev":
                    new_column.append("may_y")
                elif cat == "ext":
                    new_column.append("may_z")
            elif lt == 3:
                if cat == "mod":
                    new_column.append("apr_x")
                elif cat == "sev":
                    new_column.append("apr_y")
                elif cat == "ext":
                    new_column.append("apr_z")
            elif lt == 4:
                if cat == "mod":
                    new_column.append("mar_x")
                elif cat == "sev":
                    new_column.append("mar_y")
                elif cat == "ext":
                    new_column.append("mar_z")
            elif lt == 5:
                if cat == "mod":
                    new_column.append("feb_x")
                elif cat == "sev":
                    new_column.append("feb_y")
                elif cat == "ext":
                    new_column.append("feb_z")
        else:
            new_column.append("")

    df["new_column"] = new_column
    return df


def get_subset(df):
    # Filter out rows with null values in 'hit_rate' and 'false_alarm_ratio'
    # df = df.dropna(subset=['hit_rate', 'false_alarm_ratio'])

    # Sort the DataFrame by 'peirce_score' in descending order
    df = df.sort_values(by="peirce_score", ascending=False)

    # Get the row with the maximum 'peirce_score'
    max_peirce_row = df.iloc[0]

    # Sort the DataFrame by 'bias_score' in descending order, and filter for 'bias_score' < 1.0
    df = df.loc[df["bias_score"] < 1.0].sort_values(by="bias_score", ascending=False)

    # Get the row with the maximum 'bias_score' < 1.0
    max_bias_row = df.iloc[0]

    # Sort the DataFrame by 'heidke_score' in descending order
    df = df.sort_values(by="heidke_score", ascending=False)

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


def choose_row(df):
    # Filter out rows where false_alarm_ratio or hit_rate is 1.0 or 0.0
    # filtered_df = df[(df['false_alarm_ratio'] != 1.0) & (df['false_alarm_ratio'] != 0.0) &
    #                 (df['hit_rate'] != 1.0) & (df['hit_rate'] != 0.0)]

    # Filter out rows where percentage_spi is less than or equal to 10
    # filtered_df = df[df['percentage_spi'] > 10]

    # If there are no rows left after filtering, return None
    # if filtered_df.empty:
    #    return None

    # Sort the filtered DataFrame by percentage_spi in descending order
    filtered_df = df.sort_values(by="percentage_spi", ascending=False)

    # Return the first row of the sorted DataFrame
    # chosen_row = filtered_df.iloc[0]
    chosen_row = pd.DataFrame([filtered_df.iloc[0]])

    return chosen_row


def replace_with_list(x):
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


def round_list(lst, decimal_places):
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
    return [round(x, decimal_places) for x in lst]


# %% table plot matplotlib

### Define the picture size and remove the ticks


### functions for whole column, row editing
def legend_maker(text1, color_list, legend_title):
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
        fontsize=8,
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
        fontsize=8,
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
        fontsize=8,
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
        fontsize=8,
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
        fontsize=8,
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
        fontsize=8,
        fontweight="bold",
        color="k",
        verticalalignment="center",
        transform=text1.transAxes,
    )


def set_align_for_column(table, col, align="left"):
    cells = [key for key in table._cells if key[1] == col]
    for cell in cells:
        table._cells[cell]._loc = align


def set_width_for_column(table, col, width):
    cells = [key for key in table._cells if key[1] == col]
    for cell in cells:
        table._cells[cell]._width = width


def set_height_for_row(table, row, height):
    cells = [key for key in table._cells if key[0] == row]
    for cell in cells:
        table._cells[cell]._height = height


def colorcell(tablerows, tablecols, cellDict, color_list):
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


def remove_value(tablerows, tablecols, mpl_table):
    allcells = [(x, y) for x in tablerows[1:] for y in tablecols[2:]]
    for alcls in allcells:
        mpl_table._cells[alcls]._text.set_text("")


def add_certain_value(tablerows, tablecols, mpl_table, cellDict):
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


def aset_height_for_row_except_head(table, rowlist, height):
    cells_list = []
    for row in rowlist:
        cells = [key for key in table._cells if key[0] == row]
        cells_list.append(cells)
    for cells in cells_list:
        for cell in cells:
            table._cells[cell]._height = height


def bset_height_for_row_except_head(table, rowlist, height):
    for row in rowlist:
        for col in range(len(table[row])):
            cell = table[row, col]
            cell._height = height


def cset_height_for_row_except_head(table, row_height):
    """chatGPT function"""
    for i, cell in six.iteritems(table._cells):
        if i[0] == 0:  # Skip header row
            continue
        cell.set_height(row_height)


def set_height_for_row_except_head(cellDict, header_row_count, height):
    for cell_key, cell in cellDict.items():
        row, col = cell_key
        if row < header_row_count:
            continue  # skip header rows
        cell.set_height(height)


def table_header_colour(tablerows, tablecols, cellDict, mpl_table):
    allcells = [(x, y) for x in tablerows[0:1] for y in tablecols]
    header_list = [
        "Region",
        "SPI",
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
def render_mpl_table(
    data,
    color_list,
    col_width=1.0,
    row_height=0.625,
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
        "Moderate",
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
        "Severe",
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
        "Extreme",
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


def plot_data_table(data_table, stat_var):
    # Width and height of A4 portrait with 1-inch margins
    width = 5.27 - 2  # one inch margin on each side
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
    req_list = [
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
    data1 = data_table[req_list]
    print(data1.info())
    mpl_table = render_mpl_table(
        data1, color_list, header_columns=0, col_width=0.2, ax=table
    )
    # cellDict = mpl_table.get_celld()
    # set_height_for_row_except_head(cellDict, header_row_count=1, height=0.06)
    # set_height_for_row_except_head(mpl_table, row_height=0.125)
    var = stat_var.lower()
    plt.savefig(f"{data_path}{var}_prob.jpg", dpi=300)


def decide_for_region_season(df, region_id, season):
    df_no0 = df[df["lt"] != 0]
    mask = (df_no0["lt"] == 1) & (df_no0["season"] == season)
    df_no1 = df_no0[~mask]
    df_no2 = df_no1[df_no1["subset"] == "mean"]
    # db2.info()
    df_no3 = df_no2[df_no2["region_id"] == region_id]
    df_no4 = df_no3[df_no3["season"] == season]
    df3 = create_new_column(df_no4)
    df3 = df3.assign(
        identify=df3["region_id"].astype(str)
        + "-"
        + df3["season"]
        + "-"
        + df3["new_column"]
        + "-"
        + df3["cat"]
    )
    _ = df3.drop_duplicates("identify")
    identify_list = _["identify"].tolist()
    d_odb = []
    for idl in identify_list:
        odb = df3[df3["identify"] == idl]
        odb1 = get_subset(odb)
        odb2 = choose_row(odb1)
        d_odb.append(odb2)
    ddf = pd.concat(d_odb)
    # ddf1=ddf[ddf['region_id']==region_id]
    # ddf2=ddf1[ddf1['season']==season]
    return ddf


def get_ep_for_region_season(df, region_id, season):
    df_no0 = df[df["lt"] != 0]
    mask = (df_no0["lt"] == 1) & (df_no0["season"] == "JJAS")
    df_no1 = df_no0[~mask]
    df_no2 = df_no1[df_no1["subset"] == "mean"]
    df_no3 = df_no2[df_no2["region_id"] == region_id]
    df_no4 = df_no3[df_no3["season"] == season]
    df_no5 = create_new_column(df_no4)
    df_no5 = df_no5.assign(
        identify=df_no5["region_id"].astype(str)
        + "-"
        + df_no5["season"]
        + "-"
        + df_no5["new_column"]
        + "-"
        + df_no5["cat"]
    )
    return df_no5


def mean_obs_spi(obs_data, spi_string_name):
    obs_data_mean = obs_data.mean(dim=["lat", "lon"])
    obs_data_df = obs_data_mean.to_dataframe().reset_index()
    obs_data_df1 = obs_data_df[["time", spi_string_name]]
    wdf = obs_data_df1
    wdf["year0"] = wdf["time"].apply(
        lambda x: datetime(x.year, x.month, x.day, x.hour, x.minute, x.second)
    )
    wdf["year"] = wdf["year0"].dt.strftime("%Y")
    wdf1 = wdf[[spi_string_name, "year"]]
    return wdf1
