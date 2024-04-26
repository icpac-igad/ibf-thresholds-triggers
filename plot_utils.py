import os
from dotenv import load_dotenv

import altair as alt
import pandas as pd

from utils import get_threshold
from utils import make_obs_fct_dataset
from utils import emprical_probablity

from utils import mean_emp_prob
from utils import min_emp_prob
from utils import max_emp_prob

from utils import ep_process_data
from utils import make_obs_fct_dataset
from utils import get_threshold

from decision_utils import decide_for_region_season
from decision_utils import get_ep_for_region_season
from decision_utils import mean_obs_spi


load_dotenv()

data_path = os.getenv("data_path")


def probability_plot(df, region_id, season_str):
    if region_id == 0:
        region_name = "Karamoja"
    if region_id == 1:
        region_name = "Marsabit"
    if region_id == 2:
        region_name = "Wajir"
    row_annotations = [
        alt.Chart(
            pd.DataFrame(
                {
                    "text": [
                        f"{region_name}-{season_str} lt=0, Mean of the pixel wise Probablities for the region "
                    ]
                }
            )
        )
        .mark_text(
            align="left",
            baseline="middle",
            fontSize=12,
            fontWeight="bold",
            dx=-190,
            dy=-90,
        )
        .encode(text="text:N")
        .properties(width=400, height=200),
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
        alt.Chart(pd.DataFrame({"text": ["lt=2"]}))
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
        alt.Chart(pd.DataFrame({"text": ["lt=3"]}))
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
        alt.Chart(pd.DataFrame({"text": ["lt=4"]}))
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
        alt.Chart(
            pd.DataFrame(
                {"text": ["Minimum of the pixel wise Probablities for the region "]}
            )
        )
        .mark_text(
            align="left",
            baseline="middle",
            fontSize=12,
            fontWeight="bold",
            dx=-190,
            dy=-90,
        )
        .encode(text="text:N")
        .properties(width=400, height=200),
        alt.Chart(
            pd.DataFrame(
                {"text": ["Maximum of the pixel wise Probablities for the region "]}
            )
        )
        .mark_text(
            align="left",
            baseline="middle",
            fontSize=12,
            fontWeight="bold",
            dx=-190,
            dy=-90,
        )
        .encode(text="text:N")
        .properties(width=400, height=200),
    ]

    # Update the color scale
    color_scale = alt.Scale(
        # domain=["ext", "sev", "mod"], range=["#880203", "#ffa400", "#fffe00"]
        domain=["mod", "sev", "ext"],
        range=["#fffe00", "#ffa400", "#880203"],
    )
    # Create the overlaid area chart using Altair
    base_plot = (
        alt.Chart(df)
        .mark_area()
        .encode(
            x=alt.X("year:N", axis=alt.Axis(labelAngle=90)),
            y=alt.Y("percentage_spi:Q", title="Probability (%)", stack=None),
            color=alt.Color("cat:N", scale=color_scale, sort=["sev", "mod", "ext"]),
        )
        .properties(width=400, height=200)
    )

    # ... (rest of the code for creating the panels remains the same)

    base_plot.configure_view(stroke=None).configure_axisY(
        labelFontSize=12, titleFontSize=14
    ).configure_axisX(labelFontSize=8, titleFontSize=12).configure_legend(
        labelFontSize=12, titleFontSize=14
    )

    panel_0_mean = alt.layer(
        base_plot.transform_filter(
            (alt.datum.lt == "0") & (alt.datum.subset == "mean")
        ),
        row_annotations[0],
    )
    panel_0_min = alt.layer(
        base_plot.transform_filter((alt.datum.lt == "0") & (alt.datum.subset == "min")),
        row_annotations[6],
    )
    panel_0_max = alt.layer(
        base_plot.transform_filter((alt.datum.lt == "0") & (alt.datum.subset == "max")),
        row_annotations[7],
    )

    panel_1_mean = alt.layer(
        base_plot.transform_filter(
            (alt.datum.lt == "1") & (alt.datum.subset == "mean")
        ),
        row_annotations[1],
    )
    panel_1_min = alt.layer(
        base_plot.transform_filter((alt.datum.lt == "1") & (alt.datum.subset == "min")),
    )
    panel_1_max = alt.layer(
        base_plot.transform_filter((alt.datum.lt == "1") & (alt.datum.subset == "max")),
    )

    panel_2_mean = alt.layer(
        base_plot.transform_filter(
            (alt.datum.lt == "2") & (alt.datum.subset == "mean")
        ),
        row_annotations[2],
    )
    panel_2_min = alt.layer(
        base_plot.transform_filter((alt.datum.lt == "2") & (alt.datum.subset == "min")),
    )
    panel_2_max = alt.layer(
        base_plot.transform_filter((alt.datum.lt == "2") & (alt.datum.subset == "max")),
    )

    panel_3_mean = alt.layer(
        base_plot.transform_filter(
            (alt.datum.lt == "3") & (alt.datum.subset == "mean")
        ),
        row_annotations[3],
    )
    panel_3_min = alt.layer(
        base_plot.transform_filter((alt.datum.lt == "3") & (alt.datum.subset == "min")),
    )
    panel_3_max = alt.layer(
        base_plot.transform_filter((alt.datum.lt == "3") & (alt.datum.subset == "max")),
    )

    panel_4_mean = alt.layer(
        base_plot.transform_filter(
            (alt.datum.lt == "4") & (alt.datum.subset == "mean")
        ),
        row_annotations[4],
    )
    panel_4_min = alt.layer(
        base_plot.transform_filter((alt.datum.lt == "4") & (alt.datum.subset == "min")),
    )
    panel_4_max = alt.layer(
        base_plot.transform_filter((alt.datum.lt == "4") & (alt.datum.subset == "max")),
    )

    panel_5_mean = alt.layer(
        base_plot.transform_filter(
            (alt.datum.lt == "5") & (alt.datum.subset == "mean")
        ),
        row_annotations[5],
    )
    panel_5_min = alt.layer(
        base_plot.transform_filter((alt.datum.lt == "5") & (alt.datum.subset == "min")),
    )
    panel_5_max = alt.layer(
        base_plot.transform_filter((alt.datum.lt == "5") & (alt.datum.subset == "max")),
    )

    panels = alt.vconcat(
        alt.hconcat(panel_0_mean, panel_0_min, panel_0_max),
        alt.hconcat(panel_1_mean, panel_1_min, panel_1_max),
        alt.hconcat(panel_2_mean, panel_2_min, panel_2_max),
        alt.hconcat(panel_3_mean, panel_3_min, panel_3_max),
        alt.hconcat(panel_4_mean, panel_4_min, panel_4_max),
        # alt.hconcat(panel_5_mean, panel_5_min, panel_5_max),
    )

    panels.configure_view(stroke=None).configure_axisY(
        labelFontSize=12, titleFontSize=14
    ).configure_axisX(labelFontSize=10, titleFontSize=12).configure_legend(
        labelFontSize=12, titleFontSize=14
    )

    return panels


def temp_kimwa_ep_plot():
    region_id = 0
    season_str = "MAM"
    spi_string_name = "spi3"
    dfs_mn = []
    dfs_mi = []
    dfs_mx = []
    for lead_int in range(5):
        df_mn, df_mi, df_mx = ep_process_data(
            region_id, season_str, lead_int, spi_string_name
        )
        dfs_mn.append(df_mn)
        dfs_mi.append(df_mi)
        dfs_mx.append(df_mx)
        print(f"done on {lead_int}")
    df_mn_final = pd.concat(dfs_mn, ignore_index=True)
    df_mi_final = pd.concat(dfs_mi, ignore_index=True)
    df_mx_final = pd.concat(dfs_mx, ignore_index=True)
    df_kmj_mam = pd.concat([df_mn_final, df_mi_final, df_mx_final], axis=0)
    df_kmj_mam["percentage_spi"] = df_kmj_mam[spi_string_name] * 100
    plot = probability_plot(df_kmj_mam, region_id, season_str)
    plot.save(f"{region_id}-{season_str}.pdf")

    df_mn_final, df_mi_final, df_mx_final, df_kmj_mam = [], [], [], []
    print(f"{region_id}-{season_str}-{spi_string_name}")

    region_id = 0
    season_str = "JJAS"
    spi_string_name = "spi4"
    dfs_mn = []
    dfs_mi = []
    dfs_mx = []
    for lead_int in range(4):
        df_mn, df_mi, df_mx = ep_process_data(
            region_id, season_str, lead_int, spi_string_name
        )
        dfs_mn.append(df_mn)
        dfs_mi.append(df_mi)
        dfs_mx.append(df_mx)
    df_mn_final = pd.concat(dfs_mn, ignore_index=True)
    df_mi_final = pd.concat(dfs_mi, ignore_index=True)
    df_mx_final = pd.concat(dfs_mx, ignore_index=True)
    df_kmj_jjas = pd.concat([df_mn_final, df_mi_final, df_mx_final], axis=0)
    df_kmj_jjas["percentage_spi"] = df_kmj_jjas[spi_string_name] * 100
    plot = probability_plot(df_kmj_jjas, region_id, season_str)
    plot.save(f"{region_id}-{season_str}.pdf")

    df_mn_final, df_mi_final, df_mx_final, df_kmj_jjas = [], [], [], []
    print(f"{region_id}-{season_str}-{spi_string_name}")

    region_id = 1
    season_str = "MAM"
    spi_string_name = "spi3"
    dfs_mn = []
    dfs_mi = []
    dfs_mx = []
    for lead_int in range(5):
        df_mn, df_mi, df_mx = ep_process_data(
            region_id, season_str, lead_int, spi_string_name
        )
        dfs_mn.append(df_mn)
        dfs_mi.append(df_mi)
        dfs_mx.append(df_mx)
    df_mn_final = pd.concat(dfs_mn, ignore_index=True)
    df_mi_final = pd.concat(dfs_mi, ignore_index=True)
    df_mx_final = pd.concat(dfs_mx, ignore_index=True)
    df_mbt_mam = pd.concat([df_mn_final, df_mi_final, df_mx_final], axis=0)
    df_mbt_mam["percentage_spi"] = df_mbt_mam[spi_string_name] * 100
    plot = probability_plot(df_mbt_mam, region_id, season_str)
    plot.save(f"{region_id}-{season_str}.pdf")

    df_mn_final, df_mi_final, df_mx_final, df_mbt_mam = [], [], [], []
    print(f"{region_id}-{season_str}-{spi_string_name}")

    region_id = 1
    season_str = "OND"
    spi_string_name = "spi3"
    dfs_mn = []
    dfs_mi = []
    dfs_mx = []
    for lead_int in range(5):
        df_mn, df_mi, df_mx = ep_process_data(
            region_id, season_str, lead_int, spi_string_name
        )
        dfs_mn.append(df_mn)
        dfs_mi.append(df_mi)
        dfs_mx.append(df_mx)
    df_mn_final = pd.concat(dfs_mn, ignore_index=True)
    df_mi_final = pd.concat(dfs_mi, ignore_index=True)
    df_mx_final = pd.concat(dfs_mx, ignore_index=True)
    df_mbt_ond = pd.concat([df_mn_final, df_mi_final, df_mx_final], axis=0)
    df_mbt_ond["percentage_spi"] = df_mbt_ond[spi_string_name] * 100
    plot = probability_plot(df_mbt_ond, region_id, season_str)
    plot.save(f"{region_id}-{season_str}.pdf")

    df_mn_final, df_mi_final, df_mx_final, df_mbt_ond = [], [], [], []
    print(f"{region_id}-{season_str}-{spi_string_name}")

    region_id = 2
    season_str = "MAM"
    spi_string_name = "spi3"
    dfs_mn = []
    dfs_mi = []
    dfs_mx = []
    for lead_int in range(5):
        df_mn, df_mi, df_mx = ep_process_data(
            region_id, season_str, lead_int, spi_string_name
        )
        dfs_mn.append(df_mn)
        dfs_mi.append(df_mi)
        dfs_mx.append(df_mx)
    df_mn_final = pd.concat(dfs_mn, ignore_index=True)
    df_mi_final = pd.concat(dfs_mi, ignore_index=True)
    df_mx_final = pd.concat(dfs_mx, ignore_index=True)
    df_wjr_mam = pd.concat([df_mn_final, df_mi_final, df_mx_final], axis=0)
    df_wjr_mam["percentage_spi"] = df_wjr_mam[spi_string_name] * 100
    plot = probability_plot(df_wjr_mam, region_id, season_str)
    plot.save(f"{region_id}-{season_str}.pdf")

    df_mn_final, df_mi_final, df_mx_final, df_wjr_mam = [], [], [], []
    print(f"{region_id}-{season_str}-{spi_string_name}")

    region_id = 2
    season_str = "OND"
    spi_string_name = "spi3"
    dfs_mn = []
    dfs_mi = []
    dfs_mx = []
    for lead_int in range(5):
        df_mn, df_mi, df_mx = ep_process_data(
            region_id, season_str, lead_int, spi_string_name
        )
        dfs_mn.append(df_mn)
        dfs_mi.append(df_mi)
        dfs_mx.append(df_mx)
    df_mn_final = pd.concat(dfs_mn, ignore_index=True)
    df_mi_final = pd.concat(dfs_mi, ignore_index=True)
    df_mx_final = pd.concat(dfs_mx, ignore_index=True)
    df_wjr_ond = pd.concat([df_mn_final, df_mi_final, df_mx_final], axis=0)
    df_wjr_ond["percentage_spi"] = df_wjr_ond[spi_string_name] * 100
    plot = probability_plot(df_wjr_ond, region_id, season_str)
    plot.save(f"{region_id}-{season_str}.pdf")

    df_mn_final, df_mi_final, df_mx_final, df_wjr_ond = [], [], [], []
    print(f"{region_id}-{season_str}-{spi_string_name}")


def create_rule(df, lt_value):
    df1 = df[df["lt"] == lt_value]
    if not df1.empty:
        ext_val = (
            df1[df1["cat"] == "ext"]["decided_trigger"].iloc[0]
            if not df1[df1["cat"] == "ext"].empty
            else 0
        )
        rule_ext = (
            alt.Chart(pd.DataFrame({"percentage_spi": [ext_val]}))
            .mark_rule(
                strokeWidth=2,  # Set the thickness of the line
                stroke="#880203",  # Set the color of the line
            )
            .encode(y="percentage_spi:Q")
        )

        sev_val = (
            df1[df1["cat"] == "sev"]["decided_trigger"].iloc[0]
            if not df1[df1["cat"] == "sev"].empty
            else 0
        )
        rule_sev = (
            alt.Chart(pd.DataFrame({"percentage_spi": [sev_val]}))
            .mark_rule(
                strokeWidth=2,  # Set the thickness of the line
                stroke="#fffe00",  # Set the color of the line
            )
            .encode(y="percentage_spi:Q")
        )

        mod_val = (
            df1[df1["cat"] == "mod"]["decided_trigger"].iloc[0]
            if not df1[df1["cat"] == "mod"].empty
            else 0
        )
        rule_mod = (
            alt.Chart(pd.DataFrame({"percentage_spi": [mod_val]}))
            .mark_rule(
                strokeWidth=2,  # Set the thickness of the line
                stroke="#ffa400",  # Set the color of the line
            )
            .encode(y="percentage_spi:Q")
        )

        return rule_ext + rule_sev + rule_mod
    else:
        return alt.Chart(pd.DataFrame({"percentage_spi": [0]})).mark_rule(opacity=0)


def temp_obs_fct_bar_plot(
    df, odf, region_id, season_str, spi_string_name, threshold_dict
):
    if region_id == 0:
        region_name = "Karamoja"
    if region_id == 1:
        region_name = "Marsabit"
    if region_id == 2:
        region_name = "Wajir"
    row_annotations = [
        alt.Chart(
            pd.DataFrame(
                {
                    "text": [
                        f"{region_name}-{season_str} lt=2, Mean of the pixel wise SPI observation and forecast Probablities for the region "
                    ]
                }
            )
        )
        .mark_text(
            align="left",
            baseline="middle",
            fontSize=12,
            fontWeight="bold",
            dx=-190,
            dy=-90,
        )
        .encode(text="text:N")
        .properties(width=400, height=200),
        alt.Chart(
            pd.DataFrame(
                {
                    "text": [
                        f"{region_name}-{season_str} lt=1, Mean of the pixel wise SPI observation and forecast Probablities for the region "
                    ]
                }
            )
        )
        .mark_text(
            align="left",
            baseline="middle",
            fontSize=12,
            fontWeight="bold",
            dx=-190,
            dy=-90,
        )
        .encode(text="text:N")
        .properties(width=400, height=200),
        alt.Chart(pd.DataFrame({"text": ["lt=2"]}))
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
        alt.Chart(pd.DataFrame({"text": ["lt=3"]}))
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
        alt.Chart(pd.DataFrame({"text": ["lt=4"]}))
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
        alt.Chart(pd.DataFrame({"text": ["lt=5"]})).mark_text(
            align="left",
            baseline="middle",
            fontSize=14,
            fontWeight="bold",
            dx=-190,
            dy=-90,
        ),
    ]

    # Update the color scale
    color_scale = alt.Scale(
        # domain=["ext", "sev", "mod"], range=["#880203", "#ffa400", "#fffe00"]
        domain=["mod", "sev", "ext"],
        range=["#fffe00", "#ffa400", "#880203"],
    )
    # Create the overlaid area chart using Altair
    base_plot = (
        alt.Chart(df)
        .mark_area()
        .encode(
            x=alt.X("year:N", axis=alt.Axis(labelAngle=90)),
            y=alt.Y("percentage_spi:Q", title="Probability (%)", stack=None),
            color=alt.Color("cat:N", scale=color_scale, sort=["sev", "mod", "ext"]),
        )
        .properties(width=400, height=200)
    )

    # ... (rest of the code for creating the panels remains the same)

    base_plot.configure_view(stroke=None).configure_axisY(
        labelFontSize=12, titleFontSize=14
    ).configure_axisX(labelFontSize=8, titleFontSize=12).configure_legend(
        labelFontSize=12, titleFontSize=14
    )
    if season_str == "JJAS":
        panel_1_mean = alt.layer(
            base_plot.transform_filter(
                (alt.datum.lt == 2) & (alt.datum.subset == "mean")
            ),
            row_annotations[0],
        ) + create_rule(df, 2)

        panel_2_mean = alt.layer(
            base_plot.transform_filter(
                (alt.datum.lt == 3) & (alt.datum.subset == "mean")
            ),
            row_annotations[3],
        ) + create_rule(df, 3)

        panel_3_mean = alt.layer(
            base_plot.transform_filter(
                (alt.datum.lt == 4) & (alt.datum.subset == "mean")
            ),
            row_annotations[4],
        ) + create_rule(df, 4)

        panel_4_mean = alt.layer(
            base_plot.transform_filter(
                (alt.datum.lt == 5) & (alt.datum.subset == "mean")
            ),
            row_annotations[5],
        ) + create_rule(df, 5)

    else:
        panel_1_mean = alt.layer(
            base_plot.transform_filter(
                (alt.datum.lt == 1) & (alt.datum.subset == "mean")
            ),
            row_annotations[1],
        ) + create_rule(df, 1)

        panel_2_mean = alt.layer(
            base_plot.transform_filter(
                (alt.datum.lt == 2) & (alt.datum.subset == "mean")
            ),
            row_annotations[2],
        ) + create_rule(df, 2)

        panel_3_mean = alt.layer(
            base_plot.transform_filter(
                (alt.datum.lt == 3) & (alt.datum.subset == "mean")
            ),
            row_annotations[3],
        ) + create_rule(df, 3)

        panel_4_mean = alt.layer(
            base_plot.transform_filter(
                (alt.datum.lt == 4) & (alt.datum.subset == "mean")
            ),
            row_annotations[4],
        ) + create_rule(df, 4)

        panel_5_mean = alt.layer(
            base_plot.transform_filter(
                (alt.datum.lt == 5) & (alt.datum.subset == "mean")
            ),
            row_annotations[5],
        ) + create_rule(df, 5)

    #####
    obs_plot = (
        alt.Chart(odf)
        .mark_bar()
        .encode(
            x=alt.X("year:N", axis=alt.Axis(labelAngle=90)),
            y=alt.Y(f"{spi_string_name}:Q", title=f"{spi_string_name}", stack=None),
            color=alt.condition(
                alt.datum.value > 0,
                alt.value("orange"),  # Color for positive values
                alt.value("blue"),  # Color for negative values
            ),
        )
        .properties(width=400, height=200)
    )

    obs_plot.configure_view(stroke=None).configure_axisY(
        labelFontSize=12, titleFontSize=14
    ).configure_axisX(labelFontSize=8, titleFontSize=12).configure_legend(
        labelFontSize=12, titleFontSize=14
    )

    # Create horizontal lines at y-axis values -1, 0, and 1 with different colors and thicknesses
    rule_ext = (
        alt.Chart(pd.DataFrame({"y": [threshold_dict["ext"]]}))
        .mark_rule(
            strokeWidth=2,  # Set the thickness of the line
            stroke="#880203",  # Set the color of the line
        )
        .encode(y="y:Q")
    )

    rule_mod = (
        alt.Chart(pd.DataFrame({"y": [threshold_dict["mod"]]}))
        .mark_rule(
            strokeWidth=2,  # Set the thickness of the line
            stroke="#ffa400",  # Set the color of the line
        )
        .encode(y="y:Q")
    )

    rule_sev = (
        alt.Chart(pd.DataFrame({"y": [threshold_dict["sev"]]}))
        .mark_rule(
            strokeWidth=2,  # Set the thickness of the line
            stroke="#fffe00",  # Set the color of the line
        )
        .encode(y="y:Q")
    )

    chart_obs = obs_plot + rule_ext + rule_mod + rule_sev

    emtpy_plot = (
        alt.Chart(pd.DataFrame({"A": []}))
        .mark_text()
        .encode()
        .properties(width=400, height=200)
    )
    if season_str == "JJAS":
        panels = alt.vconcat(
            alt.hconcat(chart_obs, panel_1_mean),
            alt.hconcat(emtpy_plot, panel_2_mean),
            alt.hconcat(emtpy_plot, panel_3_mean),
            alt.hconcat(emtpy_plot, panel_4_mean),
        )
    else:
        panels = alt.vconcat(
            alt.hconcat(chart_obs, panel_1_mean),
            alt.hconcat(emtpy_plot, panel_2_mean),
            alt.hconcat(emtpy_plot, panel_3_mean),
            alt.hconcat(emtpy_plot, panel_4_mean),
            alt.hconcat(emtpy_plot, panel_5_mean),
        )

    panels.configure_view(stroke=None).configure_axisY(
        labelFontSize=12, titleFontSize=14
    ).configure_axisX(labelFontSize=10, titleFontSize=12).configure_legend(
        labelFontSize=12, titleFontSize=14
    )

    return panels


def temp_mises_ep_obs_plot():
    mdb = pd.read_csv(f"{data_path}kimwa-metrix-v1.csv")
    lead_int = 1
    region_id = 0
    season_str = "MAM"
    spi_string_name = "spi3"
    sc_season_str = season_str.lower()

    ddf = decide_for_region_season(mdb, region_id, season_str)
    ddf1 = ddf[["identify", "percentage_spi"]]
    ddf1.columns = ["identify", "decided_trigger"]
    ep_df = get_ep_for_region_season(mdb, region_id, season_str)
    ep_df1 = ep_df[["identify", "cat", "lt", "subset", "year", "percentage_spi"]]
    df = pd.merge(ddf1, ep_df1, on="identify")

    obs_data, ens_data, a_fc, a_obs = make_obs_fct_dataset(
        region_id, season_str, lead_int
    )
    threshold_dict = get_threshold(region_id, sc_season_str)
    odf = mean_obs_spi(obs_data, spi_string_name)

    barplot = temp_obs_fct_bar_plot(
        df, odf, region_id, season_str, spi_string_name, threshold_dict
    )

    barplot.save(f"{data_path}{region_id}-{sc_season_str}.pdf")

    # barplot=[]

    region_id = 0
    season_str = "JJAS"
    spi_string_name = "spi4"
    sc_season_str = season_str.lower()

    ddf = decide_for_region_season(mdb, region_id, season_str)
    ddf1 = ddf[["identify", "percentage_spi"]]
    ddf1.columns = ["identify", "decided_trigger"]
    ep_df = get_ep_for_region_season(mdb, region_id, season_str)
    ep_df1 = ep_df[["identify", "cat", "lt", "subset", "year", "percentage_spi"]]
    df = pd.merge(ddf1, ep_df1, on="identify")

    obs_data, ens_data, a_fc, a_obs = make_obs_fct_dataset(
        region_id, season_str, lead_int
    )
    threshold_dict = get_threshold(region_id, sc_season_str)
    odf = mean_obs_spi(obs_data, spi_string_name)

    barplot = temp_obs_fct_bar_plot(
        df, odf, region_id, season_str, spi_string_name, threshold_dict
    )
    barplot.save(f"{data_path}{region_id}-{sc_season_str}.pdf")

    region_id = 1
    season_str = "MAM"
    spi_string_name = "spi3"
    sc_season_str = season_str.lower()

    ddf = decide_for_region_season(mdb, region_id, season_str)
    ddf1 = ddf[["identify", "percentage_spi"]]
    ddf1.columns = ["identify", "decided_trigger"]
    ep_df = get_ep_for_region_season(mdb, region_id, season_str)
    ep_df1 = ep_df[["identify", "cat", "lt", "subset", "year", "percentage_spi"]]
    df = pd.merge(ddf1, ep_df1, on="identify")

    obs_data, ens_data, a_fc, a_obs = make_obs_fct_dataset(
        region_id, season_str, lead_int
    )
    threshold_dict = get_threshold(region_id, sc_season_str)
    odf = mean_obs_spi(obs_data, spi_string_name)

    barplot = temp_obs_fct_bar_plot(
        df, odf, region_id, season_str, spi_string_name, threshold_dict
    )
    barplot.save(f"{data_path}{region_id}-{sc_season_str}.pdf")

    region_id = 1
    season_str = "OND"
    spi_string_name = "spi3"
    sc_season_str = season_str.lower()

    ddf = decide_for_region_season(mdb, region_id, season_str)
    ddf1 = ddf[["identify", "percentage_spi"]]
    ddf1.columns = ["identify", "decided_trigger"]
    ep_df = get_ep_for_region_season(mdb, region_id, season_str)
    ep_df1 = ep_df[["identify", "cat", "lt", "subset", "year", "percentage_spi"]]
    df = pd.merge(ddf1, ep_df1, on="identify")

    obs_data, ens_data, a_fc, a_obs = make_obs_fct_dataset(
        region_id, season_str, lead_int
    )
    threshold_dict = get_threshold(region_id, sc_season_str)
    odf = mean_obs_spi(obs_data, spi_string_name)

    barplot = temp_obs_fct_bar_plot(
        df, odf, region_id, season_str, spi_string_name, threshold_dict
    )
    barplot.save(f"{data_path}{region_id}-{sc_season_str}.pdf")

    region_id = 2
    season_str = "MAM"
    spi_string_name = "spi3"
    sc_season_str = season_str.lower()

    ddf = decide_for_region_season(mdb, region_id, season_str)
    ddf1 = ddf[["identify", "percentage_spi"]]
    ddf1.columns = ["identify", "decided_trigger"]
    ep_df = get_ep_for_region_season(mdb, region_id, season_str)
    ep_df1 = ep_df[["identify", "cat", "lt", "subset", "year", "percentage_spi"]]
    df = pd.merge(ddf1, ep_df1, on="identify")

    obs_data, ens_data, a_fc, a_obs = make_obs_fct_dataset(
        region_id, season_str, lead_int
    )
    threshold_dict = get_threshold(region_id, sc_season_str)
    odf = mean_obs_spi(obs_data, spi_string_name)

    barplot = temp_obs_fct_bar_plot(
        df, odf, region_id, season_str, spi_string_name, threshold_dict
    )
    barplot.save(f"{data_path}{region_id}-{sc_season_str}.pdf")

    region_id = 2
    season_str = "OND"
    spi_string_name = "spi3"
    sc_season_str = season_str.lower()

    ddf = decide_for_region_season(mdb, region_id, season_str)
    ddf1 = ddf[["identify", "percentage_spi"]]
    ddf1.columns = ["identify", "decided_trigger"]
    ep_df = get_ep_for_region_season(mdb, region_id, season_str)
    ep_df1 = ep_df[["identify", "cat", "lt", "subset", "year", "percentage_spi"]]
    df = pd.merge(ddf1, ep_df1, on="identify")

    obs_data, ens_data, a_fc, a_obs = make_obs_fct_dataset(
        region_id, season_str, lead_int
    )
    threshold_dict = get_threshold(region_id, sc_season_str)
    odf = mean_obs_spi(obs_data, spi_string_name)

    barplot = temp_obs_fct_bar_plot(
        df, odf, region_id, season_str, spi_string_name, threshold_dict
    )
    barplot.save(f"{data_path}{region_id}-{sc_season_str}.pdf")
