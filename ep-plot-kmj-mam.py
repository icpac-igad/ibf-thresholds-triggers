import altair as alt
import pandas as pd

from utils import get_threshold
from utils import make_obs_fct_dataset
from utils import emprical_probablity

from utils import mean_emp_prob
from utils import min_emp_prob
from utils import max_emp_prob

from utils import ep_process_data


region_id = 0
season_str = "MAM"

dfs_mn = []
dfs_mi = []
dfs_mx = []

for lead_int in range(6):
    df_mn, df_mi, df_mx = ep_process_data(region_id, season_str, lead_int)
    dfs_mn.append(df_mn)
    dfs_mi.append(df_mi)
    dfs_mx.append(df_mx)

# Concatenate the dataframes for each subset
df_mn_final = pd.concat(dfs_mn, ignore_index=True)
df_mi_final = pd.concat(dfs_mi, ignore_index=True)
df_mx_final = pd.concat(dfs_mx, ignore_index=True)

df = pd.concat([df_mn_final, df_mi_final, df_mx_final], axis=0)
df["spi3p"] = df["spi3"] * 100

row_annotations = [
    alt.Chart(pd.DataFrame({"text": ["lt=0"]}))
    .mark_text(
        align="left", baseline="middle", fontSize=14, fontWeight="bold", dx=-190, dy=-90
    )
    .encode(text="text:N")
    .properties(width=400, height=200),
    alt.Chart(pd.DataFrame({"text": ["lt=1"]}))
    .mark_text(
        align="left", baseline="middle", fontSize=14, fontWeight="bold", dx=-190, dy=-90
    )
    .encode(text="text:N")
    .properties(width=400, height=200),
    alt.Chart(pd.DataFrame({"text": ["lt=2"]}))
    .mark_text(
        align="left", baseline="middle", fontSize=14, fontWeight="bold", dx=-190, dy=-90
    )
    .encode(text="text:N")
    .properties(width=400, height=200),
    alt.Chart(pd.DataFrame({"text": ["lt=3"]}))
    .mark_text(
        align="left", baseline="middle", fontSize=14, fontWeight="bold", dx=-190, dy=-90
    )
    .encode(text="text:N")
    .properties(width=400, height=200),
    alt.Chart(pd.DataFrame({"text": ["lt=4"]}))
    .mark_text(
        align="left", baseline="middle", fontSize=14, fontWeight="bold", dx=-190, dy=-90
    )
    .encode(text="text:N")
    .properties(width=400, height=200),
    alt.Chart(pd.DataFrame({"text": ["lt=5"]}))
    .mark_text(
        align="left", baseline="middle", fontSize=14, fontWeight="bold", dx=-190, dy=-90
    )
    .encode(text="text:N")
    .properties(width=400, height=200),
]

# Create the overlaid area chart using Altair
base_plot = (
    alt.Chart(df)
    .mark_area()
    .encode(
        x=alt.X("year:N", axis=alt.Axis(labelAngle=90)),
        y=alt.Y("spi3p:Q", title="Probablity (%)", stack=None),
        color=alt.Color("cat:N", scale=alt.Scale(scheme="category10")),
    )
    .properties(width=400, height=200)
)

base_plot.configure_view(stroke=None).configure_axisY(
    labelFontSize=12, titleFontSize=14
).configure_axisX(labelFontSize=8, titleFontSize=12).configure_legend(
    labelFontSize=12, titleFontSize=14
)

panel_0_mean = alt.layer(
    base_plot.transform_filter((alt.datum.lt == "0") & (alt.datum.subset == "mean")),
    row_annotations[0],
)
panel_0_min = alt.layer(
    base_plot.transform_filter((alt.datum.lt == "0") & (alt.datum.subset == "min")),
)
panel_0_max = alt.layer(
    base_plot.transform_filter((alt.datum.lt == "0") & (alt.datum.subset == "max")),
)

panel_1_mean = alt.layer(
    base_plot.transform_filter((alt.datum.lt == "1") & (alt.datum.subset == "mean")),
    row_annotations[1],
)
panel_1_min = alt.layer(
    base_plot.transform_filter((alt.datum.lt == "1") & (alt.datum.subset == "min")),
)
panel_1_max = alt.layer(
    base_plot.transform_filter((alt.datum.lt == "1") & (alt.datum.subset == "max")),
)

panel_2_mean = alt.layer(
    base_plot.transform_filter((alt.datum.lt == "2") & (alt.datum.subset == "mean")),
    row_annotations[2],
)
panel_2_min = alt.layer(
    base_plot.transform_filter((alt.datum.lt == "2") & (alt.datum.subset == "min")),
)
panel_2_max = alt.layer(
    base_plot.transform_filter((alt.datum.lt == "2") & (alt.datum.subset == "max")),
)

panel_3_mean = alt.layer(
    base_plot.transform_filter((alt.datum.lt == "3") & (alt.datum.subset == "mean")),
    row_annotations[3],
)
panel_3_min = alt.layer(
    base_plot.transform_filter((alt.datum.lt == "3") & (alt.datum.subset == "min")),
)
panel_3_max = alt.layer(
    base_plot.transform_filter((alt.datum.lt == "3") & (alt.datum.subset == "max")),
)

panel_4_mean = alt.layer(
    base_plot.transform_filter((alt.datum.lt == "4") & (alt.datum.subset == "mean")),
    row_annotations[4],
)
panel_4_min = alt.layer(
    base_plot.transform_filter((alt.datum.lt == "4") & (alt.datum.subset == "min")),
)
panel_4_max = alt.layer(
    base_plot.transform_filter((alt.datum.lt == "4") & (alt.datum.subset == "max")),
)

panel_5_mean = alt.layer(
    base_plot.transform_filter((alt.datum.lt == "5") & (alt.datum.subset == "mean")),
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
    alt.hconcat(panel_5_mean, panel_5_min, panel_5_max),
)

panels.configure_view(stroke=None).configure_axisY(
    labelFontSize=12, titleFontSize=14
).configure_axisX(labelFontSize=10, titleFontSize=12).configure_legend(
    labelFontSize=12, titleFontSize=14
)


panels.save("chart.pdf")
