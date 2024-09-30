import os
from dotenv import load_dotenv


from utils import temp_kimwa_metrices

from plot_utils import temp_kimwa_ep_plot

from plot_utils import temp_mises_ep_obs_plot

temp_mises_ep_obs_plot()


# to make the matrices calcualtion and create a large 11000 rows of csv file
df = temp_kimwa_metrices()
df.to_csv(f"{data_path}kimwa-metrix-v1.csv")

# to make plots of emprical probablity for the 87 combinations
# temp_kimwa_ep_plot()


load_dotenv()

data_path = os.getenv("data_path")

db = pd.read_csv(f"{data_path}kimwa-metrix-v1.csv")

"""
1. Filters rows with the maximum hanssen_kuipers_scores and auroc_scores above 0.5.
2. Further filters by trigger_values greater than 0.1 and bias_scores below 1.0, selecting those with the maximum bias_score and auroc_scores above 0.5.
3. Additionally, selects rows with the maximum heidke_skill_scores and auroc_scores above 0.5.
4. Finally, from the combined filtered set, selects rows with the maximum trigger_values.
"""
db1 = db[db["lt"] != 0]
# db2=db1[db1['season']=='JJAS' db1['lt']!=4]

mask = (db1["lt"] == 1) & (db1["season"] == "JJAS")

db2 = db1[~mask]
db2.info()


db3 = create_new_column(db2)

db3["identify"] = (
    db3["region_id"].astype(str) + "-" + db3["season"] + "-" + db3["new_column"]
)

db4 = db3.drop_duplicates("identify")
identify_list = db4["identify"].tolist()

d_odb = []

for idl in identify_list:
    odb = db3[db3["identify"] == idl]
    odb1 = get_subset(odb)
    odb2 = choose_row(odb1)
    d_odb.append(odb2)


df = pd.concat(d_odb)
df["pod_v"] = df.apply(lambda x: [x["hit_rate"], x["percentage_spi"]], axis=1)
df["far_v"] = df.apply(lambda x: [x["false_alarm_ratio"], x["percentage_spi"]], axis=1)


df["pod_v"] = df["pod_v"].apply(lambda x: round_list(x, 2))


mapping_dict = {0: "Karamoja", 1: "Marsabit", 2: "Wajir"}

df["region_x"] = df["region_id"].replace(mapping_dict)

p = df.pivot_table(
    index=["region_x", "season"], columns="new_column", values="pod_v", aggfunc="first"
)  # Assuming each combination is unique

# Reset the index to turn it back into columns
pf = p.reset_index()


# Apply the custom function to each cell in the DataFrame
pf1 = pf.applymap(replace_with_list)

pf1.columns.name = None

pf1["empty1"] = [[-999.0, -999.0]] * len(pf1)
pf1["empty2"] = [[-999.0, -999.0]] * len(pf1)


# #efg_col=['region_x', 'spi_prod_x', 'nov_x', 'dec_x', 'jan_x', 'feb_x', 'mar_x', 'jun_x', 'jul_x', 'aug_x', 'sep_x', 'oct_x', 'nov_y', 'dec_y', 'jan_y', 'feb_y', 'mar_y', 'jun_y', 'jul_y', 'aug_y', 'sep_y', 'oct_y', 'nov', 'dec', 'jan', 'feb', 'mar', 'jun', 'jul', 'aug', 'sep', 'oct']

efg_col1 = [
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
pf2 = pf1[efg_col1]
pf2

# pivoted_df1.to_csv(f'{data_path}pod.csv')

stat_var = "POD"
# df=thre_df_table_plot(stat_var)
# df.to_csv('output/tables/far.csv')
# df=pd.read_csv(f'{data_path}far.csv')
plot_data_table(pf2, stat_var)


df["far_v"] = df["far_v"].apply(lambda x: round_list(x, 2))


mapping_dict = {0: "Karamoja", 1: "Marsabit", 2: "Wajir"}

df["region_x"] = df["region_id"].replace(mapping_dict)

f = df.pivot_table(
    index=["region_x", "season"], columns="new_column", values="far_v", aggfunc="first"
)  # Assuming each combination is unique

# Reset the index to turn it back into columns
fp = f.reset_index()


# Apply the custom function to each cell in the DataFrame
fp1 = fp.applymap(replace_with_list)

fp1.columns.name = None

fp1["empty1"] = [[-999.0, -999.0]] * len(fp1)
fp1["empty2"] = [[-999.0, -999.0]] * len(fp1)


# #efg_col=['region_x', 'spi_prod_x', 'nov_x', 'dec_x', 'jan_x', 'feb_x', 'mar_x', 'jun_x', 'jul_x', 'aug_x', 'sep_x', 'oct_x', 'nov_y', 'dec_y', 'jan_y', 'feb_y', 'mar_y', 'jun_y', 'jul_y', 'aug_y', 'sep_y', 'oct_y', 'nov', 'dec', 'jan', 'feb', 'mar', 'jun', 'jul', 'aug', 'sep', 'oct']

efg_col1 = [
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
fp2 = fp1[efg_col1]
fp2

# pivoted_df1.to_csv(f'{data_path}pod.csv')

stat_var = "FAR"
# df=thre_df_table_plot(stat_var)
# df.to_csv('output/tables/far.csv')
# df=pd.read_csv(f'{data_path}far.csv')
plot_data_table(fp2, stat_var)
