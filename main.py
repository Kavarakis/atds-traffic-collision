# %%
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import sqlite3
from mpl_toolkits.basemap import Basemap
from IPython.display import display, HTML

# %% [markdown]
# ## Goals
#
# The Goal of this EDA is to understand California Traffic Collision Data in details.
# Since the dataset is really large, we will aim to introduce various transformations and filter columns which are carrying sensible informations.
#
# Since each traffic accident can be dangerous, we will try to understand if it is possible to use this dataset to predict if accident was fatal from the raw dataset.
#

# %% [markdown]
# To achieve that first we need to get acquinted with the Dataset:
#

# %%
con = sqlite3.connect("./data/switrs.sqlite")

cur = con.cursor()

# %%
res = cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in res.fetchall()]
# Print available tables:
print("Tables: ", tables)

# %% [markdown]
# Let's try to visualize location frequency of these accidents in California map.
#
# We will introduce also a killed_victims heatmap so we can spot any patterns which may be obvious.
#

# %%
query = (
    "SELECT latitude, longitude, killed_victims "
    "FROM collisions "
    "WHERE latitude IS NOT NULL AND longitude IS NOT NULL AND strftime('%Y',collision_date) = '2019'"
)
map_df = pd.read_sql_query(query, con)

# %% [markdown]
# Since data is too large to be loaded into openStreet map, we are using data from 2019:
#

# %%
print(map_df.head())

map_df.info()

# %%
import plotly.express as px

color_scale = [(0, "orange"), (1, "red")]
fig = px.scatter_mapbox(
    map_df,
    lat="latitude",
    lon="longitude",
    hover_name="killed_victims",
    zoom=4.5,
    height=400,
    width=800,
    color="killed_victims",
    color_continuous_scale=color_scale,
    size="killed_victims",
    title="Location of coliisions in 2019",
)
fig.update_layout(mapbox_style="open-street-map")
fig.show()
del map_df

# %% [markdown]
# Let's explore collision data, notably how the data relates to time:
#

# %%
query = "SELECT * FROM collisions order by RANDOM() LIMIT 1000"
col_df = pd.read_sql_query(query, con)

# %%
print(col_df.columns)
display(HTML(col_df.head().to_html()))

# %% [markdown]
# Here we are trying to observe number of accidents throughout years:
#

# %%
query = """SELECT SUM(killed_victims) AS no_casualties,
                  COUNT(CASE WHEN killed_victims=0 THEN killed_victims END) as zero_casualties,
                    strftime('%Y',collision_date) AS year 
                        FROM collisions 
                            group by strftime('%Y',collision_date)"""
killed_df = pd.read_sql_query(query, con)

# %%
fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)
fig.set_size_inches(16, 8)
sns.barplot(data=killed_df, y="year", x="no_casualties", ax=ax1)
sns.barplot(data=killed_df, y="year", x="zero_casualties", ax=ax2)
ax1.grid()
ax2.grid()
fig.suptitle("Number of accidents with and without casualties during years")
plt.show()

# %% [markdown]
# In this case, we are observing frequency of accidents during a hours in a day:
#

# %%
query = """SELECT CASE WHEN killed_victims=0 THEN 0 ELSE 1  END as is_fatal,
                    strftime('%H',collision_time) AS hour 
                        FROM collisions order by hour asc """
killed_df = pd.read_sql_query(query, con)

# %%
# it is pretty skewed towards single value:
killed_df["is_fatal"].value_counts()

# %%
# Normalizing between 0 and 1  (min/max scaler)
temp_data = killed_df.groupby(["hour", "is_fatal"]).size().reset_index()
temp_data["count"] = None
plt.rcParams["figure.figsize"] = (16, 8)
temp_scaler = temp_data[temp_data["is_fatal"] == 0][0].copy()
temp = (temp_scaler - temp_scaler.min()) / (temp_scaler.max() - temp_scaler.min())
temp_data.loc[temp.index, "count"] = temp.values
temp_scaler = temp_data[temp_data["is_fatal"] == 1][0].copy()
temp = (temp_scaler - temp_scaler.min()) / (temp_scaler.max() - temp_scaler.min())
temp_data.loc[temp.index, "count"] = temp.values
sns.lineplot(data=temp_data, x="hour", y="count", hue="is_fatal", markers=True)
plt.grid()
plt.title("Comparison of fatality trends during hours in a day (Normalized)")
plt.show()
del temp_data

# %% [markdown]
# How does victim's degree of injury relate with collisions?
#

# %%
# Distribution of the number of accidents per each day of the week
query = """ 
SELECT collisions.case_id,collisions.collision_date,victims.victim_degree_of_injury
     FROM collisions 
     INNER JOIN victims on collisions.case_id = victims.case_id 
          INNER JOIN parties on collisions.case_id=parties.case_id 
          order by RANDOM() LIMIT 1000
"""
df = pd.read_sql_query(query, con)
df.info()

# %%
df.collision_date = pd.to_datetime(df.collision_date)
df["weekday"] = df.collision_date.dt.weekday
df.groupby("victim_degree_of_injury")["weekday"].agg(pd.Series.mode)
df["victim_degree_of_injury"].value_counts()

# %% [markdown]
# What is the most frequent week day for accidents?
#

# %%
sns.histplot(data=df, x="weekday", discrete=True, stat="percent")

# %% [markdown]
# If we take into consideration the weekend trens ( Fri,Sat,Sun), we see that 42.2% of accidents happens on a weekend
#

# %%
print(len(df[(df["weekday"] > 4) | (df["weekday"] < 1)]["weekday"]) / len(df))
del df

# %%
#   Average age of drivers who were at fault that involves another motor-vehicle
query = """
        SELECT parties.party_age, 
                  parties.party_sex 
        from parties 
        JOIN collisions ON parties.case_id = collisions.case_id
        WHERE parties.at_fault = 1
            AND party_type = 'driver'
            AND collisions.motor_vehicle_involved_with  IN ('motor vehicle on other roadway', 'other motor vehicle')
            AND (COALESCE(party_age, party_sex) IS NOT NULL 
                OR COALESCE(party_age, party_sex) != '')
        
        """

table = pd.read_sql_query(query, con)

# %%


# %%
# Pretty similar here
sns.boxplot(data=table, x="party_sex", y="party_age")

# %% [markdown]
# Which type collisions causes most of fatal accidents?
#

# %%
data = pd.read_sql_query(
    "SELECT * FROM parties JOIN collisions USING (case_id) order by RANDOM() LIMIT 1000",
    con,
)

# %%
data.info()
data["at_fault"].value_counts()

# %%
# Fill 'collision' null values with 0
df = data
collision_set = [
    "pedestrian_collision",
    "bicycle_collision",
    "motorcycle_collision",
    "truck_collision",
]

for i in collision_set:
    df[i].fillna(0, inplace=True)

# Show 'collision' columns
df[collision_set]
df[collision_set].mean().sort_values(ascending=False).plot(kind="bar")
plt.title("Percentages of total fatal collisions by collision type")
plt.xticks(rotation=45)
plt.grid()
plt.ylabel("Percentages (%)")

# %%
data[collision_set].info()

# %% [markdown]
# What is the most significat age interval to participate in an accident?
#

# %%
sns.kdeplot(data=data, x="party_age")
plt.grid()
plt.title("A kernel density estimate (KDE) for age of parties in a collision")
print(
    "Most frequent age of collision party that is at fault is:",
    data["party_age"].mode().to_list()[0],
)

# %% [markdown]
# How does alcohol relates to parties at fault in a collision?
#

# %%
column_name = "alcohol_involved"
value_to_replace = 0

data[column_name].fillna(value_to_replace, inplace=True)

data["alcohol_involved"] = data["alcohol_involved"].astype(np.int64)

value_count = value_count = data[column_name].value_counts()
null_value = data[column_name].isnull().sum()
print(value_count)
print("---------------------------")
print(f"null values in {column_name} : {null_value}")

# %%
sns.histplot(
    data=data, x="alcohol_involved", hue="at_fault", multiple="stack", stat="percent"
)

# %%
data["injured_victims"].value_counts()

# %%
data["control_device"].value_counts()

# %%
data["primary_collision_factor"].value_counts()

# %%
data["lighting"].hist()
plt.xticks(rotation=45)

# %%
data["motor_vehicle_involved_with"].value_counts()

# %%
data["is_fatal"] = data["killed_victims"] > 0

# %%
data["killed_victims"].value_counts()

# %%
sns.histplot(
    data=data,
    x="motorcycle_collision",
    hue="is_fatal",
    multiple="dodge",
    stat="percent",
)

# %%
feature_columns = [
    # qualified as a predictor
    "alcohol_involved",
    # qualified as a predictor - larger damage larger prob to induce fatal accident
    "collision_severity",
    # qualified as a predictor -show that time related predictors are showing patterns
    "collision_date",
    # qualified as a predictor -show that time related predictors are showing patterns
    "collision_time",
    # Should be part of feature importance test - showing if vehicle was controllable in time of accident
    "control_device",
    # this should be transformed as a boolean since most of the data skewed around true/false
    "injured_victims",
    # this is our target variable
    "killed_victims",
    #  # Should be part of feature importance test - could hold time pattern data
    "lighting",
    #  Should be part of feature importance test - participants info could lead to fatal accident
    "motor_vehicle_involved_with",
    # Should be part of feature importance test - could have larger prob
    "motorcycle_collision",
    "pcf_violation_category",
    "pedestrian_action",
    # shown relation between fatal and collision type
    "pedestrian_collision",
    # size of urban area raises probability for a collision
    "population",
    # Should be part of feature importance test - showing what was initial cause of collision
    "primary_collision_factor",
    # Goes together with weather conditions
    "road_surface",
    # shown relation between fatal and collision type
    "truck_collision",
    "type_of_collision",
    # Relevant condition since most of the accidents happen on a dry time,
    # since most people are more careful during poor weather conditions, due to reduced visibility
    "weather_1",
]

# %%


# %% [markdown]
#

# %% [markdown]
# ## Final plans and Action Items
#
# - Preprocess data: deal with missing data and categorize possible features, thus reducing their complexity
# - Use proposed features to derive feature importance tests upon couple of Supervised classification models ( KNN,ANN,RF)
# - Tweak the model
# - Remove limitations of downsampling and use the whole dataset
# - Report the results
# - Finish the paper
#

# %%
