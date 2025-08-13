import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier  # or RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report

file_path = "/Users/nikesh/Downloads/NFLpbp_Schedule_Merged.csv"

data = pd.read_csv(file_path)

print(data.head())

data['year'] = data['game_id'].str.split('_').str[0].astype(int)

df = data[data['year'] >= 2021]

print(df.head())

# Create a DataFrame of column names and their dtypes
col_types = pd.DataFrame(df.dtypes, columns=["dtype"]).reset_index()
col_types.columns = ["column_name", "dtype"]

# Print nicely
for col, dtype in zip(col_types["column_name"], col_types["dtype"]):
    print(f"{col}: {dtype}")

# Data Cleaning

# Define columns to keep
cols_to_keep = [
    "home_team_x",         # object
    "away_team_x",         # object
    "week_x",              # int64
    "posteam",             # object
    "side_of_field",       # object 
    "season_x",            # assuming int
    "penalty_team",        # object
    "penalty_yards",       # numeric
    "drive_time_of_possession", # string/time format
    "away_score_x",        # numeric
    "home_score_x",        # numeric
    "roof_x",              # object
    "surface_x",           # object
    "temp_x",              # numeric
    "wind_x",              # numeric
    "game_stadium",        # object
    "away_rest",           # numeric
    "home_rest",            # numeric
    "is_thursday"          # binary
]

# Keep only these columns
df_filtered = df[cols_to_keep].copy()

print(df_filtered.head())

# Fixed NFL team mapping starting at 1
team_mapping = {
    'ARI': 1, 'ATL': 2, 'BAL': 3, 'BUF': 4, 'CAR': 5, 'CHI': 6, 'CIN': 7,
    'CLE': 8, 'DAL': 9, 'DEN': 10, 'DET': 11, 'GB': 12, 'HOU': 13, 'IND': 14,
    'JAX': 15, 'KC': 16, 'LV': 17, 'LAC': 18, 'LAR': 19, 'MIA': 20, 'MIN': 21,
    'NE': 22, 'NO': 23, 'NYG': 24, 'NYJ': 25, 'PHI': 26, 'PIT': 27,
    'SEA': 28, 'SF': 29, 'TB': 30, 'TEN': 31, 'WAS': 32
}

# Apply mapping
df_filtered["home_team_x"] = df_filtered["home_team_x"].map(team_mapping)
df_filtered["away_team_x"] = df_filtered["away_team_x"].map(team_mapping)
df_filtered["posteam"] = df_filtered["posteam"].map(team_mapping)
df_filtered["side_of_field"] = df_filtered["side_of_field"].map(team_mapping)
df_filtered["penalty_team"] = df_filtered["penalty_team"].map(team_mapping)

# Convert to category (factor equivalent in Pandas)
df_filtered["home_team_x"] = df_filtered["home_team_x"].astype("category")
df_filtered["away_team_x"] = df_filtered["away_team_x"].astype("category")
df_filtered["penalty_team"] = df_filtered["penalty_team"].astype("category")
df_filtered["side_of_field"] = df_filtered["side_of_field"].astype("category")
df_filtered["roof_x"] = df_filtered["roof_x"].astype("category")
df_filtered["surface_x"] = df_filtered["surface_x"].astype("category")
df_filtered["game_stadium"] = df_filtered["game_stadium"].astype("category")
df_filtered["posteam"] = df_filtered["posteam"].astype("category")

# Convert to binary: 1 = home team, 0 = away team
df_filtered["side_of_field_binary"] = (
    df_filtered["side_of_field"] == df_filtered["home_team_x"]
).astype(int)

df_filtered["posteam_binary"] = (
    df_filtered["posteam"] == df_filtered["home_team_x"]
).astype(int)

df_filtered["penalty_team_binary"] = (
    df_filtered["penalty_team"] == df_filtered["home_team_x"]
).astype(int)

# Drop the original string/categorical columns if not needed
df_filtered.drop(columns=["side_of_field", "posteam", "penalty_team"], inplace=True)

# drive possession time conversion
df_filtered["drive_time_of_possession_seconds"] = pd.to_timedelta(
    "00:" + df_filtered["drive_time_of_possession"]
).dt.total_seconds()

# Drop the original time of possession column
df_filtered.drop(columns=["drive_time_of_possession"], inplace=True)

print(df_filtered.head())

# Create a DataFrame of column names and their dtypes
col_types = pd.DataFrame(df_filtered.dtypes, columns=["dtype"]).reset_index()
col_types.columns = ["column_name", "dtype"]

# Print nicely
for col, dtype in zip(col_types["column_name"], col_types["dtype"]):
    print(f"{col}: {dtype}")

df_filtered = df_filtered.dropna().reset_index(drop=True)

### Train Model

# Drop target column (is thursday game)
X = df_filtered.drop("is_thursday", axis=1)
y = df_filtered["is_thursday"]

X = pd.get_dummies(
    X,
    columns=["roof_x", "surface_x", "game_stadium"],
    drop_first=True
)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)

# Model Initialization and Training
model_v1 = RandomForestClassifier(
    n_estimators=100,      # number of trees
    max_depth=None,        # can set to limit depth
    random_state=50
)
model_v1.fit(X_train, y_train)

# Predictions
y_pred = model_v1.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Feature Importance
importances = model_v1.feature_importances_

feature_importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': importances
}).sort_values(by='importance', ascending=False)

print(feature_importance_df)

feature_importance_df.to_csv('thursday_rf_feature_importance.csv', index=False)