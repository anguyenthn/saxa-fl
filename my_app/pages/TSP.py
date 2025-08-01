import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Load data
df = pd.read_csv("NFL2014_2024.csv")

# Remove 2020 and 2021
df = df[~df["season"].isin([2020, 2021])]

st.title("NFL Team Travel Map")

# Sidebar filters
seasons = sorted(df["season"].unique())
selected_season = st.sidebar.selectbox("Select Season", seasons)

teams = sorted(df[df["season"] == selected_season]["posteam"].unique())
selected_team = st.sidebar.selectbox("Select Team", teams)

# Week filter (min to max)
weeks = sorted(df["week"].unique())
selected_weeks = st.sidebar.slider(
    "Select Week Range",
    min_value=min(weeks),
    max_value=max(weeks),
    value=(min(weeks), max(weeks))
)

# Filter data by season, team, and week range
team_data = df[
    (df["season"] == selected_season)
    & (df["posteam"] == selected_team)
    & (df["week"].between(selected_weeks[0], selected_weeks[1]))
].sort_values("week")

# Get colors for weeks
weeks_filtered = sorted(team_data["week"].unique())
palette = px.colors.qualitative.Dark24
color_map = {week: palette[i % len(palette)] for i, week in enumerate(weeks_filtered)}

# Create map figure
fig = go.Figure()
prev_lat, prev_lon = None, None

for _, row in team_data.iterrows():
    lat, lon, week = row["lat"], row["lon"], row["week"]
    rest_days = row["away_rest"] if row["location"] == "Away" else row["home_rest"]

    # Draw travel line and marker
    if prev_lat is not None and prev_lon is not None:
        fig.add_trace(go.Scattergeo(
            locationmode="USA-states",
            lon=[prev_lon, lon],
            lat=[prev_lat, lat],
            mode="lines+markers",
            line=dict(width=2, color=color_map[week]),
            marker=dict(size=6, color=color_map[week]),
            hoverinfo="text",
            text=(
                f"Week {week} ({row['weekday']}): "
                f"{row['away_team']} vs {row['home_team']}<br>"
                f"Stadium: {row['stadium']}<br>"
                f"Rest Days: {rest_days}<br>"
                f"Score: {row['away_score']} - {row['home_score']}"
            )
        ))

    # Short rest marker
    if rest_days < 5:
        fig.add_trace(go.Scattergeo(
            locationmode="USA-states",
            lon=[lon],
            lat=[lat],
            mode="markers",
            marker=dict(size=12, symbol="star", color="red"),
            hoverinfo="text",
            text=f"Short Rest (<5 days) - Week {week}: {row['stadium']}"
        ))

    prev_lat, prev_lon = lat, lon

# Map layout (USA only)
fig.update_layout(
    geo=dict(
        scope="usa",  # back to USA scope
        projection_type="albers usa",
        showland=True,
        landcolor="rgb(217, 217, 217)",
        subunitwidth=1,
        countrywidth=1,
        subunitcolor="rgb(255, 255, 255)",
        countrycolor="rgb(255, 255, 255)"
    ),
    margin={"r":0,"t":0,"l":0,"b":0},
    legend_title_text="Week Colors"
)

st.plotly_chart(fig, use_container_width=True)

# ---------- BAR CHART: Total distance traveled per team ----------
st.subheader("Total Distance Traveled by Each Team")

team_travel = (
    df[(df["season"] == selected_season)
       & (df["week"].between(selected_weeks[0], selected_weeks[1]))]
    .groupby("posteam")["away_travel_miles"]
    .sum()
    .reset_index()
    .sort_values("away_travel_miles", ascending=False)
)

bar_fig = px.bar(
    team_travel,
    x="posteam",
    y="away_travel_miles",
    text="away_travel_miles",
    title=f"Total Travel Distance ({selected_season}, Weeks {selected_weeks[0]}-{selected_weeks[1]})",
    labels={"away_travel_miles": "Total Miles"}
)

bar_fig.update_traces(texttemplate='%{text:.0f}', textposition="outside")
bar_fig.update_layout(xaxis_title="Team", yaxis_title="Miles Traveled")

st.plotly_chart(bar_fig, use_container_width=True)
