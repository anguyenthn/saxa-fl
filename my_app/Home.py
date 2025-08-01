import streamlit as st
import pandas as pd
import plotly.express as px  

# Load data
df = pd.read_csv("NFL2014_2024.csv")

# Remove 2020 and 2021
df = df[~df["season"].isin([2020, 2021])]


st.title("NFL Analysis on International, Thursday Night, and Away Games (2014–2024)")
## Is it a Away game
df['is_away']=(df['posteam']==df['away_team']).astype(int)

# Sidebar Filters
teams = df['posteam'].dropna().unique()
metrics = ['pass_attempts', 'passing_yards', 'rushing_yards']
seasons = sorted(df['season'].dropna().unique())

selected_team = st.sidebar.selectbox("Select Team", sorted(teams))  
selected_metric = st.sidebar.selectbox("Select Game Metric", metrics)
selected_season = st.sidebar.selectbox("Select Season", seasons)

# Filter data
filtered_df = df[
    (df['posteam'] == selected_team) &
    (df['season'] == selected_season)
]
st.sidebar.markdown("""**Note:** 
International Data Begins in 2005 and there are 14 Thursday night teams a season (some teams can get multiple and some can get none) """)
# International Games 
st.subheader("International Games")
intl_games = filtered_df[filtered_df['is_international'] == 1]

if not intl_games.empty:
    fig = px.bar(
        filtered_df,
        x='game_id',
        y=selected_metric,
        title=f"{selected_team} – {selected_metric.replace('_', ' ').title()} in International Games",
        labels={'game_id': 'Game ID', selected_metric: selected_metric.replace('_', ' ').title()},
        color='is_international'
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.write("No international games found for this team and season.")

#  Thursday Night Games
st.subheader("Thursday Night Games")
thursday_games = filtered_df[filtered_df['is_thursday'] == 1]

if not thursday_games.empty:
    fig2 = px.bar(
        filtered_df,
        x='game_id',
        y=selected_metric,
        title=f"{selected_team} – {selected_metric.replace('_', ' ').title()} on Thursday Night",
        labels={'game_id': 'Game ID', selected_metric: selected_metric.replace('_', ' ').title()},
        color='is_thursday'
    )
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.write("No Thursday Night games found for this team and season.")

# Plot 3: Away Games
st.subheader("Away Games")
away_games = filtered_df[filtered_df['is_away'] == 1]

if not away_games.empty:
    fig3 = px.bar(
        filtered_df,
        x='game_id',
        y=selected_metric,
        title=f"{selected_team} – {selected_metric.replace('_', ' ').title()} in Away Games",
        labels={'game_id': 'Game ID', selected_metric: selected_metric.replace('_', ' ').title()},
        color='is_away'
    )
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.write("No away games found for this team and season.")
