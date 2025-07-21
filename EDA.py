import pandas as pd
import matplotlib as plt
#import plotly as px
import numpy as np 
#import nfl_data_py as nfl

# years=[2002,2024]
# metrics=nfl.import_seasonal_data(years)
# metrics.to_csv('seasonal_metrics', index=False)
#Import Data
NextGenPassing=pd.read_csv('data/NextGenPassing.csv')
NextGenReceiving=pd.read_csv('data/NextGenReceiving.csv')
NextGenRushing=pd.read_csv('data/NextGenRushing.csv')
ScheduleOutcome=pd.read_csv('data/scheduleOutcomeData.csv')
PlayByPlay2002_2007=pd.read_csv('data/PlayByPlay2002_2007.csv')

unique_stadiums=ScheduleOutcome['stadium'].dropna().unique()

#print(unique_stadiums)
# PlayByPlay=pd.read_csv('data/PlayByPlay.csv')
# PlayByPlay=pd.read_csv('data/PlayByPlay.csv')
# PlayByPlay=pd.read_csv('data/PlayByPlay.csv')
# By game by each team
# Assumptions 
NextGenPassing=NextGenPassing[NextGenPassing['season_type']=='REG']
NextGenReceiving=NextGenReceiving[NextGenReceiving['season_type']=='REG']
NextGenRushing=NextGenRushing[NextGenRushing['season_type']=='REG']
ScheduleOutcome=ScheduleOutcome[ScheduleOutcome['game_type']=='REG']
PlayByPlay2002_2007=PlayByPlay2002_2007[PlayByPlay2002_2007['season_type']=='REG']

## Pass Attempts (passed or not)
PlayByPlay=PlayByPlay2002_2007[PlayByPlay2002_2007['season'].isin([2002,2003,2004,2005,2006,2007])]
passAttemptsPerGame2002=PlayByPlay2002_2007[PlayByPlay2002_2007['pass_attempt']==1].groupby(['game_id','posteam','season']).size().reset_index(name='pass_attempts')
#passAttemptsPerGame2002.to_csv('PassAttemptsPerGame2002.csv',index=False)


## Passing Yards (Total passing yards) 
passingYardsPerGame2002=PlayByPlay2002_2007[PlayByPlay2002_2007['passing_yards'].notnull()].groupby(['game_id','posteam','season'])['passing_yards'].sum().reset_index(name='passing_yards')
#passingYardsPerGame2002.to_csv('passingYardsPerGame2002.csv',index=False)

## CP (Completion probability)
#completionProbabilityPerGame2002=PlayByPlay2002_2007[PlayByPlay2002_2007['cp']==1].groupby(['game_id','posteam']).size().reset_index(name='cp')
#completionProbabilityPerGame2002.to_csv('completionProbabilityPerGame2002.csv',index=False)

## Air Yards (yds to point of reception)

## Yards after Catch ()

## Rushing Yards
rushingYardsPerGame2002=PlayByPlay2002_2007[PlayByPlay2002_2007['rushing_yards'].notnull()].groupby(['game_id','posteam','season'])['rushing_yards'].sum().reset_index(name='rushing_yards')
## Receiving Yards

## Yards Gained 

## Is International Game
intl_games=[
    'Wembley Stadium',
    'Tottenham Stadium',
    'Twickenham Stadium',
    'Azteca Stadium',
    'Rogers Centre',
    'Allianz Arena',
    'Deutsche Bank Park',
    'Arena Corinthians'
]

ScheduleOutcome['is_international']=ScheduleOutcome['stadium'].isin(intl_games).astype(int)

## The first international game date 2005

## Is it a Thursday Game
ScheduleOutcome['is_thursday']=(ScheduleOutcome['weekday']=='Thursday').astype(int)
#ScheduleOutcome['is_away']=(ScheduleOutcome['away']=='Thursday').astype(int)

## Merged Metrics
NFL2002metrics = passAttemptsPerGame2002.merge(
    passingYardsPerGame2002, on=['game_id', 'posteam','season'], how='outer'
).merge(
    rushingYardsPerGame2002, on=['game_id', 'posteam','season'], how='outer'
)


## Merged Metrics to Schedule Data
NFL2002 = pd.merge(
    NFL2002metrics,
    ScheduleOutcome[['game_id','away_rest','home_rest','stadium_id','stadium','weekday','home_team','away_team',
                     'home_score','away_score','is_international','location','is_thursday']],
    on='game_id',
    how='left'
)



# Step 4: Save to CSV
NFL2002.to_csv('NFL2002_2007.csv', index=False)

