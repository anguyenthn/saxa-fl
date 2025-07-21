library(nflreadr)

# Load schedules and scores for 2002-2024 season
# schedulesOutcome <- load_schedules(seasons=c(2002:2024))
# write.csv(schedulesOutcome,"ScheduleOutcomeData.csv")
# write.csv(dictionary_schedules,"scheduleOutcomeDict.csv")
# # Load nextGen for 2002-2024
# nextGenPassing<- load_nextgen_stats(seasons=c(2016:2024),stat_type=c("passing"))
# nextGenReceiving<- load_nextgen_stats(seasons=c(2016:2024),stat_type=c("receiving"))
# nextGenRushing<- load_nextgen_stats(seasons=c(2016:2024),stat_type=c("rushing"))
# write.csv(nextGenPassing,"NextGenPassing.csv")
# write.csv(nextGenReceiving,"NextGenReceiving.csv")
# write.csv(nextGenRushing,"NextGenRushing.csv")
# write.csv(dictionary_nextgen_stats,"NextGenDict.csv")

# # Load play by play data for 2002-2024
playByPlay2002_2007<- load_pbp(seasons=c(2002:2007))
write.csv(playByPlay2002_2007,"PlayByPlay2002_2007.csv")
playByPlay2008_2013<- load_pbp(seasons=c(2008:2013))
write.csv(playByPlay2008_2013,"PlayByPlay2008_2013.csv")
playByPlay2014_2019<- load_pbp(seasons=c(2014:2019))
write.csv(playByPlay2014_2019,"PlayByPlay2014_2019.csv")
playByPlay2020_2024<- load_pbp(seasons=c(2020:2024))
write.csv(playByPlay2020_2024,"PlayByPlay2020_2024.csv")
#write.csv(dictionary_pbp, "PlayByPlayDict.csv")