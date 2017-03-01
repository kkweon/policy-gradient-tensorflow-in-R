library(ggplot2)

PlotRewardTimeline = function(filename) {
    tmp = jsonlite::fromJSON(filename)
    tmp = as.data.frame(tmp)
    ggplot(tmp, aes(1:nrow(tmp), episode_rewards)) + geom_point() + geom_smooth() + ggtitle("Reward after each episode run") + ylab("Reward per episode") + xlab("Episode")
}

