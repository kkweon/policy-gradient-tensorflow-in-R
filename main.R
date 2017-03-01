library(tensorflow)
library(gym)


remote_base = "http://127.0.0.1:5000"
client = create_GymClient(remote_base)
env_name = "CartPole-v0"
instance_id = env_create(client, env_name)

log_dir = file.path(getwd(), "log")
env_monitor_start(client, instance_id, log_dir, force = T, resume = F)
episode_count <- 1
max_steps <- 200
reward <- 0
done <- FALSE
memory = list()
for (i in 1:episode_count) {
    ob <- env_reset(client, instance_id)
    
    for (i in 1:max_steps) {
        action <- env_action_space_sample(client, instance_id)
        results <- env_step(client, instance_id, action, render = FALSE)
        # List(observation, reward, done, info)
        memoery        
        if (results[["done"]]) break
    }
}

# Dump result info to disk
env_monitor_close(client, instance_id)

env_close(client, instance_id)
