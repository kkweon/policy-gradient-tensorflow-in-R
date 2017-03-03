remote_base = "http://127.0.0.1:5000"
client = create_GymClient(remote_base)
env_name = "CartPole-v0"
instance_id = env_create(client, env_name)
test_log_dir = file.path(getwd(), "log", "test")
env_monitor_start(client, instance_id, directory = test_log_dir, force = T, resume = F)
N = 200
result = numeric(N)
for(i in 1:N){
    out = RunEpisode(client, 
                     instance_id,
                     policy.graph,
                     sess,
                     timestep = 300,
                     bad.reward = 0,
                     render=T)
    
    
    cat(sprintf("\nReward (total): %d", sum(out$rewards)))
    
    result[i] = sum(out$rewards)
    
}

cat(sprintf("\nAverage Rewards: %.2f", mean(result)))
env_monitor_close(client, instance_id)
env_close(client, instance_id)
