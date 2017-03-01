remote_base = "http://127.0.0.1:5000"
client = create_GymClient(remote_base)
env_name = "CartPole-v0"
instance_id = env_create(client, env_name)
obs = RunEpisode(client, 
                 instance_id,
                 policy.grad,
                 sess,
                 timestep = 10000,
                 render=TRUE)
cat(sprintf("\nReward (total): %d", sum(obs$rewards)))
env_close_all()
