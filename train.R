library(tensorflow)
library(gym)

remote_base = "http://127.0.0.1:5000"
client = create_GymClient(remote_base)
env_name = "CartPole-v0"
instance_id = env_create(client, env_name)

log_dir = file.path(getwd(), "log")
env_monitor_start(client, instance_id, log_dir, force = T, resume = F)
episode_count <- 1000
max_steps <- 5000


tf$reset_default_graph()
policy.grad = PolicyGradientBuilder()
value.grad = ValueGradientBuilder()

sess = tf$InteractiveSession()
init = tf$global_variables_initializer()
sess$run(init)



for (i in 1:episode_count) {
    output = RunEpisode(client, instance_id, policy.grad, sess, timestep = max_steps)
    # $obs, rewards, actions
    result = ProcessMemory(output, discount.rate = 0.9, value.grad, sess)
    # $advantages, values.true
    
    # Train Value
    value.input.states = value.grad$input$states
    value.input.true.values = value.grad$input$true_values
    sess$run(value.grad$op$train_op, 
             feed_dict=dict(value.input.states=output$obs,
                            value.input.true.values=result$values.true))
    # Train Policy
    input.states = policy.grad$input$states
    input.advant = policy.grad$input$advantages
    input.action = policy.grad$input$actions
    sess$run(policy.grad$op$train_op,
             feed_dict=dict(input.states=output$obs,
                            input.advant=result$advantages,
                            input.action=output$actions))
    
    episode_reward = sum(output$rewards)
    cat(sprintf("\n[ Episode: %d ] Reward: %d", i, episode_reward))
}


# Dump result info to disk
env_monitor_close(client, instance_id)
env_close(client, instance_id)
