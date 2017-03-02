library(tensorflow)
library(gym)

source("helpers/tensorflow_graph.R")
source("helpers/main_methods.R")

remote_base = "http://127.0.0.1:5000"
client = create_GymClient(remote_base)
env_name = "CartPole-v0"
instance_id = env_create(client, env_name)

log_dir = file.path(getwd(), "log")
env_monitor_start(client, instance_id, log_dir, force = T, resume = F)
episode_count = 10000
max_steps = 300
model_path = 'model/model.ckpt'


tf$reset_default_graph()
policy.graph = PolicyGraphBuilder()
value.graph = ValueGraphBuilder()

saver = tf$train$Saver()

sess = tf$InteractiveSession()
init = tf$global_variables_initializer()
sess$run(init)

if(file.exists(file.path('model', 'checkpoint'))) {
    saver$restore(sess, model_path)
    cat("\nModel Restored")
} else {
    cat("\nNew Model")
}

for (i in 1:episode_count) {
    output = RunEpisode(client, instance_id, policy.graph, sess, timestep = max_steps)
    # $obs, rewards, actions
    result = ProcessMemory(output, discount.rate = 0.9, value.graph, sess)
    # $advantages, values.true
    
    # Train Value
    value.input.states = value.graph$input$states
    value.input.true.values = value.graph$input$true_values
    sess$run(value.graph$op$train_op, 
             feed_dict=dict(value.input.states=output$obs,
                            value.input.true.values=result$values.true))
    # Train Policy
    input.states = policy.graph$input$states
    input.advant = policy.graph$input$advantages
    input.action = policy.graph$input$actions
    sess$run(policy.graph$op$train_op,
             feed_dict=dict(input.states=output$obs,
                            input.advant=result$advantages,
                            input.action=output$actions))
    
    episode_reward = sum(output$rewards)
    cat(sprintf("\n[ Episode: %d ] Reward: %d", i, episode_reward))
}
saver$save(sess, model_path)


# Dump result info to disk
env_monitor_close(client, instance_id)
env_close(client, instance_id)
