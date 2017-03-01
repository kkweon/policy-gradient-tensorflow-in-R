library(testthat)

ProcessState = function(observation.List) {
    # Convert a single state list to a matrix
    #
    # Args:
    #   - observation.List: List[[i]] = Value
    #
    # Returns: 
    #   - (1, 4) shape matrix
    # 
    # Example:
    #   > state = list(val1, val2, val3, val4)
    #   > ProcessState(state)
    #        [,1] [,2] [,3] [,4]
    #   [1,] val1 val2 val3 val4   
    t(as.matrix(unlist(observation.List)))
}

DiscountReward = function(reward.matrix, discount.rate) {
    result = numeric(length(reward.matrix))
    total.sum = 0
    for(i in length(reward.matrix):1L) {
        total.sum = reward.matrix[[i]] + total.sum * discount.rate
        result[[i]] = total.sum
    }
    dim(result) = dim(reward.matrix)
    result
}

ChooseAction = function(sess, policy.prob, input.placeholder, observation) {
    action.prob = sess$run(policy.prob, feed_dict=dict(input.placeholder=observation))
    p = runif(1)
    if (p <= action.prob[[1]])
        action = 0
    else
        action = 1
    action
}

RunEpisode = function(client, instance.id, policy.grad, sess, timestep=500, render=F) {
    # Run single episode (game)
    # Args:
    #   - client (GymClient): from create_GymClient
    #   - instance.id (int): from env_create
    #   - policy.grad (list): policy gradient graph
    #   - value.grad (list): value gradient graph
    #   - sess (tf$Session): Tensorflow Session
    #   - timestep (int): max time step of one episode
    # Returns:
    #   - result (list): list of actions
    #       $obs (matrix): observation matrix, shape(N, 4)
    #       $rewards (vector): reward vector after each action
    #       $actions (matrix): action one hot matrix, shape (N, 2)
    result = list()
    
    obs = env_reset(client, instance.id)
    obs = ProcessState(obs)
    for(i in 1:timestep) {
        action = ChooseAction(sess, 
                              policy.grad$op$prob, 
                              policy.grad$input$states,
                              obs)
        
        action_vector = numeric(2L)
        action_vector[action+1] = 1
        
        data = env_step(client, instance.id, action, render=render)
        
        result$obs = rbind(result$obs, obs, deparse.level = 0)
        result$rewards = rbind(result$rewards, data$reward, deparse.level = 0)
        result$actions = rbind(result$actions, action_vector, deparse.level = 0)
        
        if (data$done) {
            break
        }
        
        # Prepare Next Step 
        obs = ProcessState(data$observation)
    }
    
    result
}


ProcessMemory = function(memory.list, discount.rate, value.grad, sess) {
    result = list()
    states = memory.list$obs
    expect_equal(class(states), 'matrix')
    reward = memory.list$rewards
    expect_equal(class(reward), 'matrix')
    reward.discounted = DiscountReward(reward, discount.rate)
    expect_equal(dim(reward), dim(reward.discounted), info = "dim check fail")
   
    states.input = value.grad$input$states
    values.output = value.grad$op$values
    values.pred = sess$run(values.output, feed_dict=dict(states.input = states))
    
    expect_equal(dim(values.pred), dim(reward.discounted))
    result$advantages = reward.discounted - values.pred
    result$values.true = reward.discounted
    result
}
