if(!require(testthat)) {
    install.packages('testthat')
}
library(gym)
source('helpers/main_methods.R')
source("helpers/tensorflow_graph.R")

Test.ChooseAction = function() {
    with(tf$Session() %as% sess, {
        input = tf$placeholder(tf$float32, shape=shape(NULL, 4L))
        W = tf$Variable(tf$truncated_normal(shape(4L, 2L), dtype = tf$float32))
        policy = tf$matmul(input, W)
        observation = list(0.111, 0.222, 0.333, 0.444)
        sess$run(tf$global_variables_initializer()) 
        action = ChooseAction(sess, policy, input, ProcessState(observation))
        expect_gte(action, 0)
    })
}

Test.RunEpisode = function() {
    remote_base = "http://127.0.0.1:5000"
    env_name = "CartPole-v0"
    client = create_GymClient(remote_base)
    instance_id = env_create(client, env_name)
    policy.gradient = PolicyGradientBuilder()
    with(tf$Session() %as% sess, {
        init = tf$global_variables_initializer()
        sess$run(init)
        result = RunEpisode(client, instance_id, policy.gradient, sess)
    })
    expect_equal(names(result), c("obs", "rewards", "actions"))
    expect_equal(class(result[['obs']]), "matrix")
    expect_equal(class(result[['rewards']]), "matrix")
    expect_equal(class(result[['actions']]), "matrix")
}
env_close_all = function() {
    remote_base = "http://127.0.0.1:5000"
    env_name = "CartPole-v0"
    client = create_GymClient(remote_base)
    
    id_vectors = names(env_list_all(client))
    for(id in id_vectors)
        env_close(client, id)
}

Test.ProcessMemory = function() {
    with(tf$Session() %as% sess, {
        
        memory.list = list()
        memory.list$obs = matrix(runif(16), ncol=4)
        memory.list$rewards = matrix(1, nrow=4)
        discount.rate = 0
        value.grad = ValueGradientBuilder()
        
        init = tf$global_variables_initializer()
        sess$run(init)
        output = ProcessMemory(memory.list = memory.list,
                               discount.rate = discount.rate,
                               value.grad = value.grad,
                               sess = sess)
    })
    expect_equal(names(output), c("advantages", "values.true"))
    expect_equal(class(output$advantages), 'matrix')
    expect_equal(class(output$values.true), 'matrix')
    expect_equal(dim(output$advantages), c(4, 1)) 
    expect_equal(dim(output$values.true), c(4, 1))
}

Test.DiscountReward = function() {
    input = c(1, 1, 1, 1, 1)
    rate = 0
    output = DiscountReward(input, rate)
    expect_that(output, equals(c(1,1,1,1,1)))
    
    input = c(1, 1, 1)
    rate = 0.95
    output = DiscountReward(input, rate)
    expect = c(1 + 1 * rate + 1 * rate^2, 1 + 1*rate, 1)
    expect_that(output, equals(expect))
}

Test.Main = function(test.list) {
    for(func in test.list) {
        func()
    }
}
Test.Main(c(
    Test.DiscountReward, 
    Test.ChooseAction,
    Test.RunEpisode,
    Test.ProcessMemory))

env_close_all()