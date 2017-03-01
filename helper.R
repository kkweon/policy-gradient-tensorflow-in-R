library(testthat)

DiscountReward = function(reward.list, discount.rate) {
    result = numeric(length(reward.list))
    total.sum = 0
    for(i in length(reward.list):1L) {
        total.sum = reward.list[[i]] + total.sum * discount.rate
        result[[i]] = total.sum
    }
    result
}

ChooseAction = function(sess, policy.tensor, input.placeholder, observation) {
    action.prob = sess$run(policy.tensor, feed_dict=dict(input.placeholder=observation))
    p = runif(1)
    if (p <= action.prob[[1]])
        action = 0
    else
        action = 1
    action
}

Test.ChooseAction = function() {
    with(tf$Session() %as% sess, {
        input = tf$placeholder(tf$float32, shape=shape(NULL, 4L))
        W = tf$Variable(tf$truncated_normal(shape(4L, 2L), dtype = tf$float32))
        policy = tf$matmul(input, W)
        observation = list(0.111, 0.222, 0.333, 0.444)
        sess$run(tf$global_variables_initializer()) 
        action = ChooseAction(sess, policy, input, ProcessState(observation))
        expect_that(action, testthat::is_more_than(0))
    })
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
Test.Main(c(Test.DiscountReward, Test.ChooseAction))