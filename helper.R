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

