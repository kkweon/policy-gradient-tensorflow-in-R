library(tensorflow)

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

ToList = function(List, key1, key2, val) {
    # Add value to List and return a new list
    #
    # Args:
    #   - List: any list
    #   - key1: string
    #   - key2: string
    #   - val: value
    #
    # Returns:
    #   - List: list
    #
    # Example:
    #   > List = list()
    #   > List = ToList(List, "abc", "def", "foo")
    #   List[["abc]][["def"]] == "foo"
    tmp = list()
    tmp[[key2]] = val
    List[[key1]] = c(List[[key1]], tmp)
    List
}



tf$reset_default_graph()
PolicyGradientBuilder = function(lr=0.01, scope="policy") {
    # Build a Poligy Gradient Graph
    # Args:
    #   - lr (float): learning rate
    #   - scope (character): name of the graph
    # Returns:
    #   - out (list): output of the graph
    out = list()
    
    with(tf$variable_scope(scope), {
        with(tf$name_scope("input"), {
            states = tf$placeholder(tf$float32, shape(NULL, 4L), name='state')
            actions = tf$placeholder(tf$float32, shape(NULL, 2L), name='actions')
            advantages = tf$placeholder(tf$float32, shape(NULL, 1L), name='advantages')
            
            # Add to output    
            out = toList(out, 'input', 'states', states)
            out = toList(out, 'input', 'actions', actions)
            out = toList(out, 'input', 'advantages', advantages)
        })
        
        with(tf$variable_scope("op"), {
            # W * states + b -> (N, 2)
            net = tf$contrib$layers$linear(states, num_outputs=2L, scope="fc")
            # Each action probability -> (N, 2)
            prob = tf$nn$softmax(net) 
            # Keep real actions -> (N, 2)
            net = tf$multiply(prob, actions) 
            # Reduce dimension -> (N, 1)
            net = tf$reduce_sum(net, axis = 1L) 
            
            # Log Probability 
            net = tf$log(net) 
            net = tf$multiply(net, advantages)
            net = tf$reduce_sum(net)
            loss = tf$negative(net)
            train_op = tf$train$AdamOptimizer(lr)$minimize(loss)
            
            out = ToList(out, 'op', 'prob', prob) 
            out = ToList(out, 'op', 'loss', loss)
            out = ToList(out, 'op', 'train_op', train_op)
        })
    })
    return(out)
}

ValueGradientBuilder = function(hidden_dim=10L, lr=0.01, scope="value") {
    with(tf$variable_scope(scope), {
        out = list()
        
        with(tf$name_scope("input"), {
            states = tf$placeholder(tf$float32, shape(NULL, 4L), name='states')
            true.values = tf$placeholder(tf$float32, shape(NULL, 1L), name='true_values')
            
            out = ToList(out, "input", "states", states)
            out = ToList(out, "input", "true_values", true.values)
        }) 
        
        with(tf$variable_scope("op"), {
            net = tf$contrib$layers$fully_connected(states,     # input dim
                                                    hidden_dim, # output dim
                                                    activation_fn=tf$nn$relu,
                                                    scope='fc1')
            values = tf$contrib$layers$fully_connected(net, # input dim
                                                    1L,  # output dim
                                                    activation_fn=NULL,
                                                    scope='fc2')
            loss = tf$subtract(values, true.values)
            loss = tf$square(loss)
            loss = tf$reduce_sum(loss)
            train_op = tf$train$AdamOptimizer(lr)$minimize(loss)
            
            out = ToList(out, 'op', 'values', values)
            out = ToList(out, 'op', 'loss', loss)
            out = ToList(out, 'op', 'train_op', train_op)
        })
        return(out)
    })
}

