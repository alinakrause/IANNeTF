import tensorflow as tf

class WeightDrop(tf.keras.layers.Layer):
    """ Performs weight drop on specified weight matrices of the passed RNN layer/model

    Arguments:
        module (tf.keras.layers.Layer): RNN whose weights shall be dropped
        weights (list containing string): list of index of weights dropout should be applied to (index for get_weights())
        dropout (float): dropout rate
        variational (bool): irrelevant (in original code there was an alternative method of dropping weights)
    """
    def __init__(self, module, weights, dropout=0, variational=False):
        super().__init__()

        self.module = module # rnn model/layer
        self.which_weights = weights
        self.dropout = dropout # weight dropout rate
        self.variational = variational # alternative procedure of applying droput (not implemented here)

    def _setweights(self):
        """ Performs Dropout on weight matrices for recurrent connections)
        """
        for weight in self.which_weights:
            w = self.module.get_weights()
            ## here the variational alternative is missing but it does ultimatively the same thing
            w[weight] = tf.nn.dropout(x=w[weight], rate=self.dropout)
            #print(w[weight])
            self.module.set_weights(w)

    def call(self, *args):
        self._setweights() # perform weights dropout
        return self.module(*args) # call rnn layer



# testing weight drop on linear and lstm model
# (?) lstm fails so far
if __name__ == '__main__':
    import tensorflow as tf
    from weight_drop import WeightDrop

    print('Testing WeightDrop')
    print('=-=-=-=-=-=-=-=-=-=')

    # Input is (seq, batch, input)
    x = tf.constant(tf.random.uniform(shape=[2, 1, 10]))

    print('Testing WeightDrop with Linear')

    dense = tf.keras.layers.Dense(units=10)
    dense.build(input_shape=[None, 1, 10])
    wd_dense = WeightDrop(dense, [0], dropout=0.9)
    run1 = [x.sum() for x in wd_dense(x).numpy()]
    run2 = [x.sum() for x in wd_dense(x).numpy()]

    print('All items should be different')
    print('Run 1:', run1)
    print('Run 2:', run2)

    assert run1[0] != run2[0]
    assert run1[1] != run2[1]

    print('---')

    print('Testing WeightDrop with LSTM')

    lstm = tf.keras.layers.LSTM(units=10)
    lstm.build(input_shape=[[None, 1, 10]])
    wd_lstm = WeightDrop(lstm, [1], dropout=0.9)

    run1 = [x.sum() for x in wd_lstm(x)[0].numpy()]
    run2 = [x.sum() for x in wd_lstm(x)[0].numpy()]

    print('First timesteps should be equal, all others should differ')
    print('Run 1:', run1)
    print('Run 2:', run2)

    # First time step, not influenced by hidden to hidden weights, should be equal
    assert run1[0] == run2[0]
    # Second step should not
    assert run1[1] != run2[1] # fails: all outputs are the same (not just first)

    print('---')
