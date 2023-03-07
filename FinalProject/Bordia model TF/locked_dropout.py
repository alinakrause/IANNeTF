import tensorflow as tf
import tensorflow_probability as tfp


class LockedDropout(tf.keras.Model):
    """ Way to apply dropout to inputs in a consistent way across all time steps of RNN
        can help prevent overfitting and improve generalization performance
        Idea is from a paper, originally same mask is applied across time steps of rnn,
        this implementation generates a new mask for each forward pass which is then applied across all time steps
    """
    def __init__(self):
        super().__init__()

    @tf.function
    def call(self, x, dropout=0.5, training=False):
        # unchanged if not in training or no dropout
        if not training or not dropout:
            return x
        # otherwise apply bernoulli mask for dropout
        size = x.shape
        # create mask with dropout rate of bernoulli probability of dropout argument
        # and normalizes (?) the tensor (if dropout rate is high, mask gets multiplied with larger number)
        mask = tfp.distributions.Bernoulli(probs=(1-dropout), dtype='float32').sample(sample_shape=([1, size[1], size[2]])) / (1 - dropout)
        mask = tf.cast(tf.broadcast_to(mask, size), "int32") # (?) same mask for each element in batch

        return mask * x # apply mask on input tensor
