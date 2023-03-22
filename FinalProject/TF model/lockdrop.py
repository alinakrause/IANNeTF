import tensorflow as tf
import tensorflow_probability as tfp

class LockedDropout(tf.keras.layers.Layer):
    """
    Dropout layer that is applied to its inputs in a consistent way across all time steps of RNN.
    Generates a new mask for each forward pass which is then applied across all time steps.
    """
    def __init__(self):
        """ Initializes the LockedDropout layer. """
        super().__init__()

    def call(self, x, dropout=0.5, training=False):
        """
        Applies dropout to the input tensor x.

        Args:
            x (tensor): Input tensor.
            dropout (float): Dropout rate, between 0 and 1. Default is 0.5.
            training (bool): Whether the layer is being used in training mode. Default is False.

        Returns:
            tensor: Tensor with dropout applied to it.
        """
        # unchanged if not in training or no dropout
        if not training or not dropout:
            return x
        # otherwise apply bernoulli mask for dropout
        size = x.shape
        # create mask with dropout rate of bernoulli probability of dropout argument
        # and normalizes the tensor (if dropout rate is high, mask gets multiplied with larger number)
        mask = tfp.distributions.Bernoulli(probs=(1-dropout), dtype='float32').sample(sample_shape=([1, size[1], size[2]])) / (1 - dropout)
        mask = tf.cast(tf.broadcast_to(mask, size), "float32")

        return mask * x # apply mask on input tensor

def embed_drop(encoder, dropout):
    """ 
    Applies bernoulli dropout to the weights matrix of the input encoder layer.

    Args:
        encoder (tf.keras.layers.Layer): The input encoder layer.
        dropout (float): The probability of dropping out a weight.
    """
    if dropout:
        weights = encoder.get_weights()

        mask = tfp.distributions.Bernoulli(probs=(1-dropout), dtype='float32').sample(sample_shape=(weights[0],1)) / (1 - dropout)
        mask = tf.cast(tf.broadcast_to(mask, size), "float32").numpy()

        encoder.set_weights(mask * weights)
