import tensorflow as tf
import tensorflow_probability as tfp

class LockedDropout(tf.keras.layers.Layer):
    """
    Dropout layer that is applied to its inputs in a consistent way across all time steps of RNN.
    Generates a new mask for each forward pass which is then applied across all time steps.

    Methods:
        call: Applies dropout to the input tensor.
    """
    def __init__(self):
        """ Initializes the LockedDropout layer. """
        super().__init__()

    def __call__(self, x, dropout=0.5, training=False):
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
        # otherwise create and apply bernoulli mask for dropout
        size = x.shape
        mask = tfp.distributions.Bernoulli(probs=(1-dropout), dtype='float32').sample(sample_shape=([1, size[1], size[2]])) / (1 - dropout)
        mask = tf.cast(tf.broadcast_to(mask, size), "float32")

        return mask * x