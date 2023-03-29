import tensorflow as tf
import tensorflow_probability as tfp

class EncoderDecoder(tf.keras.layers.Layer):
    """
    A custom layer that serves as an encoder-decoder for embedding and decoding inputs.

    Arguments:
        input_dim (int): The size of the input vocabulary.
        output_dim (int): The size of the embedding vectors.
        dropout (float): The dropout rate to apply during training.
        embeddings_initializer: The initializer for the embedding weights.

    Methods:
        build: Initializes the embedding weights matrix with the given initializer.
        call: Applies the embedding layer with internal dropout to the inputs.
        decode: Performs a matrix multiplication of the inputs with the transpose of the weight matrix,
            followed by a softmax activation.

    """
    def __init__(self, input_dim, output_dim, dropout, embeddings_initializer):
        """ Initializes the Layer with the given parameters.

        Args:
            input_dim (int): The size of the input vocabulary.
            output_dim (int): The size of the embedding vectors.
            dropout (float): The dropout rate to apply during training.
            embeddings_initializer: The initializer for the embedding weights.
        """

        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embeddings_initializer = embeddings_initializer
        self.dropout = dropout


    def build(self):
        """Initializes the embedding weights matrix with the given initializer."""
        self.w = tf.Variable(
            initial_value=self.embeddings_initializer(shape=(self.input_dim, self.output_dim), dtype='float32'),
            trainable=True
            )

    def __call__(self, inputs, training=False):
        """
        Applies the embedding layer with internal dropout to the inputs.

        Args:
            inputs (tensor): The inputs to the layer, in integer format.
            training (bool): Whether to apply dropout or not. Defaults to False.

        Returns:
            A tensor of shape (batch_size, input_dim, output_dim) that represents the embedded inputs after applying dropout.

        """
        # create dropout mask
        if training:
            mask = tfp.distributions.Bernoulli(probs=(1-self.dropout), dtype='float32').sample(sample_shape=(self.input_dim, 1)) / (1 - self.dropout)
            mask = tf.cast(tf.broadcast_to(mask, (self.input_dim, self.output_dim)), "float32")
        else:
            mask = tf.ones([self.input_dim, self.output_dim])
        # apply mask
        weights = self.w * mask

        return tf.gather(weights, inputs)


    def decode(self, inputs):
        """
        Performs a matrix multiplication of the inputs with the transpose of the weight matrix,
        followed by a softmax activation.

        Args:
            inputs (tensor): The inputs to the layer.

        Returns:
            A tensor of shape (batch_size, num_classes) that represents the output of the layer.
        """
        output = tf.matmul(inputs, tf.transpose(self.w))

        return tf.nn.softmax(output)
