import tensorflow as tf
import tensorflow_probability as tfp

class EncoderDecoder(tf.keras.layers.Layer):
    # custom encoder and decoder layer
    # does what an embedding layer does + applies dropout to weights matrix
    # does what dense layer does by using transform of embedding weights (tie weights)
    #       right now without bias but with softmax activation

    def __init__(self, input_dim, output_dim, dropout, embeddings_initializer):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embeddings_initializer = embeddings_initializer
        self.dropout = dropout


    def build(self):
        # initialize weights matrix
        self.w = tf.Variable(
            initial_value=self.embeddings_initializer(shape=(self.input_dim, self.output_dim),
                                dtype='float32'),
            trainable=True)
        """
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(self.input_dim,), dtype='float32'),
            trainable=True)
        """

    # functions like embedding layer with internal dropout
    def __call__(self, inputs, training=False):
        # create dropout mask
        if training:
            mask = tfp.distributions.Bernoulli(probs=(1-self.dropout), dtype='float32').sample(sample_shape=(self.input_dim, 1)) / (1 - self.dropout)
            mask = tf.cast(tf.broadcast_to(mask, (self.input_dim, self.output_dim)), "float32")
        else: # useless mask if not training
            mask = tf.ones([self.input_dim, self.output_dim])
        # apply mask
        weights = self.w * mask

        return tf.gather(weights, inputs) # create embedding output

    # functions lake dense layer but without bias
    def decode(self, inputs):
        output = tf.matmul(inputs, tf.transpose(self.w)) #+ self.b
        return tf.nn.softmax(output)
