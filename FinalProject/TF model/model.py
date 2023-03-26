import tensorflow as tf

from lockdrop import LockedDropout
from debiasing import bias_regularization_encoder
from endecoder import EncoderDecoder

class RNNModel(tf.keras.Model):
    """
    Implements a language model using a multi-layer recurrent neural network (RNN) with long short-term memory (LSTM)
    cells. The model learns to predict the next word in a sequence given the previous words. The network architecture
    consists of an encoder, a decoder, and multiple stacked LSTMs.
    The model is inspried from the PyTorch implementation "LSTM Language Model Toolkit" by Salesforce:
    https://github.com/salesforce/awd-lstm-lm
    """
    def __init__(self, args):
        """
        Initializes the RNN model (encoder, decoder and lstm layers) with the given hyperparameters.

        Args:
            args (argparse.ArgumentParser): ArgumentParser containing hyperparameters for the model and training.
            -> used arguments of ArgumentParser:
                lr (float): learning rate for the Adam optimizer
                vocabulary_size (int): vocabulary size
                ninp (int): size of word embeddings
                nhid (int): number of hidden units per layer
                nlayers (int): number of layers
                dropout (float): dropout rate applied to the output of each LSTM layer (default: 0.5)
                dropouth (float): dropout rate applied to the output of each LSTM layer (default: 0.5)
                dropouti (float): dropout rate applied to the embedding layer (default: 0.5)
                dropoute (float): dropout rate applied to remove words from the embedding layer (default: 0.1)
                wdrop (float): dropout rate applied to the RNN hidden-to-hidden matrix (default: 0)
        """
        super().__init__()

        # training necessities
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(30., 10, 0.9) # decays lr will be decayed with every batch that is processed by the optimizer
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=lr_scheduler, weight_decay=wdecay)
        self.metrics_list = [tf.keras.metrics.Mean(name="loss")]

        # parameters
        self.voc_size = args.vocabulary_size
        self.ninp = args.ninp            # size of word embedding
        self.nhid = args.nhid            # number of hidden units (in rnn layers)
        self.nlayers = args.nlayers      # number of rnn layers
        self.dropout = args.dropout      # dropout for overall output of rnn layers
        self.dropouti = args.dropouti    # dropout for embedding output
        self.dropouth = args.dropouth    # dropout for rnn layers
        self.dropoute = args.dropoute    # dropout for embedding layer
        self.wdrop = args.wdrop         # dropout for hidden weights matrix of lstm
        self.tie_weights = args.tie_weights # whether or not to tie encoder-decoder weights
        self.lockdrop = LockedDropout() # dropout wrapper

        # encoder/decoder
        # weights initialized with uniform distribution (-0.1, 0.1)
        initrange = 0.1
        self.endecoder = EncoderDecoder(
            input_dim=self.voc_size+1,
            output_dim=self.ninp,
            dropout = self.dropoute,
            embeddings_initializer=tf.keras.initializers.RandomUniform(-initrange, initrange)
        )
        self.endecoder.build()

        # rnn layers with L1 + L2 regularizer
        # wdrop: amount of weight dropout to apply to the RNN hidden to hidden matrix (for lstm: recurrent_kernel)
        reg = tf.keras.regularizers.L1L2(l1=0.001, l2=0.002)
        self.rnns = [tf.keras.layers.LSTM(units=self.nhid if n != self.nlayers-1 else self.ninp, activity_regularizer=reg, recurrent_dropout=self.wdrop, return_state=True, return_sequences=True) for n in range(self.nlayers)]


    def call(self, input, training=False):
        """
        Forward pass of the model.

        Args:
            input (tf.Tensor): Tensor with shape [batch_size, sequence_length] that contains the input sequences.
            hidden (list): list of Tensors, each with shape [batch_size, hidden_size] that contains the hidden states of each RNN layer.
            cell (list): list of Tensors, each with shape [batch_size, hidden_size] that contains the cell states of each RNN layer.
            return_h (bool): Whether to return hidden states and cell states of all layers. Default is False.
            training (bool): Whether the model is in training mode or not. Default is False.

        Returns:
            Tuple of (decoded, hidden, cell) if return_h is False, else
            Tuple of (decoded, hidden, cell, raw_outputs, outputs), where
                decoded (tf.Tensor): Tensor with shape [batch_size, output_size] that contains the output of the decoder layer.
                hidden (list): list of Tensors, each with shape [batch_size, hidden_size] that contains the hidden states of each RNN layer.
                cell (list): list of Tensors, each with shape [batch_size, hidden_size] that contains the cell states of each RNN layer.
                raw_outputs (list): list of Tensors, each with shape [batch_size, sequence_length, hidden_size] that contains the outputs directly after each RNN layer.
                outputs (list): list of Tensors, each with shape [batch_size, sequence_lenght, hidden_size] that contains the outputs of each RNN layer after dropout has been performed.
        """
        # embedding + dropout
        emb = self.endecoder(input, training=training)
        emb = self.lockdrop(emb, self.dropouti)

        # pass embedding and the hidden state through all rnn layers
        for n, rnn in enumerate(self.rnns):
            output, new_h, _ = rnn(emb) # call lstm
            new_h = tf.stop_gradient(new_h) ## I don't think it works like that!!
            if n < self.nlayers-1: # dropout for all rnn layers but last
                output = self.lockdrop(output, self.dropouth)
        ouput = self.lockdrop(output, self.dropout) # dropout for last rnn layer

        # decoding
        decoded = self.endecoder.decode(output)

        return decoded


    def reset_metrics(self):
        """ Resets the states of all metrics of this model. """
        for metric in self.metrics:
            metric.reset_states()


    @tf.function
    def train_step(self, data, D, N, debiasing, clip=0, norm=True):
        """
        Perform one training step for a given batch of data.

        Args:
            data (tuple): A tuple of input and target data.
            hidden (list): A list of hidden states for each RNN layer.
            cell (list): A list of cell states for each RNN layer.
            D (tf.Tensor): A tensor of shape (d,) representing the gender direction vector.
            N (tf.Tensor): A tensor of shape (d, n) representing the bias subspace.
            norm (bool): Whether to normalize the gender direction vector (default=True).

        Returns:
            dict: A dictionary mapping metric names to current value.
            list: A list of updated hidden states for each RNN layer.
            list: A list of updated cell states for each RNN layer.
        """
        inputs, targets = data

        # update parameters with gradient tape
        with tf.GradientTape() as tape:

            # inputs are run through the model to get the predictions
            predictions = self(inputs, training=True)

            # loss is calculated using targets and predictions
            loss = self.loss_function(targets, predictions) + tf.reduce_sum(self.losses)

            # bias regularization is calculated using gender pairs and neutral words
            if debiasing:
                bias_loss = bias_regularization_encoder(self, D, N, var_ratio, lmbda, norm=True)
                loss += bias_loss


        # gradients are clipped and applied to the trainable parameters
        gradients = tape.gradient(loss, self.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, clip)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # update loss metric
        self.metrics[0].update_state(loss)

        # Return a dictionary mapping metric names to current value and hidden state
        return {m.name: m.result() for m in self.metrics}


    @tf.function
    def test_step(self, data):
        """
        Performs a single evaluation step for the model on the batch of input data.

        Args:
            data (tuple): A tuple of Tensors containing the input data and corresponding targets.
            hidden (list): A list of hidden states for each RNN layer.
            cell (list): A list of cell states for each RNN layer.

        Returns:
            dict: A dictionary mapping metric names to current value.
            list: A list of updated hidden states for each RNN layer.
            list: A list of updated cell states for each RNN layer.

        """
        # data is split into inputs and targets and the inputs are run through the model to get the predictions
        inputs, targets = data

        predictions = self(inputs, training=False)

        # using the targets and the predictions the loss is calculated
        loss = self.loss_function(targets, predictions) + tf.reduce_sum(self.losses)

        # loss metrics are updated
        self.metrics[0].update_state(loss)

        return {m.name: m.result() for m in self.metrics}
