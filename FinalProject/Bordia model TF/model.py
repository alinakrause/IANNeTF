import tensorflow as tf

from locked_dropout import LockedDropout

class RNNModel(tf.keras.Model):
    """ Layers: Encoder -> nlayers RNN layers (LSTM) -> Decoder
        applies varies forms of dropouts

        Difference to original:
        - only LSTM (no QRNN or GRU option) implemented
        - tying weights not (yet?) implemented
        - dropouts often simplified (but should do approximately the same?)
    """
    def __init__(self, ntoken, ninp, nhid, nlayers, ers, dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0):
        """ arguments:
        ntoken (int): vocabulary size
        ninp (int): size of word embeddings -> called with 400
        nhid (int): number of hidden units per layer -> called with 1150
        nlayers (int): number of layers -> called with 3
        dropout (float): dropout applied to layers -> called with 0.4 (default: 0.5)
        dropouth (float): dropout for rnn layers -> called with 0.3 (default: 0.5)
        dropouti (float): dropout for input embedding layers -> called with 0.65 (default: 0.5)
        dropoute (float): dropout to remove words from embedding layer -> called with 0.1 (default)
        wdrop (float): amount of weight dropout to apply to the RNN hidden to hidden matrix -> called with 0.5 (default: 0)
        """
        super().__init__()

        # parameters
        self.voc_size = ntoken      # vocabulary size
        self.ninp = ninp            # size of word embedding
        self.nhid = nhid            # number of hidden units (in rnn layers)
        self.nlayers = nlayers      # number of rnn layers
        self.dropout = dropout      # dropout for overall output of rnn layers
        self.dropouti = dropouti    # dropout for embedding output
        self.dropouth = dropouth    # dropout for rnn layers
        self.dropoute = dropoute    # dropout for embedding layer

        self.lockdrop = LockedDropout() # dropout wrapper

        # encoder + decoder
        # weights initialized with uniform distribution (-0.1, 0.1)
        initrange = 0.1
        self.encoder = tf.keras.layers.Embedding(
            input_dim=ntoken,
            output_dim=ninp,
            embeddings_initializer=tf.keras.initializers.RandomUniform(-initrange, initrange)
        )
        self.decoder = tf.keras.layers.Dense(
            units=ntoken,
            kernel_initializer=tf.keras.initializers.RandomUniform(-initrange, initrange)
        )
        ## tying weights


        # rnn layers
        # wdrop: amount of weight dropout to apply to the RNN hidden to hidden matrix (for lstm: recurrent_kernel)
        # (?) does the recurrent_dropout argument apply the dropout to the recurrent_kernel matrices?
        ## tying weights
        self.rnns = [tf.keras.layers.LSTM(units=nhid, recurrent_dropout=wdrop) for l in range(nlayers)]


    def call(self, input, hidden, return_h=False, training=False):

        # embedding + dropout
        emb = self.encoder(input)
        emb = self.lockdrop(emb, self.dropoute) # original: embed_regulariz
        emb = self.lockdrop(emb, self.dropouti)

        output = emb
        new_hidden = [] # stores hidden states of each rnn layer
        raw_outputs = [] # stores outputs directly after each rnn
        outputs = [] # stores dropout outputs of rnn layers
        # pass embedding and the hidden state through all rnn layers
        for l, rnn in enumerate(self.rnns):
            current_input = output
            output, new_h, _ = rnn(output, initial_state=[hidden[l]]) # call lstm
            new_hidden.appens(new_h)
            raw_outputs.append(output)
            # apply dropouth to output unless last layer
            if l != self.nlayers - 1:
                output = self.lockdrop(output, self.dropouth)
                outputs.append(output)
        # apply dropout to last layer output
        ouput = self.lockdrop(output, self.dropout)
        outputs.append(output)

        hidden = new_hidden # update hidden state
        result = tf.reshape(output, [output.shape[0]*output.shape[1], output.shape[2]])

        if return_h: # return outputs and hidden state without decoding
            return result, hidden, raw_outputs, outputs

        # decoding
        decoded = self.decoder(result)

        return tf.reshape(decoded, [output.shep[0], output.shape[1], decoded.shape[1]]), hidden
