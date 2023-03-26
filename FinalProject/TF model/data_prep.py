import tensorflow as tf
import numpy as np


def data_preprocessing(data, batch_size, bptt, evaluation=False):
    """
    Slices the input data into sequences of varying lengths and creates a dataset with input and target sequences.

    Args:
        data (list): A list of data points to be preprocessed.
        batch_size (int): The batch size to use for the created dataset.
        bptt (float): The backpropagation through time (BPTT) length to use when generating the sequences.

    Returns:
        data (tf.data.Dataset): A shuffled and batched TensorFlow dataset containing input and target sequences.
    """

    # slicing the whole data into differently sized sequences
    sequences = []
    max_seq_len = 0 # largest sequence length
    i = 0 # index for iterating whole data list
    while i < len(data):
        if not evaluation:
            # generate random sequence length dependend on bptt
            bptt2 = bptt if np.random.random() < 0.95 else bptt / 2.
            seq_len = max(5, int(np.random.normal(bptt2, 5)))
        else:
            seq_len = bptt
        # check if sequence length is less or equal to remainder of data
        if seq_len >= len(data) - i:
            break

        # slice sequence (+1 is for target set)
        sequence = data[i:i+seq_len+1]
        sequences.append(sequence)

        # update largest sequence length and iterating index
        max_seq_len = seq_len if max_seq_len < seq_len else max_seq_len
        i += seq_len

    # pad sequences and make it into dataset with input and target sequence
    sequences = tf.keras.utils.pad_sequences(sequences, maxlen=91)
    data = tf.data.Dataset.from_tensor_slices((sequences))
    data = data.map(lambda x: (x[:90], x[1:]))

    print("data preprocessing successful")

    return data.shuffle(10000).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
