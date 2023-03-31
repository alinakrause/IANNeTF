import os
import re
import pickle
import tensorflow as tf
import numpy as np

def load_tokenize_data(path, voc_size):
    """
    Load and tokenize text data from a file using a tokenizer from TensorFlow.
    And serializes the word-token dictionary of the tokenizer.

    Args:
        path (str): The path to the directory containing the text file to load.
        voc_size (int): The size of the vocabulary to use for the tokenizer.

    Returns:
        list: A list of integers representing the tokenized text.
        tokenizer (tensorflow.keras.preprocessing.text.Tokenizer): Tokenizer object that was fitted on text.

    """

    text = ""

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=voc_size)

    # read articles chunk-wise from text file and filter it
    with open(os.path.join(path, "articles.txt"), 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # filter text + add eos marker
    for line in lines:
        line = line.replace('@highlight', '').replace('\n\n', ' eos ').lower()
        line = re.sub(r"[^a-z ]", "", line)

    # train tokenizer
    tokenizer.fit_on_texts(lines)

    # serialize tokenizer
    tokenizer_file = open(os.path.join(path, "token_dictionary"), 'wb')
    pickle.dump(tokenizer.word_index, tokenizer_file)
    tokenizer_file.close()

    # tokenize text
    print("tokenize words")
    tokens_lines = tokenizer.texts_to_sequences(lines)
    tokens = []
    for line in tokens_lines:
        tokens += line
    print("tokenizing finished")

    return tokens, tokenizer


def data_preprocessing(data, batch_size, bptt, evaluation=False):
    """
    Slices the input data into sequences of varying lengthsm pads the sequences to the same length
    and creates a dataset with input and target sequences.

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
            seq_len = min(seq_len, bptt+20) # upper bound for
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
    sequences = tf.keras.utils.pad_sequences(sequences, maxlen=max_seq_len+1)
    data = tf.data.Dataset.from_tensor_slices((sequences))
    data = data.map(lambda x: (x[:max_seq_len], x[1:]))

    print("data preprocessing successful")

    return data.shuffle(10000).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
