"""
In this file are all the methods needed for getting the datasets for training.
- load_data(path): Loads data from file, filters text, splits into train/val/test set.
- tokenize_data(datasets, voc_size): Creates tokenizer that tokenizes the tree datasets.
- data_preprocessing(data, batch_size, bptt): Makes input/target sequences and batches for training.
"""
import os
import re
import tensorflow as tf
import numpy as np


def load_data(path):
    """
    Reads articles from a text file located in the given path, filters the text for unwanted
    phrases and characters and adds end-of-sequence markers.
    It splits the text into train, validation and test sets (0.6-0.2-0.2 ratio),
    writes the texts into corresponding txt files, and returns the texts.

    Args:
        path (str): The path to the directory where the source file is saved and where text file and the train,
                    validation and test files will be saved.

    Returns:
        list of str: A list of texts, where each text is a string containing the articles for the corresponding data set.
    """

    # read articles from text file
    read_file = open(os.path.join(path, "articles.txt"), 'r', encoding='utf-8')
    text = read_file.read()
    read_file.close()

    # filter text + add eos marker
    text = text.replace('@highlight', '').replace('\n\n', ' eos ').lower()
    text = re.sub(r"[^a-z ]", "", text)

    # split text in train, validation and test set
    l = len(text)
    texts = [text[:int(l*.6)], text[int(l*.6):int(l*.8)], text[int(l*.8):]]

    # write text in accoring file
    files = ["train.txt", "valid.txt", "test.txt"]
    for i, f in enumerate(files):
        write_file = open(os.path.join(path, f), 'w', encoding='utf-8')
        write_file.write(texts[i])
        write_file.close()

    return texts


def tokenize_data(datasets, voc_size):
    """
    Tokenizes the given datasets using the Keras Tokenizer.

    Args:
        datasets (list of str): A list of strings representing the datasets to be tokenized.
        voc_size (int): The size of the vocabulary, i.e., the maximum number of words to keep based on word frequency.

    Returns:
        tuple of
            tokenizer (keras.preprocessing.text.Tokenizer): The trained Keras Tokenizer.
            list of list of int: The tokenized datasets that are represented as a list of lists of integers,
                                where each inner list represents a sequence of tokens for a given dataset.
    """

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=voc_size)
    tokenizer.fit_on_texts(datasets)
    datasets = tokenizer.texts_to_sequences(datasets)

    return tokenizer, datasets


def data_preprocessing(data, batch_size, bptt):
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
    inputs = []
    max_seq_len = 0 # largest sequence length
    i = 0 # index for iterating whole data list
    while i < len(data):
        # generate random sequence length dependend on bptt
        bptt = bptt if np.random.random() < 0.95 else bptt / 2.
        seq_len = max(5, int(np.random.normal(bptt, 5)))

        # slice sequence (+1 is for target set)
        sequence = data[i:i+seq_len+1]
        inputs.append(sequence)

        # update largest sequence length and iterating index
        max_seq_len = seq_len if max_seq_len < seq_len else max_seq_len
        i += seq_len

    # pad sequences and make it into dataset with input and target sequence
    sequences = tf.keras.utils.pad_sequences(inputs, maxlen=max_seq_len+1)
    data = tf.data.Dataset.from_tensor_slices((sequences))
    data = data.map(lambda x: (x[:max_seq_len], x[1:]))

    return data.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
