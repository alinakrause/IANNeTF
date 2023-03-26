import os
import tensorflow as tf

def get_gender_pairs(path, vocabulary_size, tokenizer):
    """
    Reads a list of gendered word pairs from a file, creates the defining set containing
    the tokenized gendered word pairs, and a set of neutral words containing all the tokens that are  not
    in the defining set.

    Args:
        vocabulary_size (int): An integer representing the maximum number of words to include in the vocabulary.

    Returns:
        set: A set of gendered words (both male and female).
        tf.Tensor: A tensor of gendered word pairs, where each row is a pair of indices into the vocabulary.
        tf.Tensor: A tensor of neutral words, represented as indices into the vocabulary.
    """

    # read gender pairs file
    with open(os.path.join(path,"gender-pairs.txt"), 'r') as f:
        gender_pairs = f.readlines()

    # make gendered words lists
    female_words, male_words = [], []
    for gp in gender_pairs:
        f, m = gp.split()
        female_words.append(f)
        male_words.append(m)

    gender_words = set(female_words) | set(male_words)

    # defining set
    # D: tensor with gendered word pairs (tokenized)
    # shape: (number of gender pairs, 2)
    word_index = tokenizer.word_index # tokenizer dict
    word2idx = {key: word_index[key] for key in list(word_index.keys())[:vocabulary_size]}
    D = tf.constant([[word2idx[wf], word2idx[wm]]
                     for wf, wm in zip(female_words, male_words)
                     if wf in word2idx and wm in word2idx])

    # N: tensor with neutral words (tokenized)
    # shape: (number of gender neutral words)
    N = tf.constant([word2idx[w] for w in word2idx if w not in gender_words])

    return gender_words, D, N
