"""
Reads articles from a text file located in the given path, filters the text for unwanted
phrases and characters and adds end-of-sequence markers. Also, it trains a tokenizer on the
text and converts the words into tokens with this tokenizer. Afterwards, it serializes the
tokenized text and the word-token dictionary of the tokenizer into files.
"""
import os
import re
import tensorflow as tf
import pickle

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=voc_size)
text = ""

# read articles chunk-wise from text file and filter it
with open(os.path.join(path, "articles.txt"), 'r', encoding='utf-8') as f:
    chunk_size = 1024 * 1024 # 1MB chunk size
    while True:
        chunk = f.read(chunk_size)
        if not chunk:
            break

        # filter text + add eos marker
        chunk = chunk.replace('@highlight', '').replace('\n\n', ' eos ').lower()
        chunk = re.sub(r"[^a-z ]", "", chunk)

        # train tokenizer
        print("chunk added")
        tokenizer.fit_on_texts(chunk)
        print("vocabulary updatet")

        text += chunk

# tokenize text
print("begin tokenizing text")
text = tokenizer.texts_to_sequences(text)
print("tokenizing finished")

# serialize the tokenized text and the word-token dictionary of the tokenizer
tokens_file = open(os.path.join(path, "text_tokenized"), 'wb')
pickle.dump(text, tokens_file)
tokens_file.close()

dict_file = open(os.path.join(path, "tokenizer_dict"), 'wb')
pickle.dump(tokenizer.word_index, dict_file)
dict_file.close()
