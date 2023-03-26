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
    lines = file.readlines()

# filter text + add eos marker
for line in lines:
    line = line.replace('@highlight', '').replace('\n\n', ' <eos ').lower()
    line = re.sub(r"[^a-z ]", "", line)

# train tokenizer
tokenizer.fit_on_texts(lines)

# serialize tokenizer
tokenizer_file = open(os.path.join(path, "tokenizer"), 'wb')
pickle.dump(tokenizer, tokenizer_file)
tokenizer_file.close()
print("tokenizer serialized")

# tokenize text
print("begin tokenizing text")
token_lines = tokenizer.texts_to_sequences(lines)
tokens=[]
for line in token_lines:
    tokens += line
print("tokenizing finished")

# serialize the tokenized text and the word-token dictionary of the tokenizer
tokens_file = open(os.path.join(path, "text_tokenized"), 'wb')
pickle.dump(tokens, tokens_file)
tokens_file.close()
