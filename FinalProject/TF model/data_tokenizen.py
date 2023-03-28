import os
import re
import tensorflow as tf
import pickle

path = ""
# tokenize text in chunks of 1000 sentences
file = open(os.path.join(path, "tokenizer"), "rb")
tokenizer = pickle.load(file)

file = open('articles.txt')
lines = file.readlines()
file.close()

print("tokenize words")
tokens = tokenizer.texts_to_sequences(lines)
tokens = []
for line in tokens_lines:
    tokens += line
print("tokenizing finished")

# serialize the tokenized text and the word-token dictionary of the tokenizer
tokens_file = open(os.path.join(path, "text_tokenized"), 'wb')
pickle.dump(tokens, tokens_file)
tokens_file.close()
