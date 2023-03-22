import argparse
import os
import re
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import datetime
import pprint
import tqdm
import pickle

from data_prep import load_data, tokenize_data, data_preprocessing
from training import training_loop, testing
from config import config_name
from model import RNNModel

%load_ext tensorboard

# set the hyperparameters
parser = argparse.ArgumentParser(description="Hyperparameters for training RNN")
# data
parser.add_argument('--path', type=str, default="", help='path to folder where training data and gender pairs are stored')
parser.add_argument('--vocabulary_size', type=int, default=50000, help='size of the vocabulary of the corpus')
parser.add_argument('--bptt', type=int, default=70, help='sequence length')
parser.add_argument('--train_bsz', type=int, default=80, help='batch size for training dataset')
parser.add_argument('--val_bsz', type=int, default=10, help='batch size for validation dataset')
parser.add_argument('--test_bsz', type=int, default=1, help='batch size for test dataset')
# training loop
parser.add_argument('--epochs', type=int, default=500, help='number of epochs for training')
parser.add_argument('--debiasing', type=bool, default=True, help='whether to apply bias regularization')
parser.add_argument('--var_ratio', type=float, default=0.5, help='ratio of variance used for determining size of gender subspace for bias regularization')
parser.add_argument('--lmbda', type=float, default=1.0, help='bias regularization loss weight factor')
parser.add_argument('--alpha', type=float, default=2.0, help='parameter for L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1.0, help='parameter for slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience (if -1, early stopping is not used)')
# model
parser.add_argument('--nlayers', type=int, default=0.5, help='amount of rnn layers in the model')
parser.add_argument('--lr', type=float, default=30, help='initial learning rate')
parser.add_argument('--wdecay', type=float, default=1.2e-6, help='weight decay applied to all weights')
parser.add_argument('--ninp', type=int, default=400, help='size of word embedding')
parser.add_argument('--nhid', type=int, default=1150, help='number of hidden units per layer')
parser.add_argument('--dropout', type=float, default=0.4, help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.3, help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.65, help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1, help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.5, help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--tie_weights', type=bool, default=True, help='whether to tie encoder and decoder weights')
parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')

args = parser.parse_args()

# load datasets
datasets = load_data(args.path)

# tokenize datasets
tokenizer, datasets = tokenize_data(datasets, args.vocabulary_size)

# preprocess datasets
train_ds = data_preprocessing(datasets[0], args.train_bsz, args.bptt)
val_ds = data_preprocessing(datasets[1], args.val_bsz, args.bptt, evaluation=True)
test_ds = data_preprocessing(datasets[2], args.test_bsz, args.bptt, evaluation=True)

# instantiate model
model = RNNModel(args)

# train the model
%tensorboard --logdir logs/
train_writer, val_writer = config_name()
training_loop(model=model,
              train_ds=train_ds,
              val_ds=train_ds,
              args=args,
              tokenizer=tokenizer,
              train_summary_writer=train_writer,
              val_summary_writer=val_writer
              )

# get word embeddings matrix
word_embeddings = model.encoder.get_weights()
# get word-token dictionary
token_dict = tokenizer.word_index

# serialize
emb_file = open(os.path.join(path, "word_embedding_{bias}-".format(bias = "debiased" if args.debiasing else "biased")), 'wb')
pickle.dump(word_embeddings, emb_file)
emb_file.close()
dict_file = open(os.path.join(path, "token_dictionary"), 'wb')
pickle.dump(token_dict, dict_file)
dict_file.close()

# testing
testing(model, test_ds, args)
