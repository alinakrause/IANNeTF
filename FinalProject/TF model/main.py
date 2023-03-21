import argparse

from data_prep import load_data, tokenize_data, data_preprocessing
from training import training_loop
from config import config_name
from model import RNNModel

# set the hyperparameters
parser = argparse.ArgumentParser(description="Hyperparameters for training RNN")
# data
parser.add_argument('--path', type=str, default="", help='path to folder where training data is stored')
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
parser.add_argument('--nlayers', type=inp, default=0.5, help='amount of rnn layers in the model')
parser.add_argument('--lr', type=float, default=30, help='initial learning rate')
parser.add_argument('--ninp', type=int, default=400, help='size of word embedding')
parser.add_argument('--nhid', type=int, default=1150, help='number of hidden units per layer')
parser.add_argument('--ninp', type=int, default=400, help='size of word embedding')
parser.add_argument('--dropout', type=float, default=0.4, help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.3, help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.65, help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1, help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.5, help='amount of weight dropout to apply to the RNN hidden to hidden matrix')

args = parser.parse_args()

# load datasets
path = ""
datasets = load_data(path)

# tokenize datasets
datasets, tokenizer = tokenize_data(datasets, args.vocabulary_size)

# preprocess datasets
batch_sizes = [80, 10, 1]
train_ds = data_preprocessing(datasets[0], args.train_bsz, args.ptt)
val_ds = data_preprocessing(datasets[1], args.val_bsz, args.bptt, evaluation=True)
test_ds = data_preprocessing(datasets[2], args.test_bsz, args.bptt, evaluation=True)

# instantiate model
model = RNNModel(args)

# train the model
train_writer, val_writer = config_name()
training_loop(model=model,
              train_ds=train_ds,
              val_ds=train_ds,
              arguments=args,
              train_summary_writer=train_writer,
              val_summary_writer=val_writer)
