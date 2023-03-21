from data_prep import load_data, tokenize_data, data_preprocessing
from training import training_loop
from config import config_name
from model import RNNModel

# load datasets
path = ""
datasets = load_data(path)

# tokenize datasets
vocabulary_size = 10000
datasets, tokenizer = tokenize_data(datasets, vocabulary_size)

# preprocess datasets
bptt = 70
batch_sizes = [80, 10, 1]
train_ds = data_preprocessing(datasets[0], batch_sizes[0], bptt)
val_ds = data_preprocessing(datasets[1], batch_sizes[1], bptt, evaluation=True)
test_ds = data_preprocessing(datasets[2], batch_sizes[2], bptt, evaluation=True)

# instantiate model
model = RNNModel(lr=30, ntoken=vocabulary_size, ninp= 400, nhid=1150, nlayers = 3, wdrop=0.5)

# train the model
train_writer, val_writer = config_name()
training_loop(model=model,
              train_ds=train_ds,
              val_ds=train_ds,
              epochs=500,
              vocabulary_size=vocabulary_size,
              var_ratio=0.5,
              lmbda=0.8,
              debiasing=True,
              alpha=2,
              beta=1,
              train_summary_writer=train_writer,
              val_summary_writer=val_writer)
