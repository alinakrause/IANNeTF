from data_prep import load_data, tokenize_data, data_preprocessing

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
