# Gender bias debiasing language Model

This folder contains the code for a LSTM language model and its training in TensorFlow. During the training, a debiasing method is applied by computing a bias regularization loss term. The training data is the CNN/DailyMail corpus. The tokenizer and the model's embedding weights matrix are serialized for measurements.

The underlying RNN language model implementeation (in PyTorch) is originally by Salesforce and can be found here: https://github.com/salesforce/awd-lstm-lm

The implementation of this model including the debiasing method (also in PyTorch) was originally done by Bordia and Bowman and can be found here: https://github.com/BordiaS/language-model-bias

## Execution

1. Download the CNN/DailyMail articles from https://cs.nyu.edu/~kcho/DMQA/.
2. Run the extract_data.py file for both the CNN articles and the DailyMail articles.
3. Run the main.py file. This will preprocess the extracted data and train the model on the data. 
