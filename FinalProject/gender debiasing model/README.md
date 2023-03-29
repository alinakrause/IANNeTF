#Gender bias debiasing language Model

This folder contains the code for a LSTM language model and its training in TensorFlow. During the training, a debiasing method is applied by computing a bias regularization loss term.

The underlying RNN language model implementeation (in PyTorch) is originally by Salesforce and can be found here: https://github.com/salesforce/awd-lstm-lm
The implementation of this model including the debiasing method (also in PyTorch) was originally done by Bordia and Bowman and can be found here: https://github.com/BordiaS/language-model-bias
