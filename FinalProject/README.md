# Implementing and Analyzing a method for reducing gender bias in an LSTM language model.

This repository contains the code for a university course project, implementing the gender debiasing language model from Bordia and Bowman (2019) in TensorFlow (originally it is implemented in PyTorch) and analyzing its performance with respect to capturing the indirect gender bias as Gonen and Goldberg (2019) did it with other gender debiasing methods. 

Authors: Imogen Huesing (ihuesing@uos.de), Paula Heigl (pheigl@uos.de)

[Link to the video of the project](https://youtu.be/0vZJXA0rGx4)

## Model including debiasing

The "debiasing model" folder contains the code for a LSTM language model and its training in TensorFlow. During the training, a debiasing method is applied by computing a bias regularization loss term. The training data is intended to be CNN/DailyMail corpus but could be replaced by some other corpus. The tokenizer and the model's embedding weights matrix are serialized for measurements.

As mentioned in References, the underlying RNN language model implementeation is originally by Salesforce and in done with PyTorch. The implementation of this model including the debiasing method (also in PyTorch) was done by Bordia and Bowman. 

### Execution

1. Download the CNN/DailyMail articles from [here](https://cs.nyu.edu/~kcho/DMQA/).
2. Run the extract_data.py file for both the CNN articles and the DailyMail articles.
3. Optional: make any adjustments to the training and model parameter in the ArgumentParser at the top of the main file, e.g. whether or not to include the bias regularization.
4. Run the main.py file. This will preprocess the extracted data and train the model on the data and serialize the embeddings.

## Experiments for indirect bias

The "experiments" folder contains the Jupyter Notebook code by Gonen and Goldberg for measuring the indirect bias of the word embeddings with and without debiasing. It is only slightly adjusted to the embeddings that are being serialized during the training of our debiasing model. 


## References

Here are the references of the underlying resources:
- The original RNN model: 
  - papers: [Regularizing and Optimizing LSTM Language Models by Merity et al (2017)](https://arxiv.org/abs/1708.02182), and [An Analysis of Neural Language Modeling at Multiple Scales by Merity et al. (2018)](https://arxiv.org/abs/1803.08240)
  - code: [LSTM and QRNN Language Model Toolkit](https://github.com/salesforce/awd-lstm-lm) with BSD 3-Clause License
- The debiasing method included into the model: 
  - paper: [Identifying and Reducing Gender Bias in Word-Level Language Models by Bordia et al. (2019)](https://arxiv.org/abs/1904.03035)
  - code: [Debiasing Language Model](https://github.com/BordiaS/language-model-bias)
- The measurements of the indirect bias: 
  - paper: [Lipstick on a Pig: Debiasing Methods Cover up Systematic Gender Biases in Word Embeddings But do not Remove Them by Gonen et al. (2019)](https://arxiv.org/abs/1903.03862)
  - code: [Experiments for Capturing Indirect Gender Biases in Word Embeddings](https://github.com/gonenhila/gender_bias_lipstick) with Apache License 2.0


