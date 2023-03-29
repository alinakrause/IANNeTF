# Implementing and Analyzing a method for reducing gender bias in an LSTM language model.

This repository contains the code for a university course project, implementing the gender debiasing language model from Bordia and Bowman (2019) in TensorFlow (originally it is implemented in PyTorch) and analyzing its performance with respect to capturing a indirect gender bias as Gonen and Goldberg (2019) did it with other gender debiasing methods.

Here are the references of the underlying resources:
- The original RNN model: 
  - papers: [Regularizing and Optimizing LSTM Language Models by Merity et al (2017)](https://arxiv.org/abs/1708.02182), and [An Analysis of Neural Language Modeling at Multiple Scales by Merity et al. (2018)](https://arxiv.org/abs/1803.08240)
  - code: [LSTM and QRNN Language Model Toolkit](https://github.com/salesforce/awd-lstm-lm)
- The debiasing method included into the model: 
  - paper: [Identifying and Reducing Gender Bias in Word-Level Language Models by Bordia et al. (2019)](https://arxiv.org/abs/1904.03035)
  - code: [Debiasing Language Model](https://github.com/BordiaS/language-model-bias)
- The measurements of the indirect bias: 
  - paper: [Lipstick on a Pig: Debiasing Methods Cover up Systematic Gender Biases in Word Embeddings But do not Remove Them by Gonen et al. (2019)](https://arxiv.org/abs/1903.03862)
  - code: [Experiments for Capturing Indirect Gender Biases in Word Embeddings](https://github.com/gonenhila/gender_bias_lipstick)
