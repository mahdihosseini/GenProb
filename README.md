# [In Search of Probeable Generalization Measures]() #

## Table of Contents ##
- [Adas: Adaptive Scheduling of Stochastic Gradients](#adas--adaptive-scheduling-of-stochastic-gradients)
  * [Introduction](#introduction)
    + [Generalization Measures](#gmeasures)
    + [GenProb Dataset](#genprob)
    + [Results](#results)
  * [Requirements](#requirements)
    + [Software/Hardware](#software-hardware)
    + [Usage](#usage)
    + [Common Issues (running list)](#common-issues--running-list-)
  * [TODO](#todo)

## Overview ##
In Search of Probeable Generalization Measures (LINK TO PAPER) evaluates and compares generalization measures to establish firm ground for further investigation and incite the production of novel deep learning algorithms that improve generalization. This repository contains the scripts used to parse through GenProb, a dataset of trained deep CNNs, processing model layer weights and computing generalization measures. You can use this code to better understand how GenProb can be used to test generalization measures and HPO algorithms. Measure calculation scripts are also provided.

![image](https://user-images.githubusercontent.com/77180677/137248344-66d65abf-0a94-4b43-a269-b9f2b6c78e12.png)

### Generalization Measures ###
*Stable quality* (SQ) refers to the stability of encoding in a deep layer that is calculated with the relative ratio of stable rank and condition number of a layer. 

*Effective rank* (E) refers to the dimension of the output space of the transformation operated by a deep layer that is calculated with the Shannon entropy of the normalized singular values of a layer as defined in.

*Frobenius norm* (F) refers to the magnitude of a deep layer that is calculated with the sum of the squared values of a weight tensor.

*Spectral norm* (S) refers to the maximum magnitude of mapping by a transformation operated by a layer that is calculated as the maximum singular value of a weight tensor.

Further elaboration of these metrics and their equations can be found in the paper. The layer-wise processing of these metrics can be found under /source/process.py along with a list of other metrics discluded from the paper. Convolution weight tensors are first unfolded along channel axes into a 2d matrix before metrics are calculated via processing of singular values or other norm calculations. The low rank factorization preprocessing of weight matrices is also included under the EVBMF function. Metrics are aggregated accross layers 

### GenProb Dataset ###
Generalization Dataset for Probeable Measures is a family of trained models used to test the effectiveness of the measures for tracking generalization performance at earlier stages of training. We train families of models with varied hyperparameter and channel size configurations as elaborated in the paper.

The full dataset of pytorch model files can be accessed at: (LINK) --currently being uploaded

### Results ###
Generalization measures plotted against generalization performance metrics at progressive epochs of training for models optimized with Adam from the GenProb dataset.
<img src="https://user-images.githubusercontent.com/44271301/136673234-bc5e6f4b-0375-4f50-a4ba-95e0653dcbda.png">

Evolution of generalization measure correlation with generalization performance metrics over epochs of training for models optimized with Adam from the GenProb dataset.
<img src="https://user-images.githubusercontent.com/44271301/136673243-6ef7016b-5e39-4ffd-8a86-d222f3c2faed.png">

## Requirements ##
We use `Python 3.7`.

### Software ###
Please find required libraries in the `requirements.txt` file.

### Usage ###
#### Pretrained Models ####
GenProb pretrianed model weights should be placed in the `GenProb/models/GenProb`. Other pretrained model weight may be placed anywhere, and the path must be specified in `source/parsing_agent.py`.

Within `source/main.py`, the library of models must be specified, alongside the hyperparameter configuration wanted. For GenProb, that includes the number of epochs trained for, and the dataset. Evaluations may be done in batches, using the boolean `new`. If set to 0, evaluation will begin at the index specified by `start`. The name of the file the results should be appened to must be specified as well. Otherwise, it will begin at the first file in the folder, and appened results to a new file.

This outputs a csv file, with the metrics evaluation on a layer-wise basis. These may be aggregated as wanted, or by using methods specified in the paper through use of the file `source/qualities.py`.

### Common Issues (running list) ###
