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

## Introduction ##
In Search of Probeable Generalization Measures (LINK TO PAPER) evaluates and compares generalization measures to establish firm ground for further investigation and incite the production of novel deep learning algorithms that improve generalization. This repository contains the scripts used to parse through GenProb, a dataset of trained deep CNNs, processing model layer weights and computing generalization measures. You can use this code to better understand how GenProb can be used to test generalization measures and HPO algorithms.

### Generalization Measures ###
We define several measures that quantify the quality of a trained model (i.e. quality metrics or complexity measures) and describe its generalization ability. These quality metrics are probeable on individual layers of deep neural networks, and quantify the contribution of each layer as a holistic measure for network representation, unlike other popular and successful measures. The overview and equations of all chosen measures are presented in Table I below. 

*Stable quality* (SQ) refers to the stability of encoding in a deep layer that is calculated with the relative ratio of stable rank and condition number of a layer. Stable rank encodes the space expansion under the matrix mapping of the layer, and condition number indicates the numerical sensitivity of the mapping layer. Altogether the measure introduces a quality measure of the layer as an autoencoder. *Effective rank* (E) refers to the dimension of the output space of the transformation operated by a deep layer that is calculated with the Shannon entropy of the normalized singular values of a layer as defined in.

*Frobenius norm* (F) refers to the magnitude of a deep layer that is calculated with the sum of the squared values of a weight tensor. Frobenius norm is also calculated with the sum of the squared singular values of a layer. *Spectral norm* (S) refers to the maximum magnitude of mapping by a transformation operated by a layer that is calculated as the maximum singular value of a weight tensor.
<div align="center">
 
![image](https://user-images.githubusercontent.com/77180677/136481979-b2241e0a-b859-4a9c-a2a7-bc0e5cb4f3ad.png)

</div>

The notation convention used in Table I to represent different quality  metrics is: <img src="https://render.githubusercontent.com/render/math?math=Q_{M}^{AGG}"> where  aggregation AGG ∈ {L2 = depth-normalized L2 norm, p = depth-normalized product} and metric M ∈ {SQ = stable quality, E = effective rank, F = Frobenius norm, S = stable norm}.

The layer-wise processing of these metrics can be found under /source/process.py along with a list of other metrics discluded from the paper. Convolution weight tensors are first unfolded along channel axes into a 2d matrix before metrics are calculated via processing of singular values or other norm calculations. The low rank factorization preprocessing of weight matrices is also included under the EVBMF function.

### GenProb Dataset ###
Generalization Dataset for Probeable Measures is a family of trained models used to test the effectiveness of the measures for tracking generalization performance at earlier stages of training. We train families of models with varied hyperparameter and channel size configurations for 70 epochs on CIFAR10 and CIFAR100 with various optimizers. These variations are specified in the table below.

<img src="https://user-images.githubusercontent.com/44271301/136673205-02c7653c-1ea9-4292-a966-e16128628fa2.png">


The model architecture used is described in the below table.

<img src="https://user-images.githubusercontent.com/44271301/136673216-c6dd2c1f-564a-4f61-ab82-ff2f6025b232.png">


The convolutional blocks can be described as directed acyclic graphs with five nodes of activations. All nodes re ordered, and each node is connected to all nodes in front of it with a 3x3 convolution.

<img src="https://user-images.githubusercontent.com/44271301/136673227-d7dab206-cd0f-422d-a1d9-bbdcade9e9ef.png">

### Results ###
To visualize the relationship between the quality metrics and both generalization gap and test accuracy, we produce scatter plots of test accuracy and generalization gap over the quality metrics. Furthermore, by organizing these separate scatter plots relative to the quantity of training each set of models has undergone, we can study the evolution of the relationship.

We observe a lack of form in the effective rank scatter plots on models trained with AdaM on CIFAR10 at earlier epochs. The quality metric evolves into clear, strong trends with test accuracy and generalization gap as training progresses and learned structure develops in the model weights.

We observe a clear linear relationship between the effective rank measure and generalization gap at later epochs, and a 2<sup>nd</sup> order relationship between the effective rank measure and test accuracy. The plateauing trend with test accuracy delineates a bound on test accuracy; maximizing effective rank above this bound would still increase generalization gap (linear trend) however, suggesting an increase in train accuracy without changes in test accuracy. It is still evident that for a model trained on CIFAR10 with AdaM, a greater effective rank indicates greater test accuracy, and a greater (negative) generalization gap.

<img src="https://user-images.githubusercontent.com/44271301/136673234-bc5e6f4b-0375-4f50-a4ba-95e0653dcbda.png">

By plotting the correlations of the quality metrics with test accuracy and generalization gap in the below figures, we can understand the relative progression of the effectiveness of these measures through different stages of training. As the aforementioned trends become more distinct, the corresponding correlations increase in magnitudes, some nearly up to 1.

The large correlations indicate robustness to changes in training hyperparameters, and model channel sizes. Effective rank and stable quality measures prove to be the most effective and robust generalization measures through all training phases and across dataset complexities.

<img src="https://user-images.githubusercontent.com/44271301/136673243-6ef7016b-5e39-4ffd-8a86-d222f3c2faed.png">

## Requirements ##
We use `Python 3.7`.

### Software ###
Please find required libraries in the `requirements.txt` file.

### Usage ###

### Common Issues (running list) ###
