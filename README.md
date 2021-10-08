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


### Results ###

## Requirements ##


### Software ###

### Usage ###

### Common Issues (running list) ###
