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

### Generalization Measures ###
We define several measures that quantify the quality of a trained model (i.e. quality metrics or complexity measures) and describe its generalization ability. These quality metrics are probeable on individual layers of deep neural networks, and quantify the contribution of each layer as a holistic measure for network representation, unlike other popular and successful measures. The overview and equations of all chosen measures are presented in Table I below. 

*Stable quality* (SQ) refers to the stability of encoding in a deep layer that is calculated with the relative ratio of stable rank and condition number of a layer. Stable rank encodes the space expansion under the matrix mapping of the layer, and condition number indicates the numerical sensitivity of the mapping layer. Altogether the measure introduces a quality measure of the layer as an autoencoder. *Effective rank* (E) refers to the dimension of the output space of the transformation operated by a deep layer that is calculated with the Shannon entropy of the normalized singular values of a layer as defined in.

*Frobenius norm* (F) refers to the magnitude of a deep layer that is calculated with the sum of the squared values of a weight tensor. Frobenius norm is also calculated with the sum of the squared singular values of a layer. *Spectral norm* (S) refers to the maximum magnitude of mapping by a transformation operated by a layer that is calculated as the maximum singular value of a weight tensor.
<div align="center">
 
![image](https://user-images.githubusercontent.com/77180677/136481979-b2241e0a-b859-4a9c-a2a7-bc0e5cb4f3ad.png)

</div>

### GenProb Dataset ###


### Results ###

## Requirements ##


### Software ###

### Usage ###

### Common Issues (running list) ###
