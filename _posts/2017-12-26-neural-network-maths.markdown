---
layout: post
title: Neural network forward and back propagation
date: 2017-12-27 00:12
comments: true
external-url:
categories: sgd forward_propagation back_propagation
---

> Demonstration of forward and back propagation on neural networks

$D$ number of features from x
$M$ number of hidden layer of network  
$K$ Number of output classification classes

$W_{dm}$ Matrix of weights from input to hidden layer $z$
$V_{mk}$ Matrix of weights from hidden layer to output $y$  

![basic network example](/assets/basic-network.png)

$$P(Y=1|X)=\prod_{i=1}^N \prod_{k=1}^2 h_k(x^{(i)})^{y_k^{(i)}}$$


Glosary  
* *MLE*: Maximum Likelihood Estimation  
* *sigmoid*: function to map $[-\infty,\infty]$ to $[0,1]$  
* *Epoch*: Epoch means one iteration of Stochastic Gradient Descent