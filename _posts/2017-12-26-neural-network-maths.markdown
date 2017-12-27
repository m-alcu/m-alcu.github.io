---
layout: post
title: Neural network back propagation
date: 2017-12-27 00:12
comments: true
external-url:
categories: neural_network back_propagation
---

> Demonstration of forward and back propagation on neural networks

$D$ number of features from x  
$M$ number of hidden layer of network  
$K$ Number of output classification classes   
$X$ input for first hidden layer  
$Z$ input for last layer  
$Y$ output of last layer  
$T$ trained classification output [0,1]  
$W_{dm}$ Matrix of weights from input to hidden layer $z$  
$V_{mk}$ Matrix of weights from hidden layer to output $y$  

![basic network example](/assets/basic-network.png)

$$P(Y=1|X)=\prod_{n=1}^N \prod_{k=1}^2 y_k^{t_k^{(n)}}$$

$$Ln(P(Y=1|X))=\sum_{n=1}^N \sum_{k=1}^K t_k^{(n)}ln(y_k^{(n)})$$

from forward propagation formulas:

$$z_m = tanh(\sum_{d=1}^D {W_{dm}X_d+b)}$$

$$a_k = \sum_{m=1}^MV_{mk}z_m+c$$

$$y = softmax(a)$$  

$$y_k = \dfrac{e^{a_k}}{ \sum_{i=1}^Ke^{a_i}}$$  

back propagation gradient searching max of Ln (Likelihood):

$$V_{mk} = V_{mk} + \alpha \dfrac{\partial Ln}{\partial V_{mk}}$$

$$W_{dm} = W_{dm} + \alpha \dfrac{\partial Ln}{\partial W_{dm}}$$

$$\dfrac{\partial Ln}{\partial V_{mk}}=\sum_n \sum_{k'}\dfrac{\partial }{\partial V_{mk}} \Biggl(t^{(n)}_kln(y^{(n)}_k)\Biggl)$$

$$\dfrac{\partial Ln}{\partial V_{mk}}=\sum_n \sum_{k'}t^{(n)}_{k'} \dfrac{\partial Ln}{\partial y^{(n)}_k} \dfrac{\partial y^{(n)}_k}{\partial a_k}\dfrac{\partial a_k}{\partial V_{mk}}$$

$$\dfrac{\partial Ln}{\partial V_{mk}}=\sum_n \sum_{k'}t^{(n)}_{k'} \dfrac {1}{y^{(n)}_{k'}}\dfrac{\partial y^{(n)}_k}{\partial a_k}\dfrac{\partial a_k}{\partial V_{mk}}$$

From derivative [softmax](https://m-alcu.github.io/blog/2017/12/15/derivative-softmax/):

$$\dfrac{\partial y^{(n)}_{k'}}{\partial a_k}= y_{k'}(\delta_{kk'}-y_k)$$

From forward propagation formulas:

$$\dfrac{\partial a_k}{\partial V_{mk}}= z^{(n)}_m$$

$$\dfrac{\partial Ln}{\partial V_{mk}}= \sum_n ( t^{(n)}_k-y^{(n)}_k)z^{(n)}_m$$

$$\dfrac{\partial Ln}{\partial W_{dm}}=\sum_n \sum_{k}\dfrac{\partial }{\partial W_{dm}} \Biggl(t^{(n)}_kln(y^{(n)}_k)\Biggl)$$

$$\dfrac{\partial Ln}{\partial V_{mk}}=\sum_n \sum_k \dfrac{\partial Ln}{\partial z^{(n)}_m}\dfrac{\partial z^{(n)}_m}{\partial W_{dm}}$$

