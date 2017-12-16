---
layout: post
title: Neural network theory
date: 2017-12-16 08:24
comments: true
external-url:
categories: neural_networks
---

Neural network formulas:

* From one layer to the next layer

$$N\text {: samples}$$  
$$D\text {: Size of last but one hidden layer}$$  
$$M\text {: Size of last hidden layer}$$
$$W\text {: weights of last hidden layer}$$  
$$Bm\text {: bias of last hidden layer}$$  


$$o_1 = tanh(W_{D\times M}^\intercal x_1+Bm)$$  
$$o_2 = tanh(W_{D\times M}^\intercal x_2+Bm)$$  
$$...$$  
$$o_n = tanh(W_{D\times M}^\intercal x_n+Bm)$$  


$$\begin{pmatrix}
 o_1  \\
 o_2 \\
 \vdots  \\
 o_n \\   
 \end{pmatrix} = tanh(X_{N\times D}W_{D\times M}+Bm_{1\times M})$$


* From the last hiden to the output (sigmoid):

$$N\text {: samples}$$  
$$D\text {: Size of last but one hidden layer}$$  
$$M\text {: Size of last hidden layer}$$
$$W\text {: weights of last hidden layer}$$  
$$Bm\text {: bias of last hidden layer}$$  
$$V\text {: weights of output neuron}$$
$$c\text {: bias for output neuron}$$  

$$y_1 = \sigma(V^\intercal tanh(W_{D\times M}^\intercal x_1+Bm_{1\times M})+c)$$  
$$y_2 = \sigma(V^\intercal tanh(W_{D\times M}^\intercal x_2+Bm_{1\times M})+c)$$  
$$...$$  
$$y_n = \sigma(V^\intercal tanh(W_{D\times M}^\intercal x_n+Bm_{1\times M})+c)$$  

$$\begin{pmatrix}
 y_1  \\
 y_2 \\
 \vdots  \\
 y_n \\   
 \end{pmatrix} = \sigma(tanh(X_{N\times D}W_{D\times M}+Bm_{1\times M})V+c)$$

* From the last hiden to the output K>2 (softmax):

$$\begin{pmatrix}
 y_1  \\
 y_2 \\
 \vdots  \\
 y_n \\   
 \end{pmatrix} = softmax(tanh(X_{N\times D}W_{D\times M}+Bm_{1\times M})V+c)$$

