---
layout: post
title: Neural network forward propagation
date: 2017-12-16 08:24
comments: true
external-url:
categories: neural_network forward_propagation
---

Neural network foward propagation formulas:

* From one layer to the next layer

$$N\text {: samples}$$  
$$D\text {: Size of last but one hidden layer}$$  
$$M\text {: Size of last hidden layer}$$  
$$W\text {: weights of last hidden layer}$$  
$$b\text {: bias of last hidden layer}$$  


$$o_1 = tanh(W_{D\times M}^\intercal x_1+b_{M\times 1})$$  
$$o_2 = tanh(W_{D\times M}^\intercal x_2+b_{M\times 1})$$  
$$...$$  
$$o_n = tanh(W_{D\times M}^\intercal x_n+b_{M\times 1})$$  

$$\begin{pmatrix}
 o_1  \\
 o_2 \\
 \vdots  \\
 o_n \\   
 \end{pmatrix} = tanh(X_{N\times D}W_{D\times M}+B_{M\times 1})$$  

* From the last but one hidden to the output (sigmoid):

$$N\text {: samples}$$  
$$D\text {: Size of last but one hidden layer}$$  
$$M\text {: Size of last hidden layer}$$  
$$W\text {: weights of last hidden layer}$$  
$$b\text {: bias of last hidden layer}$$  
$$V\text {: weights of output neuron}$$  
$$c\text {: bias for output neuron}$$  

$$y_1 = \sigma(V^\intercal tanh(W_{D\times M}^\intercal x_1+b_{M\times 1})+c)$$  
$$y_2 = \sigma(V^\intercal tanh(W_{D\times M}^\intercal x_2+b_{M\times 1})+c)$$  
$$...$$  
$$y_n = \sigma(V^\intercal tanh(W_{D\times M}^\intercal x_n+b_{M\times 1})+c)$$  

$$\begin{pmatrix}
 y_1  \\
 y_2 \\
 \vdots  \\
 y_n \\   
 \end{pmatrix} = \sigma(tanh(X_{N\times D}W_{D\times M}+b_{M\times 1})V+c)$$

 $$X_{N\times D}W_{D\times M}+bm_{M\times 1} = X_{N\times D}W_{D\times M} +\begin{pmatrix}
 - & b^\intercal & - \\
 - & b^\intercal & - \\
- &  \vdots  & - \\
 - & b^\intercal & - \\   
 \end{pmatrix}_{N\times M}$$

* From the last but one hidden to the output K>2 (softmax):

$$\begin{pmatrix}
 y_1  \\
 y_2 \\
 \vdots  \\
 y_n \\   
 \end{pmatrix} = softmax(tanh(X_{N\times D}W_{D\times M}+bm_{M\times 1})V+c)$$
