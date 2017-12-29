---
layout: post
title: Neural network forward propagation
date: 2017-12-16 08:24
comments: true
external-url:
categories: neural_network forward_propagation
---

> Demonstration of forward propagation on neural networks

$x$ input for first hidden layer    
$D$ number of features from x  
$z$ input for hidden layer  
$M$ number of hidden layer of network  
$K$ Number of output classification classes   
$a$ input of last layer  
$y$ output of last layer  
$t$ trained classification output [0,1]  
$W_{dm}$ Matrix of weights from input to hidden layer $z$  
$b$ Bias of input hidden layer $z$  
$V_{mk}$ Matrix of weights from hidden layer to output $y$  
$c$ Bias of input hidden layer $z$  
$f(x)$ is the function of the middle neuron [$sigmoid(x)$, $tanh(x)$, $reLU(x)$]  
$g(x)$ is the function of the last neuron [$sigmoid(x)$, $softmax(x)$, $linear(x)$]  

![basic network example](/assets/basic-network.png)

$$P(Y=T|X)=\prod_{n=1}^N \prod_{k=1}^K y_k^{t_k^{(n)}} \tag{1}$$

$$Ln(P(Y=T|X))=\sum_{n=1}^N \sum_{k=1}^K t_k^{(n)}ln(y_k^{(n)}) \tag{2}$$

Example for $f(x)$ in tanh:  

$$z_m = tanh(\sum_{d=1}^D {W_{dm}X_d+b)} \tag{3}$$

Last layer example for $g(x)$ in softmax:

$$a_k = \sum_{m=1}^MV_{mk}z_m+c \tag{4}$$

$$y = softmax(a) \tag{5}$$

Softmax is a vector function, every output depends on all inputs ($a_k$):

$$y_k = \dfrac{e^{a_k}}{ \sum_{i=1}^Ke^{a_i}} \tag{6}$$    

> Neural network foward propagation formulas in matrix form:

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

 $$X_{N\times D}W_{D\times M}+b_{M\times 1} = X_{N\times D}W_{D\times M} +\begin{pmatrix}
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
 \end{pmatrix} = softmax(tanh(X_{N\times D}W_{D\times M}+b_{M\times 1})V+c)$$

