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

$$P(Y=1|X)=\prod_{n=1}^N \prod_{k=1}^2 y_k^{t_k^{(n)}} \tag{1}$$

$$Ln(P(Y=1|X))=\sum_{n=1}^N \sum_{k=1}^K t_k^{(n)}ln(y_k^{(n)}) \tag{2}$$

from forward propagation formulas:

$$z_m = tanh(\sum_{d=1}^D {W_{dm}X_d+b)} \tag{3}$$

$$a_k = \sum_{m=1}^MV_{mk}z_m+c \tag{4}$$

$$y = softmax(a) \tag{5}$$  

$$y_k = \dfrac{e^{a_k}}{ \sum_{i=1}^Ke^{a_i}} \tag{6}$$  

back propagation gradient searching max of Ln (Likelihood):

$$V_{mk} = V_{mk} + \alpha \dfrac{\partial Ln}{\partial V_{mk}} \tag{7}$$

$$W_{dm} = W_{dm} + \alpha \dfrac{\partial Ln}{\partial W_{dm}} \tag{8}$$

$$\dfrac{\partial Ln}{\partial V_{mk}}=\sum_n \sum_{k'}\dfrac{\partial }{\partial V_{mk}} \Biggl(t^{(n)}_kln(y^{(n)}_k)\Biggl) \tag{9}$$

$$\dfrac{\partial Ln}{\partial V_{mk}}=\sum_n \sum_{k'}t^{(n)}_{k'} \dfrac{\partial Ln}{\partial y^{(n)}_k} \dfrac{\partial y^{(n)}_k}{\partial a_k}\dfrac{\partial a_k}{\partial V_{mk}} \tag{10}$$

$$\dfrac{\partial Ln}{\partial V_{mk}}=\sum_n \sum_{k'}t^{(n)}_{k'} \dfrac {1}{y^{(n)}_{k'}}\dfrac{\partial y^{(n)}_{k'}}{\partial a_k}\dfrac{\partial a_k}{\partial V_{mk}} \tag{11}$$

From derivative [softmax](https://m-alcu.github.io/blog/2017/12/15/derivative-softmax/):

$$\dfrac{\partial y^{(n)}_{k'}}{\partial a_k}= y_{k'}(\delta_{kk'}-y_k) \tag{12}$$  

$$\require{cancel}\dfrac{\partial Ln}{\partial V_{mk}}=\sum_n \sum_{k'}t^{(n)}_{k'} \dfrac {1}{\cancel {y^{(n)}_{k'}}}\cancel {y^{(n)}_{k'}}(\delta_{kk'}-y^{(n)}_k)\dfrac{\partial a_k}{\partial V_{mk}} \tag{13}$$

$$\dfrac{\partial Ln}{\partial V_{mk}}=\sum_n \sum_{k'}t^{(n)}_{k'} \delta_{kk'}\dfrac{\partial a_k}{\partial V_{mk}}-\sum_n \sum_{k'}t^{(n)}_{k'} y^{(n)}_{k'}\dfrac{\partial a_k}{\partial V_{mk}} \tag{14}$$

$$\dfrac{\partial Ln}{\partial V_{mk}}=\sum_n t^{(n)}_{k} \dfrac{\partial a_k}{\partial V_{mk}}-\sum_n  y^{(n)}_{k'}\dfrac{\partial a_k}{\partial V_{mk}} \tag{15}$$

$$\dfrac{\partial Ln}{\partial V_{mk}}=\sum_n (t^{(n)}_{k'} -y^{(n)}_k)\dfrac{\partial a_k}{\partial V_{mk}} \tag{16}$$  

From forward [propagation formulas](https://m-alcu.github.io/blog/2017/12/16/forward-propagation-deep-learning/):

$$\dfrac{\partial a_k}{\partial V_{mk}}= z^{(n)}_m \tag{17}$$

$$\bbox[5px,border:2px solid black] {\begin{align*} \dfrac{\partial Ln}{\partial V_{mk}}= \sum_n ( t^{(n)}_k-y^{(n)}_k)z^{(n)}_m \end{align*}} \tag{18}$$

$$\dfrac{\partial Ln}{\partial W_{dm}}=\sum_n \sum_{k'}\dfrac{\partial }{\partial W_{dm}} \Biggl(t^{(n)}_{k'} ln(y^{(n)}_{k'})\Biggl) \tag{19}$$

$$\dfrac{\partial Ln}{\partial W_{dm}}=\sum_n \sum_{k'} \dfrac{\partial Ln}{\partial y^{(n)}_{k'}} \dfrac{\partial y^{(n)}_{k'}}{\partial a_k} \dfrac{\partial a_k}{\partial z^{(n)}_m}\dfrac{\partial z^{(n)}_m}{\partial W_{dm}} \tag{20}$$

$$f=f(a_1(z_m), a_2(z_m), ...,a_K(z_m))\tag{21}$$

$$\frac {df}{dz_m} = \sum^K_{k=1}  \dfrac{\partial f}{\partial a_k} \dfrac{\partial a_k}{\partial z_m} \tag{22}$$

$$\dfrac{\partial Ln}{\partial W_{dm}}=\sum_n \sum_{k'}  t^{(n)}_{k'} \dfrac{1}{y^{(n)}_{k'}} \sum_{k}\dfrac{\partial y^{(n)}_{k'}}{\partial a_k} \dfrac{\partial a_k}{\partial z^{(n)}_m}\dfrac{\partial z^{(n)}_m}{\partial W_{dm}} \tag{23}$$

$$\require{cancel}\dfrac{\partial Ln}{\partial W_{dm}}=\sum_n \sum_{k'} \sum_{k}t^{(n)}_{k'} \dfrac{1}{\cancel {y^{(n)}_{k'}}} \cancel {y^{(n)}_{k'}} ( \delta_{kk'}- y^{(n)}_k)\dfrac{\partial a_k}{\partial z^{(n)}_m}\dfrac{\partial z^{(n)}_m}{\partial W_{dm}} \tag{24}$$

$$\dfrac{\partial Ln}{\partial W_{dm}}=\sum_n \sum_{k} ( t^{(n)}_k - y^{(n)}_k)\dfrac{\partial a_k}{\partial z^{(n)}_m}\dfrac{\partial z^{(n)}_m}{\partial W_{dm}} \tag{25}$$

From [derivative of tanh](https://m-alcu.github.io/blog/2017/12/20/tanh-derivative/)

$$\dfrac{\partial Ln}{\partial W_{dm}}=\sum_n \sum_{k} ( t^{(n)}_k - y^{(n)}_k) z^{(n)}_m(1-z^{(n)}_m)\dfrac{\partial z^{(n)}_m}{\partial W_{dm}}\tag{26}$$

$$z_m = tanh(\sum^D_{d=1} W_{dm}x_d) \tag{27}$$

$$\bbox[5px,border:2px solid black] {\begin{align*} \dfrac{\partial Ln}{\partial W_{dm}}=\sum_n \sum_{k} ( t^{(n)}_k - y^{(n)}_k) z^{(n)}_m(1-z^{(n)}_m)x^{(n)}_d \end{align*}} \tag{28}$$


