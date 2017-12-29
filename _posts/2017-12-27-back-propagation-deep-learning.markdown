---
layout: post
title: Neural network back propagation
date: 2017-12-27 00:12
comments: true
external-url:
categories: neural_network back_propagation
---

> Demonstration of back propagation on neural networks

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

From [forward propagation formulas](https://m-alcu.github.io/blog/2017/12/16/forward-propagation-deep-learning/):

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

From [forward propagation formulas](https://m-alcu.github.io/blog/2017/12/16/forward-propagation-deep-learning/):

$$\dfrac{\partial a_k}{\partial V_{mk}}= z^{(n)}_m \tag{17}$$

$$\bbox[5px,border:2px solid black] {\begin{align*} \dfrac{\partial Ln}{\partial V_{mk}}= \sum_n ( t^{(n)}_k-y^{(n)}_k)z^{(n)}_m \end{align*}} \tag{18}$$

$$\dfrac{\partial Ln}{\partial W_{dm}}=\sum_n \sum_{k'}\dfrac{\partial }{\partial W_{dm}} \Biggl(t^{(n)}_{k'} ln(y^{(n)}_{k'})\Biggl) \tag{19}$$

From [partial derivatives rule](https://m-alcu.github.io/blog/2017/12/20/derivatives-algebra/)  

$$f=f(a_1(x), a_2(x), ...,a_K(x))\tag{20}$$

$$\frac {df}{dx} = \sum^K_{k=1}  \dfrac{\partial f}{\partial a_k} \dfrac{\partial a_k}{\partial x} \tag{21}$$

$$\dfrac{\partial Ln}{\partial W_{dm}}=\sum_n \sum_{k'} \dfrac{\partial Ln}{\partial y^{(n)}_{k'}} \sum_k\dfrac{\partial y^{(n)}_{k'}}{\partial a_k} \dfrac{\partial a_k}{\partial z^{(n)}_m}\dfrac{\partial z^{(n)}_m}{\partial W_{dm}} \tag{22}$$  

$$\dfrac{\partial Ln}{\partial W_{dm}}=\sum_n \sum_{k'}  t^{(n)}_{k'} \dfrac{1}{y^{(n)}_{k'}} \sum_{k}\dfrac{\partial y^{(n)}_{k'}}{\partial a_k} \dfrac{\partial a_k}{\partial z^{(n)}_m}\dfrac{\partial z^{(n)}_m}{\partial W_{dm}} \tag{23}$$

$$\require{cancel}\dfrac{\partial Ln}{\partial W_{dm}}=\sum_n \sum_{k'} \sum_{k}t^{(n)}_{k'} \dfrac{1}{\cancel {y^{(n)}_{k'}}} \cancel {y^{(n)}_{k'}} ( \delta_{kk'}- y^{(n)}_k)\dfrac{\partial a_k}{\partial z^{(n)}_m}\dfrac{\partial z^{(n)}_m}{\partial W_{dm}} \tag{24}$$

$$\dfrac{\partial Ln}{\partial W_{dm}}=\sum_n \sum_{k} ( t^{(n)}_k - y^{(n)}_k)\dfrac{\partial a_k}{\partial z^{(n)}_m}\dfrac{\partial z^{(n)}_m}{\partial W_{dm}} \tag{25}$$

From [derivative of sigmoid](https://m-alcu.github.io/blog/2017/12/20/sigmoid-derivative/)  

$$z_m = sigmoid(\sum^D_{d=1} W_{dm}x_d) \tag{27}$$  

$$\dfrac{\partial a_k}{\partial z^{(n)}_m}= \dfrac{\partial}{\partial z^{(n)}_m}(\sum^M_{m'=1} V_{m'k}z_{m'}) = V_{mk} \tag{28}$$

$$\dfrac{\partial z^{(n)}_m}{\partial W_{dm}} = z^{(n)}_m(1-z^{(n)}_m) x_d \tag{29}$$

$$\bbox[5px,border:2px solid black] {\begin{align*} \dfrac{\partial Ln}{\partial W_{dm}}=\sum_n \sum_{k} ( t^{(n)}_k - y^{(n)}_k) V_{mk}z^{(n)}_m(1-z^{(n)}_m)x^{(n)}_d \end{align*}} \tag{30}$$


