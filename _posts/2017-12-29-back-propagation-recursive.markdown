---
layout: post
title: Neural network back propagation recursive
date: 2017-12-29 19:55
comments: true
external-url:
categories: neural_network back_propagation
---

> Demonstration of back recursive propagation on neural networks

Purpose of this post is to show that calculation of error propagation is recursive:

Options for neural network:

$$\begin{array}{c|lcr}
 & f(x) &f'(x)\\
\hline
 sigmoid& {s(x)} & \bbox[yellow,5px]{s(x)(1-s(x))} \\
\hline
tanh & tanh(x) & 1-tanh^{2}(x) \\
\hline
relu & relu(x) & 1(x>0) \\
\end{array}$$


![recursive network](/assets/network-recursive.png)

$$ \dfrac{\partial L}{\partial W^3_{sk}}= \bbox[white,5px]{(t_k-y_k)}z^3_s $$

$$\dfrac{\partial L}{\partial W^2_{rs}}= \bbox[silver,5px]{\sum_k\bbox[white,5px]{(t_k-y_k)}W^3_{sk}\bbox[yellow,5px]{z^3_s(1-z^3_s)}}z^2_r$$

$$\dfrac{\partial L}{\partial W^1_{qr}}=\bbox[grey,5px]{\sum_s\bbox[silver,5px]{\sum_k\bbox[white,5px]{(t_k-y_k)}W^3_{sk}\bbox[yellow,5px]{z^3_s(1-z^3_s)}}W^2_{rs}\bbox[yellow,5px]{z^2_r(1-z^2_r)}}z^1_q$$

$$\dfrac{\partial L}{\partial W^0_{dq}}= \sum_r \bbox[grey,5px]{\sum_s\bbox[silver,5px]{\sum_k\bbox[white,5px]{(t_k-y_k)}W^3_{sk}\bbox[yellow,5px]{z^3_s(1-z^3_s)}}W^3_{rs}\bbox[yellow,5px]{z^2_r(1-z^2_r)}}W^1_{qr}\bbox[yellow,5px]{z^1_q(1-z^1_q)}x_d$$

![recursive network](/assets/network-recursive-2.png)

$$\dfrac{\partial L}{\partial W_{m_nm_{n-1}}}=\sum_{m_{n-1}}\bbox[silver,5px]{\dfrac{\partial L}{\partial W_{m_{n-1}m_{n-2}}}\dfrac{1}{z_{m_{n-1}}}}W_{m_{m+1}m_n}\bbox[yellow,5px]{z_{m_{n-1}}(1-z_{m_{n-1}})}z_{m_n}$$