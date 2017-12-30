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

![recursive network](/assets/network-recursive.png)

$$ \dfrac{\partial L}{\partial W^3_{sk}}= \bbox[white,5px]{(t_k-y_k)}z^3_s $$

$$\dfrac{\partial L}{\partial W^2_{rs}}= \bbox[aqua,5px]{\sum_k\bbox[white,5px]{(t_k-y_k)}W^3_{sk}\bbox[yellow,5px]{z^3_s(1-z^3_s)}}z^2_r$$

$$\dfrac{\partial L}{\partial W^1_{qr}}=\bbox[fuchsia,5px]{\sum_s\bbox[aqua,5px]{\sum_k\bbox[white,5px]{(t_k-y_k)}W^3_{sk}\bbox[yellow,5px]{z^3_s(1-z^3_s)}}W^2_{rs}\bbox[yellow,5px]{z^2_r(1-z^2_r)}}z^1_q$$

$$\dfrac{\partial L}{\partial W^0_{dq}}= \sum_r \bbox[fuchsia,5px]{\sum_s\bbox[aqua,5px]{\sum_k\bbox[white,5px]{(t_k-y_k)}W^3_{sk}\bbox[yellow,5px]{z^3_s(1-z^3_s)}}W^3_{rs}\bbox[yellow,5px]{z^2_r(1-z^2_r)}}W^1_{qr}\bbox[yellow,5px]{z^1_q(1-z^1_q)}x_d$$