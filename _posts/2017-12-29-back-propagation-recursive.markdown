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

$$ \dfrac{\partial L}{\partial W^3_{sk}}= (t_k-y_k)z^3_s = \color{red}{\delta^3_{sk}}$$

$$\dfrac{\partial L}{\partial W^2_{rs}}= \sum_k(t_k-y_k)W^3_{sk}z^3_s(1-z^3_s)z^2_r = \sum_k\color{red}{\delta^3_{sk}}W^3_{sk}(1-z^3_s)z^2_r = \color{blue}{\delta^2_{rs}}$$

$$\dfrac{\partial L}{\partial W^1_{qr}}= \sum_s\sum_k(t_k-y_k)W^3_{sk}z^3_s(1-z^3_s)W^2_{rs}z^2_r(1-z^2_r)z^1_q = \sum_s\color{blue}{\delta^2_{rs}}W^2_{rs}(1-z^2_r)z^1_q = \color{fuchsia}{\delta^1_{qr}}$$

$$\dfrac{\partial L}{\partial W^0_{dq}}= \sum_r \sum_s\sum_k(t_k-y_k)W^3_{sk}z^3_s(1-z^3_s)W^3_{rs}z^2_r(1-z^2_r)W^1_{qr}z^1_q(1-z^1_q)x_d = \sum_r\color{fuchsia}{\delta^1_{qr}}W^1_{qr}(1-z^1_q)x_d$$