---
layout: post
title: Neural network L1 and L2 regulatization
date: 2017-12-30 13:10
comments: true
external-url:
categories: neural_network back_propagation regularization
---

> Demonstration of L1 and L2 regularization in back recursive propagation on neural networks

Purpose of this post is to show that additional calculations in case of regularization L1 or L2.

Lambda is a hyperparameter that controls the L2 regularization

$$Ln(P(Y=T|X))=\sum_{n=1}^N \sum_{k=1}^K t_k^{(n)}ln(y_k^{(n)})+\dfrac{\lambda}{2n}\sum_m\sum_kV^2_{mk}+\dfrac{\lambda}{2n}\sum_d\sum_mW^2_{dm}$$

$$V_{mk} = V_{mk} + \alpha \dfrac{\partial Ln}{\partial V_{mk}}$$