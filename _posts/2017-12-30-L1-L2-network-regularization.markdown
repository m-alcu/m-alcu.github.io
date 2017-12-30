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

Regularization ins a technique to prevent neural networks (and logistics also) to over-fit. Over-fitting occurs when you train a neural network that predicts fully your trained data but predicts poorly on any new test data. 

Regularization increases error (or reduces likelihood).

The two most common forms of regularization are called L1 and L2. In L2 regularization (the most common of the two forms), you modify the error function you use during training to include an additional term that adds a fraction (usually given Greek letter lower case lambda) of the sum of the squared values of the weights. So larger weight values lead to larger error, and therefore the training algorithm favors and generates small weight values.

Lambda is a hyperparameter that controls the L2 regularization.

$$Ln(P(Y=T|X))=\sum_{n=1}^N \sum_{k=1}^K t_k^{(n)}ln(y_k^{(n)})-\dfrac{\lambda}{2n}\sum_m\sum_kV^2_{mk}-\dfrac{\lambda}{2n}\sum_d\sum_mW^2_{dm}$$

$$V_{mk} = V_{mk} - \dfrac{\alpha}{n} \sum_n ( t^{(n)}_k-y^{(n)}_k)z^{(n)}_m - \dfrac{\lambda}{n}V_{mk}$$

$$W_{dm} = W_{dm}-\dfrac{\alpha}{n} \sum_n \sum_{k} ( t^{(n)}_k - y^{(n)}_k) V_{mk}z^{(n)}_m(1-z^{(n)}_m)x^{(n)}_d - \dfrac{\lambda}{n}W_{dm}$$

Lambda is a hyperparameter that controls the L1 regularization.

$$Ln(P(Y=T|X))=\sum_{n=1}^N \sum_{k=1}^K t_k^{(n)}ln(y_k^{(n)})-\dfrac{\lambda}{2n}\sum_m\sum_k|V_{mk}|-\dfrac{\lambda}{2n}\sum_d\sum_m|W_{dm}|$$

$$V_{mk} = V_{mk} - \dfrac{\alpha}{n} \sum_n ( t^{(n)}_k-y^{(n)}_k)z^{(n)}_m - \dfrac{\lambda}{n}sign(V_{mk})$$

$$W_{dm} = W_{dm}-\dfrac{\alpha}{n} \sum_n \sum_{k} ( t^{(n)}_k - y^{(n)}_k) V_{mk}z^{(n)}_m(1-z^{(n)}_m)x^{(n)}_d - \dfrac{\lambda}{n}sign(W_{dm})$$