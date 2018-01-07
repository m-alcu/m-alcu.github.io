---
layout: post
title: Bayesian networks
date: 2018-01-02 13:26
comments: true
hidden: 1
external-url:
categories: 
---

> Bayesian networks

Chain rule (valid for all distributions:

$$P(x_1,x_2,...x_n) = \prod^n_{i=1} P(x_i|x_1, .... x_{i-1})$$

We assume conditional independences:

$$P(x_i|x_1, .... x_{i-1}) = P(x_i|parents(X_i))$$

$$P(x_1,x_2,...x_n) = \prod^n_{i=1} P(x_i|parents(X_i))$$










