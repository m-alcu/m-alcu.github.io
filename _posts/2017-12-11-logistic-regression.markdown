---
layout: post
title: Logistic Regression (Logit)
date: 2017-12-11 21:45
comments: true
external-url:
categories: python sklearn LogisticRegression
---

> Binomial Logistic Regression is aplied to classification problems, in which there are a list of numerical (Real, integers) features that are related to the classification of one boolean output `Y[0,1]`

Logit function is an useful function that maps an unlimited input to a binary value Y. The logit function is the natural log of the *odds* that Y equals to 0 or 1. This useful function (called sigmoid) maps the $[-\infty,\infty]$ variance of $\beta+\beta_1x_1+\beta_2x_2+...+\beta_kx_k$ to a $[0,1]$ field that is the probability P that output value equals to 1. There is a much better explanation [here](https://codesachin.wordpress.com/2015/08/16/logistic-regression-for-dummies/).

$$\text{logit}(P) = ln\left({P \over 1-P}\right)=\beta+\beta_1x_1+\beta_2x_2+...+\beta_kx_k$$

Clearing P variable show the sigmoid formula:

$$P = {1 \over 1+ e^{-(\beta+\beta_1x_1+\beta_2x_2+...+\beta_kx_k)}}$$  

![sigmoid](/assets/sigmoid.png)

In this article I'm interested in the result from the hand made regression with the above formula versus the common python libraries. 


Performance: Correlated features should be removed for best performance. Number of features increase also the prediction fit results but above a certain limit overfitting occurs and performance is degradated.

