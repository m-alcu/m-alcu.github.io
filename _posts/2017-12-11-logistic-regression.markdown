---
layout: post
title: Logistic Regression (Logit)
date: 2017-12-11 21:45
comments: true
external-url:
categories: python sklearn LogisticRegression
---

> Binomial Logistic Regression is aplied to classification problems, in which there are a list of numerical (Real, integers) features that are related to the classification of one boolean output `[0,1]`

Logit function is an useful function that maps a input to a binary value. 

Correlated features should be removed for best performance. Number of features increase also the prediction fit results but above a certain limit overfitting occurs and performance is degradated.

General formula of Logit:

$$\text{logit}(P) = ln\left({P \over 1-P}\right)=\beta+\beta_1x_1+\beta_2x_2+...+\beta_kx_k$$

$$P = {1 \over 1+ e^{-(\beta+\beta_1x_1+\beta_2x_2+...+\beta_kx_k)}}$$  

![sigmoid](/assets/sigmoid.png)


