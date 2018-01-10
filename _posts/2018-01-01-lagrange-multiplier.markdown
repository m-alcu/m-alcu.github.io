---
layout: post
title: Lagrange multiplier
date: 2018-01-01 10:42
comments: true
external-url:
categories: algebra
---

> Lagrange multiplier is a mathematical optimization to obtain local maxima and minima subject to constraints, that is needed to understand regulatization L1 and L2.

maximize f(x,y)
subject to g(x,y) = 0

f, g are continuous first partial derivatives

$$\mathcal{L}(x,y,\lambda) = f(x,y) - \lambda g(x,y)$$

Then if $f(x_0,y_0)$ is a maxium of $f(x,y)$, then there exists $\lambda_0$ such that $\mathcal{L}(x_0, y_0, \lambda_0)$ is a stationary point (stationary points are those points where the partial derivatives of ${\mathcal {L}}$ are zero).

source: [Lagrange multiplier](https://en.wikipedia.org/wiki/Lagrange_multiplier)  
