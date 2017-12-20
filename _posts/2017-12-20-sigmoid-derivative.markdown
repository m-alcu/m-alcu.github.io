---
layout: post
title: Derivative of Sigmoid ($\sigma$)
date: 2017-12-20 20:10
comments: true
external-url:
categories: derivatives algebra sigmoid
---

Derivative of sigmoid or \sigma . It is used to gradient decent minimal of cost function.

$$sigmoid(x) = s(x) =  {1 \over 1-e^{-x}}$$

$$\frac d{dx}s(x) = \frac {d }{dx}\bigl({1 \over 1-e^{-x}}\bigl)$$

$$ = \bigl( {1 \over 1+e^{-x}} \bigl)^2 \frac {d }{dx}\bigl({1 -e^{-x}}\bigl) \text{quotient rule}$$

$$ = \bigl( {1 \over 1+e^{-x}} \bigl)^2 e^{-x} (-1)$$

$$ = \bigl( {1 \over 1+e^{-x}} \bigl)\bigl( {1 \over 1+e^{-x}} \bigl)( -e^{-x})$$

$$ = \bigl( {1 \over 1+e^{-x}} \bigl)\bigl( { -e^{-x} \over 1+e^{-x}} \bigl)$$

$$ = s(x)(1-s(x))$$

