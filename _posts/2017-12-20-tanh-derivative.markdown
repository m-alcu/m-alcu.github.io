---
layout: post
title: Derivative of Tanh
date: 2017-12-20 20:49
comments: true
external-url:
categories: derivatives algebra tanh
---

Derivative of tanh . It is used to gradient decent minimal of cost function.

$$tanh(x) =  {sinh(x) \over cosh(x)}$$

$$\frac d{dx}tanh(x) = \frac d{dx}\bigl({sinh(x) \over cosh(x)}\bigl)$$

$$= {cosh(x)\frac d{dx}sinh(x) - sinh(x)\frac d{dx}cosh(x) \over cosh^2(x)}$$

$$\frac d{dx}sinh(x) = cosh(x)$$

$$\frac d{dx}cosh(x) = sinh(x)$$

$$= {cosh(x)cosh(x) - sinh(x)sinh(x) \over cosh^2(x)}$$

$$=1-tanh^2(x)$$

