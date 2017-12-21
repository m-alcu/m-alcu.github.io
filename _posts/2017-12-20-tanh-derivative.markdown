---
layout: post
title: Derivative of $Tanh(x)$
date: 2017-12-20 20:49
comments: true
external-url:
categories: derivatives algebra tanh
---

Derivative of $tanh(x)$. It is used to gradient decent minimal of cost function.

$$tanh(x) =  {sinh(x) \over cosh(x)}$$

Previous: Derivative of $sinh(x)$

$$sinh(x) = {e^x-e^{-x} \over 2}$$

$$\frac d{dx}sinh(x) = \frac d{dx}\bigl({e^x-e^{-x} \over 2}\bigl)$$

$$= \frac 1 2 \frac d{dx}\bigl(e^x\bigl)-\frac 1 2\frac d{dx}\bigl(e^{-x}\bigl)$$

$$= \frac 1 2 e^x + \frac 1 2 e^{-x}$$

$$= {e^x+e^{-x} \over 2}$$

$$= cosh(x)$$

Previous: Derivative of $cosh(x)$

$$cosh(x) = {e^x+e^{-x} \over 2}$$

$$\frac d{dx}cosh(x) = \frac d{dx}\bigl({e^x+e^{-x} \over 2}\bigl)$$

$$= \frac 1 2 \frac d{dx}\bigl(e^x\bigl)+\frac 1 2\frac d{dx}\bigl(e^{-x}\bigl)$$

$$= \frac 1 2 e^x - \frac 1 2 e^{-x}$$

$$= {e^x-e^{-x} \over 2}$$

$$= sinh(x)$$

Now we can derive $tanh(x)$...

$$\frac d{dx}tanh(x) = \frac d{dx}\bigl({sinh(x) \over cosh(x)}\bigl)$$

$$= {cosh(x)\frac d{dx}sinh(x) - sinh(x)\frac d{dx}cosh(x) \over cosh^2(x)}$$

$$\frac d{dx}sinh(x) = cosh(x)$$

$$\frac d{dx}cosh(x) = sinh(x)$$

$$= {cosh(x)cosh(x) - sinh(x)sinh(x) \over cosh^2(x)}$$

$$=1-tanh^2(x)$$

