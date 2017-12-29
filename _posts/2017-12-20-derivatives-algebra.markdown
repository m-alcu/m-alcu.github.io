---
layout: post
title: Derivatives Algebra
date: 2017-12-20 20:10
comments: true
external-url:
categories: derivatives algebra
---

Reason for remind the algebra of derivatives is that they are important in order to understand tha maths behind the gradient algorith that it is used in Deep Learning.

$$f(x)=g(x)+h(x)\Rightarrow f'(x) = g'(x)+h'(x)$$

$$f(x)=g(x) \cdot h(x)\Rightarrow f'(x) = g'(x) \cdot h(x) + g(x) \cdot h'(x) \text{ "product rule"}$$

$$f(x) = {g(x) \over h(x)} \Rightarrow f'(x) = {g'(x)h(x) +g(x)h'(x) \over h(x)^2}, \text {if }h(x) \neq 0 \text{ "quotient rule"}$$

$$f(x) = g(h(x)) \Rightarrow f'(x) = g'(h(x)) \cdot h'(x) \text{ "chain rule"}$$

$$x = x(t) , y )y = y(t), f=f(x,y)$$

$$\frac {df}{dt} =  \dfrac{\partial f}{\partial x} \dfrac{\partial x}{\partial t}+\dfrac{\partial f}{\partial y} \dfrac{\partial y}{\partial t}$$

$$f=f(x_1(t), x_2(t), ...,x_K(t))$$

$$\frac {df}{dt} = \sum^K_{k=1}  \dfrac{\partial f}{\partial x_k} \dfrac{\partial x_k}{\partial t} \text{ "partial derivatives rule"}$$

$$f = |x| \rightarrow f' = \dfrac {x}{|x|} = sign(x)$$

