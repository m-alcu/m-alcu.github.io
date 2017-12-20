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

$$f(x)=g(x) \cdot h(x)\Rightarrow f'(x) = g'(x) \cdot h(x) + g(x) \cdot h'(x)$$

$$f(x) = {g(x) \over h(x)} \Rightarrow f'(x) = {g'(x)h(x) +g(x)h'(x) \over h(x)^2}, \text {if }h(x) \neq 0$$

$$f(x) = g(h(x)) \Rightarrow f'(x) = g'(h(x)) \cdot h'(x) \text{ "chain rule"}$$

