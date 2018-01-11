---
layout: post
title: Distributed Stochastic Neighbor Embedding
date: 2018-01-11 22:38
comments: true
external-url:
categories: visualization
---

If you want to see multidimensionally data, for example the MNIST dataset, there is a machine learning algoritm created by Geofrey Hinton and Laurens van der Maarten by 2008.

Basically comprises two stages:
- calculates the probability distribution of that each point par is similar $p_ij$.
- define a similar probability distribution on a low-dimensional map, and minimizes the diference between the two distributions using gradient descent.

Algorithm is free of use for non commercial use.

Python version (and another languages) is downloadable [here](https://lvdmaaten.github.io/tsne/)

Python version is an example of MNIST dataset that has the following result:

![MNIST result](/assets/result-t-sne.png)