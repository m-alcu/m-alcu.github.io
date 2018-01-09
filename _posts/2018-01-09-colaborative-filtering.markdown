---
layout: post
title: recomendation systems
date: 2018-01-09 21:26
comments: true
hidden: 0
external-url:
categories: recomendation
---

> This post is to resume different options to do make a recomendation system

All recommendations have two components (ther characteristicas could be known or unknown):
- items to be recommended, the content (m)
- users to recommend items (n)

### Content Based recommendation I

This example is extracted from coursera machine learning course, that basically consists on make a linear regression with the content features and the existing ratings of each user.

It's important to have some ratings from user on other contents in order to guess his preference from the new content.

$x^{(i)}$ = feature vector for each content ($k$ features).  
$\theta^{(j)}$ = parameter vector for user j, $\theta^{(j)} \in \mathbb{R}^{k+1}$ because the bias component

$y^{(i,j)}$ = rating by user j on content i (if defined)
$r(i,j) = 1$  if user $j$ has rated the content $i$ (O otherwise)  


You have all $x^{(i)}$ bu some $y^{(i,j)}$ are not defined.  

Solution: make a linear regresion with known parameters and use the result to compute the gaps: $y^{(i,j)}$  

$$ \underset{\theta^{(j)}}{\text{ min }} \dfrac{1}{2m^{(j)}} \sum_{i:r(i,j)=1} ((\theta^{(j)})^T(x^{(i)})-y^{(i,j)})^2+\dfrac{\lambda}{2m^{(j)}} \sum^n_{k=1} (\theta^{(j)}_k)^2$$

Learn $\theta^{(j)}$ (parameter for user j):

$$ \underset{\theta^{(j)}}{\text{ min }} \dfrac{1}{2} \sum_{i:r(i,j)=1} ((\theta^{(j)})^T(x^{(i)})-y^{(i,j)})^2+\dfrac{\lambda}{2} \sum^n_{k=1} (\theta^{(j)}_k)^2$$

Gradient descent update:

$$\theta^{(j)}_k = \theta^{(j)}_k - \alpha \sum_{i:r(i,j)=1} ((\theta^{(j)})x^{(i)}-y^{(i,j)})x^{(i)}_k \text{ (for $k=0$)}$$

$$\theta^{(j)}_k = \theta^{(j)}_k - \alpha \Biggl(\sum_{i:r(i,j)=1} ((\theta^{(j)})x^{(i)}-y^{(i,j)})x^{(i)}_k + \lambda \theta^{(j)}_k\Biggl) \text{ (for $k>0$)}$$

For each user $j$, content $i$, compute rating as $(\theta^{(j)})^T(x^{(i)})$



