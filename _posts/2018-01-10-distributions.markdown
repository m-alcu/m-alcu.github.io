---
layout: post
title: Probability distributions
date: 2017-12-21 9:01
comments: true
external-url:
categories: probability distributions algebra
---

Main distribution examples:

$$\begin{array}{c|c|c|clcr}\text{Distribution}
 & \text{PDF or PMF} & \text{Mean} & \text{Variance}\\
\hline
 Bernoulli(p)& 
\begin{cases}
p,  & \text{if $x=1$} \\
1-p &  \text{if $x=0$}
\end{cases} & p & p(1-p)\\
\hline
Binomial(n,p) & \binom{n}{k}p^k(1-p)^{n-k} \text{ for } k = 0,1,....n& np & np(1-p)\\
\hline
Geometric(p) & p(1-p)^{k-1} \text{ for } k = 0,1, 2...& \dfrac{1}{p} & \dfrac{1-p}{p^2}\\
\hline
Poisson(\lambda) & \dfrac{e^{- \lambda}\lambda^k}{k!} \text{ for } k = 0,1,2...& \lambda & \lambda\\
\hline
Uniform(a,b) & \dfrac{1}{b-a} \text{ for all } x \in (a,b)& \dfrac{a+b}{2} & \dfrac{(b-a)^2}{12}\\
\hline
Gaussian(\mu,\sigma) & \dfrac{1}{\sigma \sqrt{2\pi}}e^{-\dfrac{(x- \mu)^2}{2\sigma^2}} \text{ for all } x \in (-\infty,\infty)& \mu & \sigma^2\\
\hline
Exponential(\lambda) & \lambda e^{-\lambda x} \text{ for all } x \ge 0, \lambda \ge 0& \dfrac{1}{\mu} & \dfrac{1}{\mu^2}\\
\end{array}$$

PDF: [Probability Desity Funcion](https://en.wikipedia.org/wiki/Probability_density_function)  
PMF: [Probability Mass Function](https://en.wikipedia.org/wiki/Probability_mass_function)

[source](http://cs229.stanford.edu/section/cs229-prob-slide.pdf)