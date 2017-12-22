---
layout: post
title: Derivative of softmax
date: 2017-12-15 06:15
comments: true
external-url:
categories: softmax algebra derivatives
---

Softmax is a vector function. It takes a vector a and produces a vector as output. $S(a): \mathbb{R}^N \rightarrow \mathbb{R}^N$

$$S(a) = {\begin{pmatrix}
a_1 \\
a_2\\
\vdots\\
a_N
\end{pmatrix}} \rightarrow {\begin{pmatrix}
S_1 \\
S_2\\
\vdots\\
S_N
\end{pmatrix}}$$

$$S_j = {e^{a_i} \over \sum_{k=1}^N e^{a_k}} \forall j \in 1..N$$

There is not exactly a derivative as in sigmoid or tanh... you have to specify:
* Which output component $S_i$ are you seeking to find the derivative of.
* Which is the input $a_i$ you are asking to derivate from.


The derivative is in fact a Jacobian matrix of $N \times N$:

$$\begin{pmatrix}
 D_1S_1 & D_2S_1 & D_3S_1 & \cdots & D_NS_1 \\
 D_1S_2 & D_2S_2 & D_3S_2 & \cdots & D_NS_2 \\
 \vdots  & \vdots& \vdots & \ddots & \vdots \\
 D_1S_N & D_2S_N & D_3S_N & \cdots & D_NS_N    
 \end{pmatrix}$$

$$\dfrac{\partial S_i}{\partial a_j}= \dfrac{\partial {e^{a_i} \over \sum_{k=1}^N e^{a_k}}}{\partial a_j} = D_jS_i $$

Using cuotient rule for derivatives

$$f(x) = {g(x) \over h(x)} \Rightarrow f'(x) = {g'(x)h(x) +g(x)h'(x) \over h(x)^2}$$

where $g_i = e^{a_i}$ and $h_i = \sum_{k=1}^N e^{a_k}$

No matter which $a_j$ derivative of $h_i$ is always e^{a_j}:

$$\dfrac{\partial h_i}{\partial a_j}= e^{a_j}$$

Derivative of $g_i$:

$$  \dfrac{\partial g_i}{\partial a_j} =
\begin{cases}
e^{a_j}  & \text{if $i = j$,} \\
0 & \text{if $i \neq j.$}
\end{cases}$$

Lets calculate $D_jS_i$ when $i=j$:

$$\begin{align}
D_jS_i & = \dfrac{\partial {e^{a_i} \over \sum_{k=1}^N e^{a_k}}}{\partial a_j} \\
& = \dfrac {e^{a_i }\sum_{k=1}^N e^{a_k}-e^{a_j}e^{a_i}}{(\sum_{k=1}^N e^{a_k})^2} \\
& = \dfrac {e^{a_i}}{\sum_{k=1}^N e^{a_k}}{\dfrac {\dfrac {e^{a_i}}{\sum_{k=1}^N e^{a_k}}-e^{a_i}} {\sum_{k=1}^N e^{a_k}}} \\
& = S_i(1-S_j) \\
\end{align}$$

And calculate $D_jS_i$ when $i \neq j$:

$$\begin{align}
D_jS_i & = \dfrac{\partial {e^{a_i} \over \sum_{k=1}^N e^{a_k}}}{\partial a_j} \\
& = \dfrac {0-e^{a_j}e^{a_i}}{(\sum_{k=1}^N e^{a_k})^2} \\
& = -\dfrac {e^{a_j}}{\sum_{k=1}^N e^{a_k}}{\dfrac {e^{a_i}} {\sum_{k=1}^N e^{a_k}}} \\ 
& = -S_jS_i \\
\end{align}$$

To summarize:

$$  D_jS_i = \dfrac{\partial S_i}{\partial a_j} =
\begin{cases}
S_i(1-S_j)  & \text{if $i = j$,} \\
-S_jS_i & \text{if $i \neq j.$}
\end{cases}$$


Softmax extends binary classification to 'n' levels useful for:  
* Faces  
* Car  
* MNIST Digits 0-9  

Softmax is a generalization of Logistic function. Compress a K-dimension z Real values to a K-dimension $\sigma (z)$

Softmax for K=2 is the same as a sigmoid where $w = w_1 - w_0$ 

Softmax is a generalization of sigmoid for K>2.

Softmax for K Classes:  

$$a_1=w_1^\intercal x \text{   } a_2=w_2^\intercal x \text{   } ...\text{   } a_n=w_n^\intercal x$$

$$P(Y=k|X) = {e^{a_k} \over Z}$$

$$Z = e^{a_1}+e^{a_2}+...+e^{a_K}$$

$$W = [ w_1 w_2...w_K] \text{ (a D x K matrix)}$$

$$A_{N\times K} = X_{N\times D}W_{D\times K}\rightarrow Y_{N\times K}=softmax(A_{N\times K})$$




