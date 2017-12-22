---
layout: post
title: Logistic Regression (Logit)
date: 2017-12-11 21:45
comments: true
external-url:
categories: python sklearn LogisticRegression
---

> Binary Logistic Regression is aplied to classification problems, in which there are a list of numerical (Real, integers) features that are related to the classification of one boolean output `Y[0,1]`. 

Logistic regresion is fine for linealy separable problems, since is a linear clasifier:
* 2D: bounday is a line (as the example in this post)  
* 3D: bounday is a plane  
* 4D-nD: bounday is a hyperplane  
all of them are linear, not curved.  

Another way to see a logistic regression is the neuron (sigmoid), with $X_n+1$ inputs and a unique binary output.

Logit function is an useful function that maps an unlimited input to a binary value Y. The logit function is the natural log of the *odds* that Y equals to 0 or 1. This useful function (called sigmoid) compress the $[-\infty,\infty]$ variance of $\beta+\beta_1x_1+\beta_2x_2+...+\beta_kx_k$ to a $[0,1]$ field that is the probability P that output value equals to 1. There is a much better explanation [here](https://codesachin.wordpress.com/2015/08/16/logistic-regression-for-dummies/).

$$\text{logit}(P) = ln\left({P \over 1-P}\right)=\beta+\beta_1x_1+\beta_2x_2+...+\beta_kx_k$$

Clearing P variable show the sigmoid formula:

$$P = {1 \over 1+ e^{-(\beta+\beta_1x_1+\beta_2x_2+...+\beta_kx_k)}}$$  

![sigmoid](/assets/sigmoid.png)

In this article I'm interested in the result from the hand made regression with the above formula versus the common python libraries. 

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import statsmodels.api as sm
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)


def sigmoid(z):
    return 1/(1 + np.exp(-z))

# View images

# calculate the cross-entropy error
def cross_entropy(T, Y):
    E = 0
    for i in xrange(len(T)):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1 - Y[i])
    return E

N = 100
D = 2

X = np.random.randn(N,D)*2

# center the first 50 points at (-2,-2)
X[:50,:] = X[:50,:] - 2*np.ones((50,D))

# center the last 50 points at (2, 2)
X[50:,:] = X[50:,:] + 2*np.ones((50,D))

# labels: first 50 are 0, last 50 are 1
T = np.array([0]*50 + [1]*50)

# add a column of ones
# ones = np.array([[1]*N]).T # old
ones = np.ones((N, 1))
Xb = np.concatenate((ones, X), axis=1)

# randomly initialize the weights
w = np.random.randn(D + 1)

# calculate the model output
z = Xb.dot(w)


Y = sigmoid(z)
# let's do gradient descent 100 times
learning_rate = 0.1
for i in xrange(100):
    if i % 10 == 0:
        print cross_entropy(T, Y)

    # gradient descent weight udpate
    w += learning_rate * Xb.T.dot(T - Y)

    # recalculate Y
    Y = sigmoid(Xb.dot(w))

print "Final w-hand-made:", w

y2 = pd.Series(T.tolist())
X2 = pd.concat([pd.Series(Xb[:,1].tolist()), pd.Series(Xb[:,2].tolist())], axis=1)

X2 = sm.add_constant(X2)
logit_model=sm.Logit(y2,X2)
result=logit_model.fit()
print(result.summary())

w2 = result.params.values

print "Final w-statsmodels:", w2
# plot the data and separating line
plt.scatter(X[:,0], X[:,1], c=T, s=100, alpha=0.5)
x_axis = np.linspace(-6, 6, 100)
y_axis = -(w[0] + x_axis*w[1]) / w[2]
line_up, = plt.plot(x_axis, y_axis,'r--', label='hand made')
y_axis = -(w2[0] + x_axis*w2[1]) / w2[2]
line_down, = plt.plot(x_axis, y_axis,'g--', label='statsmodels')
plt.legend(handles=[line_up, line_down])
plt.xlabel('X(1)')
plt.ylabel('X(2)')
plt.show()
```


Note: [source](https://github.com/lazyprogrammer/machine_learning_examples)

![results graphic](/assets/logit-graphic.png)

![results summary](/assets/logit.png)

Performance: Correlated features should be removed for best performance. Number of features increase also the prediction fit results but above a certain limit overfitting occurs and performance is degradated.

*Demonstration of gradient descend formulas*

$y$ is the training input (always 0 or 1)  
$m$ is the size of de dataset input x  
$w$ are the parameters of logistic regression  

$$h_w(x_i) = sigmoid(x_i^\intercal w)={1 \over 1-e^{-x_i^\intercal w}} \text{ }\text{ }\text{ }\forall i \in 1..m$$ 

Cost funcion is the error on every iteration (cross entropy error).

$$J_i(w) = \mathrm{Cost}(h_w(x_i),y_i) \text{ }\text{ }\text{ }\forall i \in 1..m$$

$$J(w) = \dfrac{1}{m} \sum_{i=1}^m \mathrm{Cost}(h_w(x_i),y_i)$$

$$\mathrm{Cost}(h_w(x),y) =
\begin{cases}
-\log(h_w(x))  & \text{if $y = 1$,} \\
-\log(1-h_w(x)) & \text{if $y = 0.$}
\end{cases}$$

![graphic of cost](/assets/cost-logit.png)

$$\bbox[5px,border:2px solid black] {\begin{align*}& \mathrm{Cost}(h_w(x),y) = 0 \text{ if } h_w(x) = y \newline & \mathrm{Cost}(h_w(x),y) \rightarrow \infty \text{ if } y = 0 \; \mathrm{and} \; h_w(x) \rightarrow 1 \newline & \mathrm{Cost}(h_w(x),y) \rightarrow \infty \text{ if } y = 1 \; \mathrm{and} \; h_w(x) \rightarrow 0 \newline \end{align*}}$$  

$$J(w) =-\frac{1}{m} \cdot \sum_{i=1}^m\Bigl( y_i log(h_i)+(1-y_i)log(1-h_i)\Bigl)$$  

$$J(w) = \frac{1}{m} \cdot \left(-y_{M\times 1}^\intercal \log(h_w(x)_{M\times 1})-(1-y_{M\times 1})^\intercal \log(1-h_w(x)_{M\times 1})\right)$$  

There is a global minimum thar can be achieved using gradient descent.

$$\begin{align*}& Repeat \; \lbrace \newline & \; w_j := w_j - \alpha \dfrac{\partial}{\partial w_j}J(w) \newline & \rbrace\end{align*}$$

$$f(x) = log_a(g(x)) \rightarrow f'(x) = \dfrac {g'(x)}{g(x)}log_ae$$

$$\dfrac{\partial}{\partial w_j}J(w) =\dfrac{\partial}{\partial w_j} \Biggl(\dfrac{-1}{m} \sum_{i=1}^m \mathrm{Cost}(h_w(x_i),y_i)\Biggl)$$

$$=\dfrac{-log_ae}{m}\sum_{i=1}^m {y_i \dfrac{1}{\require{cancel}\cancel {h_w(x_i)}}\cancel {h_w(x_i)}(1-h_w(x_i))x_i+ (1-y_i) \dfrac{1}{1-h_w(x_i)}h_w(x_i)(h_w(x_i)-1)x_i} $$

$$=\dfrac{-log_ae}{m}\sum_{i=1}^m {y_i(1-h_w(x_i))x_i- (1-y_i) \dfrac{1}{\cancel{1-h_w(x_i)}}h_w(x_i)(\cancel{1-h_w(x_i)})x_i} $$

$$=\dfrac{-log_ae}{m}\sum_{i=1}^m {y_i(1-h_w(x_i))x_i- (1-y_i)h_w(x_i)x_i} $$

$$=\dfrac{log_ae}{m}\sum_{i=1}^m {(h_w(x_i)-y_i)x_i} $$

$$\begin{align*} & Repeat \; \lbrace \newline & \; w_j := w_j - \frac{\alpha}{m} \sum_{i=1}^m (h_w(x^{(i)}) - y^{(i)}) x_j^{(i)} \newline & \rbrace \end{align*}$$

$$w := w_{J\times 1} - \frac{\alpha}{m} X_{M\times J}^\intercal (h_{M\times 1} - y_{M\times 1})$$  

$$w := w_{J\times 1} + \gamma X_{M\times J}^\intercal (y_{M\times 1} - h_{M\times 1})$$  


Glosary  
* *MLE*: Maximum Likelihood Estimation  
* *sigmoid*: function to map $[-\infty,\infty]$ to $[0,1]$  
* *Epoch*: Epoch means one iteration of Stochastic Gradient Descent



