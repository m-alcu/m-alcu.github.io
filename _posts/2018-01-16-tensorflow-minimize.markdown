---
layout: post
title: Minimize function in Tensorflow
date: 2018-01-16 21:56
comments: true
external-url:
categories: tensorflow
---

This post is to show how easy is to use tensorflow simply to minimize a function. Found [here](https://stackoverflow.com/questions/41918795/minimize-a-function-of-one-variable-in-tensorflow)

```python
import tensorflow as tf

x = tf.Variable(10.0, trainable=True)
f_x = 2 * x* x - 5 *x + 4

loss = f_x
opt = tf.train.GradientDescentOptimizer(0.1).minimize(f_x)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        print(sess.run([x,loss]))
        sess.run(opt)
```

x minimizes to the minimum at 1.25 and the los is the value at that point that is 0.875

