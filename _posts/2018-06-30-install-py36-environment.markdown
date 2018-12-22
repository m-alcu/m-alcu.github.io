---
layout: post
title: phyton 3.6 jupyter notebook environment
date: 2018-06-30 18:36
comments: true
external-url:
categories: jupyter notebook
---

First commands in order to have a neat environment (recommend 3.6 version)

1. Install Anaconda 

anaconda: [download](https://www.anaconda.com/download/#macos)

2. Create an environment

```python
conda create -n py36 python=3.6
```

3. Enter into the environment (MacOS command)

```python
source activate py36
```

4. Install the following libraries

```python
conda install jupyter
conda install numpy
conda install pandas
conda install scikit-learn
conda install matplotlib
pip install --upgrade tensorflow
```


