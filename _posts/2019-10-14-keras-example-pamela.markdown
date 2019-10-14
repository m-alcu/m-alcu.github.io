---
layout: post
title: k-means example of reduced color quantization image
date: 2019-10-14 18:36
comments: true
external-url:
categories: k-means PIL png
---

K-means could be a good choice in order reduce color quantization

This example show how to reduce from 24 bits to 'n' colors. This example I have selected n=8

1.Imports

```python
import png
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
```

2.Get image from the same folder

```python
from PIL import Image
img = Image.open("lena.png")
im_rgb = img.convert("RGB")
image = np.array(im_rgb)
rows = image.shape[0]
cols = image.shape[1]
```

3.Do k-means with the image pixels (a rgb-vector)

```python
ncolor = 8
rimage = image.reshape(image.shape[0]*image.shape[1],3)
kmeans = KMeans(n_clusters = ncolor, n_init=10, max_iter=200)
kmeans.fit(rimage)
centers = kmeans.cluster_centers_
```

4.Plot results

```python
labels = np.asarray(kmeans.labels_).reshape(rows, cols)
compressed_image = np.zeros((rows, cols,3),dtype=np.uint8 )
for i in range(rows):
    for j in range(cols):
            compressed_image[i,j,:] = centers[labels[i,j],:]

fig, ax = plt.subplots(1, 2, figsize=(16, 6),
                       subplot_kw=dict(xticks=[], yticks=[]))
ax[0].imshow(image)
ax[0].set_title('Original Image', size=16)

ax[1].imshow(compressed_image)
ax[1].set_title(f'{ncolor}-color Image', size=16);
```

![result](/assets/lena_comparison.png)

5.Store image in 8-PNG format

```python
s = []
for i in range(rows):
    s.append(tuple(labels[i].astype(int)))

palette = []
for i in range(ncolor):
    c = (centers[i][0].astype(int), centers[i][1].astype(int), centers[i][2].astype(int))
    palette.append(c)   

w = png.Writer(width=cols, height=rows, palette=palette, bitdepth=8)
f = open('lena8.png', 'wb')
w.write(f, s)
```