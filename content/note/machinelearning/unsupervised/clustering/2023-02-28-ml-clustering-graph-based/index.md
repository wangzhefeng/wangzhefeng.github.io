---
title: 基于图论的聚类
author: 王哲峰
date: '2023-02-28'
slug: ml-clustering-graph-based
categories:
    - machinelearning
tags: 
    - model
---

# 基于图论的距离


# Affinity Propagation

> Affinity Propagation，亲和力传播

Affinity  Propagation 是一种基于图论的聚类算法，旨在识别数据中的 "exemplars"(代表点)和 "clusters"(簇)。
与 K-Means 等传统聚类算法不同，Affinity Propagation 不需要事先指定聚类数目，也不需要随机初始化簇心，
而是通过计算数据点之间的相似性得出最终的聚类结果

Affinity Propagation 算法的优点是不需要预先指定聚类数目，且能够处理非凸形状的簇。
但是该算法的计算复杂度较高，需要大量的存储空间和计算资源，并且对于噪声点和离群点的处理能力较弱

```python
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.cluster import AffinityPropagation

# model
af = AffinityPropagation(preference = -563, random_state = 0).fit(X)
# cluster centers
cluster_centers_indices = af.cluster_centers_indices_

# cluster labels
af_labels = af.labels_

# number of clusters
n_clusters_ = len(cluster_centers_indices)
print(n_clusters_)

# result
plt.close("all")
plt.figure(1)
plt.clf()

colors = cycle("bgrcmykbgrcmykbgrcmykbgrcmyk")
for k, col in zip(range(n_clusters_), colors):
    class_members = af_labels == k
    cluster_center = X[cluster_centers_indices[k]]
    plt.plot(X[class_members, 0], X[class_members, 1], col + ".")
    plt.plot(
        cluster_center[0],
        cluster_center[1],
        "o",
        markerfacecolor = col,
        markeredgecolor = "k",
        markersize = 14,
    )
    for x in X[class_members]:
        plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

plt.title("Estimated number of clusters: %d" % n_clusters_)
plt.show()
```

# 参考
