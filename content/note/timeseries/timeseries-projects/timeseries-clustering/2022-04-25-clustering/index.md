---
title: 时间序列聚类
author: wangzf
date: '2022-04-25'
slug: timeseries-clustering
categories:
  - timeseries
tags:
  - machinelearning
---

<style>
details {
    border: 1px solid #aaa;
    border-radius: 4px;
    padding: .5em .5em 0;
}
summary {
    font-weight: bold;
    margin: -.5em -.5em 0;
    padding: .5em;
}
details[open] {
    padding: .5em;
}
details[open] summary {
    border-bottom: 1px solid #aaa;
    margin-bottom: .5em;
}
img {
    pointer-events: none;
}
</style>

<details><summary>目录</summary><p>

- [时间序列聚类算法简介](#时间序列聚类算法简介)
- [时间序列聚类算法](#时间序列聚类算法)
  - [K-Means DBA 聚类](#k-means-dba-聚类)
  - [K-Medoids 聚类](#k-medoids-聚类)
  - [层次聚类](#层次聚类)
    - [分层凝聚聚类](#分层凝聚聚类)
  - [主动半监督聚类](#主动半监督聚类)
- [参考](#参考)
</p></details><p></p>

# 时间序列聚类算法简介

时间序列的聚类在工业生产生活中非常常见，大到工业运维中面对海量 KPI 曲线的隐含关联关系的挖掘，
小到股票收益曲线中的增长模式归类，都要用到时序聚类的方法帮助发现数据样本中一些隐含的、深层的信息

根据对时间序列的距离度量方法，可以采用机器学习中很多聚类算法

# 时间序列聚类算法

## K-Means DBA 聚类

时间序列的 K 均值聚类需要时间序列的平均策略。一种可能性是 DTW 重心平均(Barycenter Average DBA)

```python
model = KMeans(
    k = 4, 
    max_it = 10, 
    max_dba_it = 10, 
    dists_options = {
        "window": 40
    }
)
cluster_idx, performed_it = model.fit(
    series, 
    use_c = True, 
    use_parallel = False
)
```

对于上面的 Trace 示例，聚类并不完美，因为不同系列的基线略有不同，无法通过归一化进行校正。
这会导致累积误差大于其中一种系列中的细微正弦波。一种可能的解决方案是对信号应用差分以关注序列中的变化。
此外，还应用低通滤波器来避免噪声累积

```python
import dtaidistance
from dtaidistance.clustering.kmeans import KMeans

series = dtaidistance.preprocessing.differencing(
    series, 
    smooth = 0.1
)
model = KMeans(
    k = 4, 
    max_it = 10, 
    max_dba_it = 10, 
    dists_options = {
        "window": 40
    }
)
cluster_idx, performed_it = model.fit(
    series, 
    use_c = True, 
    use_parallel = False
)
```

DTW Barycenter Averaging:

```python
from diaidistance import dtw_barycenter

new_center = dtw_barycenter.dba(series, center, use_c = True)
new_center = dtw_barycenter.dba_loop(
    series, 
    center, 
    max_it = 10, 
    thr = 0.0001, 
    use_c = True
)
```

## K-Medoids 聚类

```python
from dtaidistance import dtw, clustering

s = np.array([
    [0., 0, 1, 2, 1, 0, 1, 0, 0],
    [0., 1, 2, 0, 0, 0, 0, 0, 0],
    [1., 2, 0, 0, 0, 0, 0, 1, 1],
    [0., 0, 1, 2, 1, 0, 1, 0, 0],
    [0., 1, 2, 0, 0, 0, 0, 0, 0],
    [1., 2, 0, 0, 0, 0, 0, 1, 1],
    [1., 2, 0, 0, 0, 0, 0, 1, 1]
])

model = clustering.KMedoids(
    dtw.distance_matrix_fast, 
    {}, 
    k = 3
)
cluster_idx = model.fit(s)
model.plot("kmedoids.png")
```

## 层次聚类

层次聚类是一种很直观的方法，就是一层一层的对数据进行聚类操作，可以自低向上进行合并聚类、也可以自顶向下进行分裂聚类

* 凝聚式：从点作为个体簇开始，每一个步合并两个最接近的簇
* 分裂式：从包含所有个体的簇开始，每一步分裂一个簇，直到仅剩下单点簇为止

所谓凝聚，其大体思想就是在一开始的时候，把点集集合中的每个元素都当做一类，
然后计算每两个类之前的相似度，也就是元素与元素之间的距离；
然后计算集合与集合之前的距离，把相似的集合放在一起，不相似的集合就不需要合并；
不停地重复以上操作，直到达到某个限制条件或者不能够继续合并集合为止

所谓分裂，正好与聚合方法相反。其大体思想就是在刚开始的时候把所有元素都放在一类里面，
然后计算两个元素之间的相似性，把不相似元素或者集合进行划分，
直到达到某个限制条件或者不能够继续分裂集合为止

在层次聚类里面，相似度的计算函数就是关键所在。在这种情况下，可以设置两个元素之间的距离公式，
距离越小表示两者之间越相似，距离越大则表示两者之间越不相似。
除此之外，还可以设置两个元素之间的相似度

### 分层凝聚聚类

> Hierarchical Agglomerative clustering

```python
from dtaidistance import dtw, clustering

# Custom Hierarchical clustering
model1 = clustering.Hierarchical(dtw.distance_matrix_fast, {})
cluster_idx = model1.fit(timeseries)

# Keep track of full tree by using the HierarchicalTree wrapper class
model2 = clustering.HierarchicalTree(model1)
cluster_idx = model2.fit(timeseries)

# pass keyword arguments identical to instantiate a Hierarchical object
model2 = clustering.HierarchicalTree(
    dists_fun = dtw.distance_matrix_fast, 
    dists_options = {}
)
cluster_idx = model2.fit(timeseries)

# SciPy linkage clustering
model3 = clustering.LinkageTree(
    dtw.distance_matrix_fast, 
    {}
)
cluster_idx = model3.fit(timeseries)
```

聚类结果可视化：

```python
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 10))
show_ts_label = lambda idx: "ts-" + str(idx)
model.plot(
    "hierarchy.png", 
    axes = ax, 
    show_ts_label = show_ts_label,
    show_tr_label = True, 
    ts_label_margin = -10,
    ts_left_margin = 10, 
    ts_sample_length = 1
)
```

## 主动半监督聚类

> Active semi-supervised clustering

使用 DTAIDistance 执行主动半监督聚类的推荐方法是使用 COBRAS 进行时间序列聚类。
COBRAS 是一个使用成对约束进行半监督时间序列聚类的库，它本身支持 `dtaidistance.dtw` 和 `kshape`

# 参考

- [时间序列聚类](https://mp.weixin.qq.com/s?__biz=Mzg3NDUwNTM3MA==&mid=2247484837&idx=1&sn=cdc922e6a213064485113bbb9b8e911e&chksm=cecef050f9b9794672f0227b36212a1fcf8acb1916c6e12e923bcdf5e9ac71b0aa9e7bb7f58d&scene=21#wechat_redirect)
* [dtaidistance Clustering](https://dtaidistance.readthedocs.io/en/latest/usage/clustering.html)
* [cobras](https://github.com/ML-KULeuven/cobras)
