---
title: 时间序列聚类
author: 王哲峰
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
</style>

<details><summary>目录</summary><p>

- [时间序列聚类算法简介](#时间序列聚类算法简介)
- [时间序列相似性度量](#时间序列相似性度量)
  - [闵可夫斯基距离](#闵可夫斯基距离)
  - [马哈拉诺比斯距离](#马哈拉诺比斯距离)
  - [DTW 距离](#dtw-距离)
  - [参数距离](#参数距离)
    - [基于相关性的相似度度量方法](#基于相关性的相似度度量方法)
    - [基于自相关系数的相似度度量方法](#基于自相关系数的相似度度量方法)
    - [基于周期性的相似度度量方法](#基于周期性的相似度度量方法)
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

# 时间序列相似性度量

两个时间序列的相似定义很简单：距离最近且形状相似。那么如何量化这些相似呢？
直觉的，要对序列之间的距离进行量化。
对于长度相同的序列，计算每两点之间的距离然后求和，距离越小相似度越高

![img](images/Euclidean_DTW_distance.png)

## 闵可夫斯基距离

> Minkowski Distance

定义：

闵氏距离比大多数距离度量更复杂。它是在范数向量空间(`$n$` 维实数空间)中使用的度量，
这意味着它可以在一个空间中使用，在这个空间中，距离可以用一个有长度的向量来表示

`$$D(x, y) = \Bigg(\sum_{i=1}^{n}|x_{i} - y_{i}|^{p}\Bigg)^{\frac{1}{p}}$$`

最有趣的一点是，我们可以使用参数 `$p$` 来操纵距离度量，使其与其他度量非常相似。
常见的 `$p$` 值有：

* `$p=1$`：曼哈顿距离
* `$p=2$`：欧氏距离
* `$p=\infty$`：切比雪夫距离

缺点：

闵氏距离与它们所代表的距离度量有相同的缺点，因此，对：曼哈顿距离、
欧几里得距离和切比雪夫距离等度量标准有个好的理解非常重要。
此外，参数 `$p$` 的使用可能很麻烦，因为根据用例，查找正确的 `$p$` 值在计算上效率低

闵可夫斯基距离比较直观，但是它与数据的分布无关，具有一定的局限性，
如果 `$x$` 方向的幅值远远大于 `$y$` 方向的幅值，这个距离公式就会过度被 `$x$` 维度的作用。
因此在加算前，需要对数据进行变换(去均值，除以标准差)。
这种方法在假设数据各个维度不相关的情况下，利用数据分布的特性计算出不同的距离。
如果数据维度之间数据相关，这时该类距离就不合适了

用例：

`$p$` 的积极一面是可迭代，并找到最适合用例的距离度量。
它允许在距离度量上有很大的灵活性，如果你非常熟悉 `$p$` 和许多距离度量，将会获益多多

## 马哈拉诺比斯距离

> Mahalanobis Distance

若不同维度之间存在相关性和尺度变换等关系，需要使用一种变化规则，
将当前空间中的向量变换到另一个可以简单度量的空间中去测量

假设样本之间的协方差矩阵是 `$\Sigma$`，利用矩阵分解(LU 分解)可以转换为下三角矩阵和上三角矩阵的乘积:

`$$\Sigma = LL^{T}$$`

消除不同维度之间的相关性和尺度变换，需要对样本 `$x$` 做如下处理：

`$$z = L^{-1}(x-\mu)$$`

经过处理的向量就可以利用欧式距离进行度量：

`$$D(x) = z^{T}z = (x - \mu)^{T}\Sigma^{-1}(x-\mu)$$`

## DTW 距离

当序列长度不相等的时候，如何比较两个序列的相似性呢？
这里我们需要动态时间规整(Dynamic Time Warping；DTW)方法，
该方法是一种将时间规整和距离测度相结合的一种非线性规整技术。
主要思想是把未知量均匀地伸长或者缩短，直到与参考模式的长度一致，
在这一过程中，未知量的时间轴要不均匀地扭曲或弯折，
以使其特征与参考模式特征对正。DTW 距离可以帮助我们找到更多序列之间的形状相似

假设有两个时间序列 `$A = \{a_{1}, a_{2}, \ldots, a_{n}\}$` 和 `$B = \{b_{1}, b_{2}, \ldots, b_{m}\}$`。
构造一个尺寸为 `$(n, m)$` 的矩阵，
第 `$(i, j)$` 单元记录了两个点 `$(a_{i}, b_{j})$` 之间的欧式距离 `$d(a_{i}, b_{j}) = |a_{i} - b_{j}| = \sqrt{(a_{i} - b_{j})^{2}}$`。如下图：

![img](images/dtw_a.png)

一条弯折的路径 `$W$`，由若干个彼此相连的矩阵单元构成，这条路径描述了 `$A$` 和 `$B$` 之间的一种映射。
设第 `$k$` 个单元定义为 `$\omega_{k} = (i, j)_{k}$`，则：

`$$\omega = \{\omega_{1}, \omega_{2}, \ldots, \omega_{K}\}, max\{n, m\} \leq K \leq n+m-1$$`

这条弯折的路径满足以下条件：

1. 边界条件
    - `$\omega_{1} = (1, 1)$` 且 `$\omega_{k} = (n, m)$`
2. 连续性
    - 设 `$\omega_{k} = (a, b)$`，`$\omega_{k-1}=(a', b')$`，那么 `$a-a' \leq 1$`，`$b-b' \leq 1$`
3. 单调性
    - 设 `$\omega_{k} = (a, b)$`，`$\omega_{k-1}=(a', b')$`，那么 `$a-a' \geq 0$`，`$b-b' \geq 0$`

在满足上述条件的多条路径中，距离最短的、花费最少的一条路径是：

`$$DTW(A, B) = min\Big(\frac{1}{K}\sqrt{\sum_{k=1}^{K}\omega_{k}}\Bigg)$$`

DTW 距离的计算是一个动态规划过程，先用欧式距离初始化矩阵，然后使用如下递推公式进行求解：

`$$r(i, j) = d(i, j) + min\{r(i-1, j-1)), r(i-1, j), r(i, j-1)\}$$`

## 参数距离

除了直接的度量原始时间序列之间的距离，很多方法尝试对时间序列进行建模，
并比较其模型参数的相似度。之前我们介绍了时间序列的统计分析方法，这里可以用到的有：

* 基于相关性的相似度度量方法
* 基于自相关系数的相似度度量方法
* 基于周期性的相似度度量方法

还有通过 ARMA 模型抽象时序的参数，进行举例度量，类似方法这里不作展开

### 基于相关性的相似度度量方法

量化两条序列 `$X$` 与 `$Y$` 之间的相关系数：

`$$cor(X_{T}, Y_{T}) = \frac{\sum_{t=1}^{T}(x_{t} - \bar{X}_{T}) \cdot (y_{t} - \bar{Y}_{T})}{\sqrt{\sum_{t=1}^{T}(x_{t} - \bar{X}_{T})} \cdot \sqrt{\sum_{t-1}^{T}(y_{t} - \bar{Y}_{T})^{2}}}$$`

* 如果相关系数 `$cor(X_{T}, Y_{T}) = 1$`，表示它们完全一致
* 如果相关系数 `$cor(X_{T}, Y_{T}) = -1$`，表示它们之间是负相关的

### 基于自相关系数的相似度度量方法

分别抽取曲线X与Y的自相关系数：

`$$\hat{\rho}_{k}  = \frac{\sum_{t=1}^{T-k}(x_{t} - \mu)\cdot(x_{t+k} - \mu)}{(T - k)\sigma^{2}}$$`

定义时间序列之间的距离如下：

`$$D(X_{T}, Y_{T}) = \sqrt{(\hat{\rho}X_{T} - \hat{\rho}Y_{T})^{T}(\hat{\rho}X_{T} - \hat{\rho}Y_{T})}$$`

### 基于周期性的相似度度量方法

通过傅里叶变换得到一组参数，然后通过这组参数来反映原始的两个时间序列时间的距离。数学表达为：

`$$I_{X_{T}}(\lambda_{k}) = T^{-1}\Bigg| \sum_{t=1}^{T}x_{t}e^{-i\lambda_{k}t} \Bigg|^{2}$$`

`$$I_{Y_{T}}(\lambda_{k}) = T^{-1}\Bigg| \sum_{t=1}^{T}y_{t}e^{-i\lambda_{k}t} \Bigg|^{2}$$`

`$$D(X_{T}, Y_{T}) = \frac{1}{n}\sqrt{\sum_{k=1}^{n}(I_{X_{T}}(\lambda_{k}) - I_{Y_{T}}(\lambda_{k}))^{2}}$$`

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
* []()
