---
title: Cluster
subtitle: 聚类模型
list_pages: true
# order_by: title
---

# 基于原型的聚类

基于原型的聚类(Prototype-based Clustering) 假设聚类结构能通过一组原形刻画. 
通常情况下, 算法先对原型进行初始化, 然后对原型进行迭代更新求解, 
采用不同的原型表示, 不同的求解方式, 将产生不同的算法

常见的基于原型的聚类有：

* [K 均值聚类(K-Means)]()
* [学习向量量化聚类(Learning Vector Quantization)]()
* [高斯混合聚类(Mixture-of-Gaussian)]()

# 基于密度的聚类

基于密度的聚类(Density-based Clustering) 假设聚类结构能通过样本分布的紧密程度确定. 
密度聚类算法从样本密度的角度来考察样本之间的可连续性, 并基于可连续样本不断扩展聚类簇以获得最终的聚类结果

常见的基于密度的聚类有：

* [DBSCAN(Density-Based Spatial Clustering of Application with Noise)](https://en.wikipedia.org/wiki/DBSCAN)
* [OPTICS(Ordering Points To Identify the Clustering Structure)](https://en.wikipedia.org/wiki/OPTICS_algorithm)

# 层次聚类算法介绍

层次聚类(Hierarchical Clustering) 也称为基于连通性的聚类. 
这种算法试图在不同层次对数据进行划分, 从而形成树形的聚类结构

数据集的划分采用不同的策略会生成不同的层次聚类算法：

* "自底向上" 的聚合策略
    - [AGNES(Agglomerative Nesting)]()
* "自顶向下" 的分拆策略

# 常见聚类算法

| 方法 | 参数 | 可扩展性 | 用例 | 几何 |
|---------------------|----|----|----|----|
| K-Means                  | | | | |
| Affinity propagation      | | | | |
| Mean-shift               | | | | |
| Spectral clustering      | | | | |
| Ward hierarchical        | | | | |
| Agglomerative clustering | | | | |
| DBSCAN                   | | | | |
| OPTICS                   | | | | |
| Gaussian                 | | | | |
| BIRCH                    | | | | |
| Bisecting K-Means        | | | | |


# 聚类算法库

* sklearn
    - sklearn.cluster
    - sklearn.feature_extraction
    - sklearn.metrics.pairwise




# 文档

