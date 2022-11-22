---
title: 层次聚类
author: 王哲峰
date: '2022-11-22'
slug: ml-clustering-hierarchical
categories:
  - machinelearning
tags:
  - model
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

- [AGNES 层次聚类](#agnes-层次聚类)
  - [算法介绍](#算法介绍)
  - [算法实现](#算法实现)
</p></details><p></p>

# AGNES 层次聚类

## 算法介绍

AGNES(Agglomerative Nesting)是一种采用自底向上聚合策略的层次聚类算法, 算法的步骤为: 

1. 先将数据集中的每个样本当做是一个初始聚类簇;
2. 然后在算法运行的每一步中找出距离最近的两个点(聚类簇)进行合并为一个聚类簇;
3. 上述过程不断重复, 直至所有的样本点合并为一个聚类簇或达到预设的聚类簇个数. 
   最终算法会产生一棵树, 称为树状图(dendrogram), 树状图展示了数据点是如何合并的.

这个算法的关键是如何计算两点之间以及两个聚类簇之间的距离

1. 如何计算两点之间的距离**[距离矩阵(Metric)]**: 

names               | Eng.                       | formula
------------------- | -------------------------- | ----------
欧氏距离            | Euclidean distance         | `$\|x-y\|_2 = \sqrt{\sum_i (x_i-y_i)^2}$`
平方欧氏距离        | Squared Euclidean distance | `$\|x-y\|_2^2 = \sum_i (x_i-y_i)^2$`
曼哈顿距离	        | Manhattan distance         | `$\|x-y\|_1 = \sum_i |x_i-y_i|$`
马氏距离(统计距离)	| Mahalanobis distance       | `$\sqrt{(x-y)^{T} \Sigma^{-1}(x-y)}$`
极大值距离	        | Maximum distance           | `$\|x-y\|_{\infty} = {max}_{i} |x_i-y_i|$`


```r
dist(x, method = "euclidean", diag = FALSE, upper = FALSE, p = 2)

## Default S3 method:
as.dist(m, diag = FALSE, upper = FALSE)

## S3 method for class 'dist'
print(x, 
      diag = NULL, 
      upper = NULL,
      digits = getOption("digits"), 
      justify = "none",
      right = TRUE, 
      ...)

## S3 method for class 'dist'
as.matrix(x, ...)
```

2. 如何计算两个聚类簇(集合)之间的距离**[链接准则(Linkage criteria)]**: 
    - **Complete-linkage clustering(全链接)**: Find the maximum possible distance between points belonging to two different clusters.
`$$d_{max}(C_{i}, C_{j})=\underset{x\in C_{i},y\in C_{j}}{max}dist(x, y)$$`
    - **Single-linkage clustering(单链接)**: Find the minimum possible distance between points belonging to two different clusters.
`$$d_{min}(C_{i}, C_{j})=\underset{x\in C_{i},y\in C_{j}}{min}dist(x, y)$$`
    - **Mean-linkage clustering(平均链接 UPGMA)**: Find all possible pairwise distances for points belonging to two different clusters and then calculate the average.
`$$d_{avg}(C_{i}, C_{j})=\frac{1}{|C_{i}||C_{j}|}\sum_{x\in C_{i}}\sum_{y\in C_{j}}dist(x, y)$$`
    - **Centroid-linkage clustering(中心链接 UPGMC)**: Find the centroid of each cluster and calculate the distance between centroids of two clusters.
`$$d_{cen}(C_{i}, C_{j})=dist(x, y),\quad x,y分别是C_{i},C_{j}的中心$$`
    - **Minimum energy clustering**
`$$d_{ene}(C_{i}, C_{j})=\frac{2}{nm}\sum_{i=1}^{n}\sum_{j=1}^{m}\|x_{i}- y_{j}\|_{2} - \frac{1}{n^2}\sum_{i=1}^{n}\sum_{j=1}^{n}\|x_{i}-x_{j}\|_{2}-\frac{1}{m^2}\sum_{i=1}^{m}\sum_{j=1}^{m}\|y_{i}-y_{j}\|_{2}$$`

**算法:**

> **输入:**
> 
> 样本集: ``$D=\{x_{1}, x_{2}, \ldots, x_{n}\}$``;
> 
> 聚类簇距离度量函数 `$d$`;
>
> 聚类簇数 `$k$`.
> 
> **过程:**
> 
> 1. **for** `$j=1, 2, \ldots, n$` **do**
> 2. `$C_{j}=\{x_{j}\}$`
> 3. **end for**
> 4. **for** `$i=1, 2, \ldots, n$` **do**
> 5. **for** `$j=1, 2, \ldots, n$` **do**
> 6. `$M(i,j)=d(C_{i}, C_{j})$`;
> 7. `$M(i,j)=M(j,i)$`
> 8. **end for**
> 9. **end for**
> 10. 设置当前聚类簇个数: `$q=n$`
> 11. **while** `$q>k$` **do**
> 12. 找出距离最近的两个聚类簇 `$C_{i^{*}}$` 和 `$C_{j^{*}}$`;
> 13. 合并 `$C_{i^{*}}$` 和 `$C_{j^{*}}$` : `$C_{i^{*}}=C_{i^{*}} \cup C_{j^{*}}$`;
> 14. **for** `$j=j^{*}+1,j^{*}+2, \ldots, q$` **do**
> 15. 将聚类簇 `$C_{j}$` 重编号为 `$C_{j-1}$`
> 16. **end for**
> 17. 删除距离矩阵 `$M$` 的 第 `$j^{*}$` 行与第 `$j^{*}$` 列;
> 18. **for** `$j=1, 2, \ldots, q-1$` **do**
> 19. `$M(i^{*},j)=d(C_{i^{*}}, C_{j})$`;
> 20. `$M(j,i^{*})=M(i^{*},j)$`
> 21. **end for**
> 22. `$q=q-1$`
> 23. **end while**
> **输出:**
> 
> 簇划分 ``$C=\{C_{1}, C_{2}, \ldots, C_{k}\}$``,

## 算法实现

**METHOD : Method Complete linkage clustering :**

```r
clusters <- hclust(dist(iris[, 3:4]))
plot(clusters)
clusterCut <- cutree(clusters, 3)
table(clusterCut, iris$Species)
```

**METHOD : Method Mean linkage clustering :**

```r
clusters <- hclust(dist(iris[, 3:4]), method = 'average')
plot(clusters)
clusterCut <- cutree(clusters, 3)
table(clusterCut, iris$Species)
```

```r
library(ggplot2)
ggplot(iris, aes(Petal.Length, Petal.Width, color = iris$Species)) +
    geom_point(alpha = 0.4, size = 3.5) + 
    geom_point(col = clusterCut) + 
    scale_color_manual(values = c('black', 'red', 'green'))
```

