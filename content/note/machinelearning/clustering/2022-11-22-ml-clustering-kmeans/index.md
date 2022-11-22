---
title: K-Means 聚类
author: 王哲峰
date: '2022-11-22'
slug: ml-clustering-kmeans
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

- [K-Means](#k-means)
  - [算法原理](#算法原理)
  - [算法优劣性](#算法优劣性)
  - [算法数学模型](#算法数学模型)
  - [算法伪代码](#算法伪代码)
  - [算法实现](#算法实现)
    - [r 实现](#r-实现)
    - [python实现](#python实现)
- [K-Means++](#k-means-1)
- [距离计算优化 elkan K-Means](#距离计算优化-elkan-k-means)
- [Mini Batch K-Means](#mini-batch-k-means)
- [PAM 聚类](#pam-聚类)
  - [算法原理](#算法原理-1)
  - [算法优劣性](#算法优劣性-1)
  - [算法数学模型](#算法数学模型-1)
  - [算法伪代码](#算法伪代码-1)
  - [算法实现](#算法实现-1)
    - [r 实现](#r-实现-1)
    - [Python 实现](#python-实现)
</p></details><p></p>

# K-Means

## 算法原理

1. 初始化数据, 选择 `$k$` 个对象作为中心点, 对于 `$k$` 的选择, 需要经过交叉验证等方法进行选取；
2. 遍历整个数据集, 计算每个点与每个中心点的距离, 将它们分配给距离中心最近的组；
3. 重新计算每个组的平均值, 作为新的聚类中心；
4. 重复上面的 2-3 步, 直到函数收敛, 不再有新的分组情况出现. 

## 算法优劣性

优点: 

* 原理比较简单, 实现容易, 收敛速度快；
* 聚类效果比较好；
* 算法的可解释性较强；
* 主要的参数仅有一个 `$k$`；

缺点: 

* K-means只适用于连续型数据集；
* `$k$`值不容易选取；
* 如果各隐含类别的数据不平衡, 比如各隐含类别的数据量严重失衡, 或者各隐含类别的方差不同, 则聚类效果不佳；
* 采用迭代方法, 得到的结果只是局部最优；
* K-means对数据异常点(极端值)比较敏感；

## 算法数学模型

给定样本集 `$D=\{x_{1}, x_{2}, \ldots, x_{n}\}$`, 
K-Means 算法针对聚类所得簇划分 `$C=\{C_{1}, C_{2}, \ldots, C_{k}\}$`, 
最小化平方误差:

`$$E = \sum^{k}_{i=1}\sum_{x \in C_{i}}||x-\mu_{i}||_{2}^{2}$$`

其中 `$\mu_{i}$` 是簇 `$C_{i}$` 的均值向量: 

`$$\mu_{i}=\frac{1}{|C_{i}|}\sum_{x \in C_{i}}x$$`

直观上看, 平方误差在一定程度上刻画了簇内样本围绕均值向量的紧密程度, 
`$E$` 值越小簇内样本相似度越高. 但最小化 `$E$` 不容易, 是一个 NP 难问题, 
K-Means 算法采用了贪心策略, 通过迭代优化来近似求解 `$E$` 的最小值. 
具体算法如下. 

## 算法伪代码

> **输入:**
> 
> * 样本集: `$D=\{x_{1}, x_{2}, \ldots, x_{n}\}$`;
> * 聚类簇数: `$k$`;
>
> **过程:**
> 
> 1. 从样本集中随机取出 `$k$` 个样本作为初始均值向量: `$\{\mu_{1}, \mu_{2}, \ldots,\mu_{k}\}$`
> 2. **repeat**
> 3. 令 `$C_{i}=\emptyset (1 \leqslant i \leqslant k)$`
> 
> 4. **for** `$j=1, 2, \ldots, n$` **do**
> 5. 计算样本 `$x_{j}$` 与各均值向量 `$\mu_{i} (1 \leqslant i \leqslant k)$` 的距离: `$d_{ji}=\|x_{j}-\mu_{i}\|_{2}$`；
> 6. 根据距离最近的均值向量确定 `$x_{j}$` 的簇标记 `$\lambda_{j}=arg min_{i \in \{1, 2, \ldots, k\}}d_{ji}$`；
> 7. 将样本 `$x_{j}$` 划入相应的簇: `$C_{\lambda_{j}}=C_{\lambda_{j}} \cup \{x_{j}\}$`; 
> 8. **end for**
> 
> 9. **for** `$i=1, 2, \ldots, k$` **do**
> 10. 计算新均值向量: `$\mu_{i}^{'}=\frac{1}{|C_{i}|}\sum_{x \in C_{i}}x$`;
> 11. **if** `$\mu_{i}^{'} \neq \mu_{i}$` **then**
> 12. 将当前均值向量 `$\mu_{i}$` 更新为 `$\mu_{i}^{'}$`
> 13. **else**
> 14. 保持当前均值向量不变
> 15. **end if**
> 16. **end for**
>
> 17. **until** 当前均值向量均未更新
> 
> **输出:**
> 
> 簇划分 `$C=\{C_{1}, C_{2}, \ldots, C_{k}\}$`,

## 算法实现

### r 实现

data

```r
# data
set.seed(0)
df = iris[, -5]
```

kmens:

```r
# clustering
cl = kmeans(df, 3)
cl
```

flexclust::kcca:

```r
if(!require(flexclust)) install.packages("flexclust")
clk = flexclust::kcca(df, k = 3)
clk
clk@centers
summary(clk)
```

### python实现



# K-Means++

1. 从输入的数据集合中随机选择一个点作为第一个聚类中心 `$\mu_1$`
2. 对于数据集中的每一个点 `$x_i$`, 计算它与已选择的聚类中心的距离, 
   最小的距离 `$$D(x)=arg min \sum_{i=1}^{k}||x_i-\mu_i||_{2}^{2}$$`
3. 选择一个新的数据点作为新的聚类中心, 选择的原则是: `$D(x)$` 较大的点, 被选做聚类中心的概率较大；
4. 重复2-3直到选择出 `$k$` 个聚类中心；
5. 利用这 `$k$` 个质心去初始化质心, 然后运行标准的K-Means算法；

# 距离计算优化 elkan K-Means

在传统的 K-Means 算法中, 在每轮迭代时, 要计算所有的样本点到所有的质心的距离, 
这样会比较的耗时. 那么, 对于距离的计算有没有能够简化的地方呢？
elkan K-Means算法就是从这块入手加以改进. 
它的目标是减少不必要的距离的计算. 那么哪些距离不需要计算呢？
elkan K-Means利用了两边之和大于等于第三边,以及两边之差小于第三边的三角形性质, 来减少距离的计算. 

* 第一种规律是对于一个样本点 `$x$` 和两个质心 `$\mu_i$`, `$\mu_j$`. 
  如果我们预先计算出了这两个质心之间的距离 `$D(i,j)$`, 
  则如果计算发现 `$2D(x,j)≤D(i,j)$`, 我们立即就可以知道 `$D(x,i)≤D(x,j)$`. 
  此时我们不需要再计算 `$D(x,j)$`, 也就是说省了一步距离计算. 
* 第二种规律是对于一个样本点 `$x$` 和两个质心 `$\mu_i$`, `$\mu_j$`. 
  我们可以得到 `$D(x,j)≥max{0,D(x,i)−D(i,j)}$`. 这个从三角形的性质也很容易得到. 

利用上边的两个规律, elkan K-Means比起传统的K-Means迭代速度有很大的提高. 
但是如果我们的样本的特征是稀疏的, 有缺失值的话, 
这个方法就不使用了, 此时某些距离无法计算, 则不能使用该算法. 

# Mini Batch K-Means

在统的K-Means算法中, 要计算所有的样本点到所有的质心的距离. 
如果样本量非常大, 比如达到10万以上, 特征有100以上, 
此时用传统的K-Means算法非常的耗时, 就算加上elkan K-Means优化也依旧. 
在大数据时代, 这样的场景越来越多. 此时Mini Batch K-Means应运而生. 
顾名思义, Mini Batch, 也就是用样本集中的一部分的样本来做传统的K-Means, 
这样可以避免样本量太大时的计算难题, 算法收敛速度大大加快. 
当然此时的代价就是我们的聚类的精确度也会有一些降低. 
一般来说这个降低的幅度在可以接受的范围之内. 

在Mini Batch K-Means中, 我们会选择一个合适的批样本大小batch size, 
我们仅仅用batch size个样本来做K-Means聚类. 那么这batch size个样本怎么来的？
一般是通过无放回的随机采样得到的. 为了增加算法的准确性, 
我们一般会多跑几次Mini Batch K-Means算法, 
用得到不同的随机采样集来得到聚类簇, 选择其中最优的聚类簇. 

# PAM 聚类

PAM (Partitioning Around Medoids)

## 算法原理

PAM算法分为两个阶段: 

1. 第1阶段BUILD, 为初始集合S选择k个对象的集合；
2. 第2阶段SWAP, 尝试用未选择的对象, 交换选定的中心点, 来提高聚类的质量.  

PAM的工作原理: 

1. 初始化数据集, 选择k个对象作为中心；
2. 遍历数据点, 把每个数据点关联到最近中心点m；
3. 随机选择一个非中心对象, 与中心对象交换, 计算交换后的距离成本；
4. 如果总成本增加, 则撤销交换的动作；
5. 上面2-4步, 过程不断重复, 直到函数收敛, 中心不再改变为止；

## 算法优劣性

* PAM对噪声和异常值更稳健, 消除了k-means算法对于孤立点的敏感性；
* PAM支持混合的数据类型, 不仅限于连续变量；
* 比k-means的计算的复杂度要高. 
* 与k-means一样, 必须设置k的值. 
* 对小的数据集非常有效, 对大数据集效率不高

## 算法数学模型

## 算法伪代码

## 算法实现

### r 实现

函数解释: 

```r
pam(
    x,                                              # 数据框或矩阵, 允许有空值(NA)
    k,                                              # 设置分组数量
    diss = inherits(x, "dist"),                     # 为TRUE时, x为距离矩阵；为FALSE时, x是变量矩阵. 默认为FALSE
    metric = "euclidean",                           # 设置距离算法, 默认为euclidean, 距离矩阵忽略此项
    medoids = NULL,                                 # 指定初始的中心, 默认为不指定. 
    stand = FALSE,                                  # 为TRUE时进行标准化, 距离矩阵忽略此项. 
    cluster.only = FALSE,                           # 为TRUE时, 仅计算聚类结果, 默认为FALSE
    do.swap = TRUE,                                 # 是否进行中心点交换, 默认为TRUE；对于超大的数据集, 可以不进行交换. 
    keep.diss = !diss && !cluster.only && n < 100,  # 是否保存距离矩阵数据
    keep.data = !diss && !cluster.only,             # 是否保存原始数据
    pamonce = FALSE,                                # 一种加速算法, 接受值为TRUE,FALSE,0,1,2
    trace.lev = 0                                   # 日志打印, 默认为0, 不打印
)                                                   
```

聚类实现: 

```r
if(!require(cluster)) install.packages("cluster")

kclus = pam(x = df, k = 2)
kclus
kclus$clusinfo

plot(df, col = kclus$clustering, main = "K-medoids Cluster")
points(kclus$medoids, col = 1:3, pch = 10, cex = 4)
```

输出结果: 

* medoids, 中心点的数据值
* id.med, 中心点的索引
* clustering, 每个点的分组
* objective, 目标函数的局部最小值
* isolation, 孤立的聚类(用L或L*表示)
* clusinfo, 每个组的基本信息
* silinfo, 存储各观测所属的类、其邻居类以及轮宽(silhouette)值
* diss, 不相似度
* call, 执行函数和参数
* data, 原始数据集

### Python 实现

