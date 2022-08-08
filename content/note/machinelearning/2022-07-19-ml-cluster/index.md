---
title: Cluster
author: 王哲峰
date: '2022-07-19'
slug: ml-cluster
categories:
  - machinelearning
tags:
  - model
  - ml
---

<style>
h1 {
  background-color: #2B90B6;
  background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
  background-size: 100%;
  -webkit-background-clip: text;
  -moz-background-clip: text;
  -webkit-text-fill-color: transparent;
  -moz-text-fill-color: transparent;
}
h2 {
  background-color: #2B90B6;
  background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
  background-size: 100%;
  -webkit-background-clip: text;
  -moz-background-clip: text;
  -webkit-text-fill-color: transparent;
  -moz-text-fill-color: transparent;
}

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

- [聚类](#聚类)
  - [聚类算法](#聚类算法)
  - [聚类数据](#聚类数据)
  - [聚类性能度量](#聚类性能度量)
  - [聚类距离计算](#聚类距离计算)
- [聚类算法介绍及实现](#聚类算法介绍及实现)
  - [基于原型的聚类](#基于原型的聚类)
    - [K-means](#k-means)
      - [算法介绍](#算法介绍)
        - [算法原理](#算法原理)
        - [算法优劣性](#算法优劣性)
        - [算法数学模型](#算法数学模型)
        - [算法伪代码](#算法伪代码)
      - [算法实现](#算法实现)
        - [r实现](#r实现)
        - [python实现](#python实现)
    - [K-Means++](#k-means-1)
    - [距离计算优化elkan K-Means](#距离计算优化elkan-k-means)
    - [Mini Batch K-Means](#mini-batch-k-means)
    - [PAM (Partitioning Around Medoids)聚类](#pam-partitioning-around-medoids聚类)
      - [算法介绍](#算法介绍-1)
        - [算法原理](#算法原理-1)
        - [算法优劣性](#算法优劣性-1)
        - [算法数学模型](#算法数学模型-1)
        - [算法伪代码](#算法伪代码-1)
      - [算法实现](#算法实现-1)
        - [r实现](#r实现-1)
        - [python实现](#python实现-1)
    - [学习向量量化聚类(Learning vector Quantization)](#学习向量量化聚类learning-vector-quantization)
      - [算法介绍](#算法介绍-2)
      - [算法实现](#算法实现-2)
    - [高斯混合聚类(Mixture-of-Gaussian)](#高斯混合聚类mixture-of-gaussian)
      - [算法介绍](#算法介绍-3)
        - [(多元)高斯分布](#多元高斯分布)
        - [(多元)高斯混合分布](#多元高斯混合分布)
        - [样本集的生成模型](#样本集的生成模型)
        - [高斯混合聚类策略](#高斯混合聚类策略)
        - [算法](#算法)
      - [算法实现](#算法实现-3)
  - [基于密度的聚类 (Density-based Clustering)](#基于密度的聚类-density-based-clustering)
    - [基于密度的聚类算法介绍](#基于密度的聚类算法介绍)
    - [DBSCAN (Density-Based Spatial Clustering of Application with Noise)](#dbscan-density-based-spatial-clustering-of-application-with-noise)
      - [算法介绍](#算法介绍-4)
        - [算法原理](#算法原理-2)
        - [算法优劣性](#算法优劣性-2)
        - [算法数学模型](#算法数学模型-2)
        - [算法伪代码](#算法伪代码-2)
      - [算法实现](#算法实现-4)
        - [R实现聚类](#r实现聚类)
        - [Python实现聚类](#python实现聚类)
    - [OPTICS (Ordering Points To Identify the Clustering Structure)](#optics-ordering-points-to-identify-the-clustering-structure)
      - [算法介绍](#算法介绍-5)
      - [算法实现](#算法实现-5)
  - [层次聚类(Hierarchical Clustering)](#层次聚类hierarchical-clustering)
    - [层次聚类算法介绍](#层次聚类算法介绍)
    - [AGNES(Agglomerative Nesting)层次聚类](#agnesagglomerative-nesting层次聚类)
      - [算法介绍](#算法介绍-6)
      - [算法实现](#算法实现-6)
</p></details><p></p>


# 聚类

* 聚类是从数据集中挖掘相似观测值集合的方法.
* 聚类试图将数据集中的样本划分为若干个通常是不相交的子集, 
  每个子集称为一个"簇"(cluster). 通过这样的划分, 
  每个簇可能对应于一些潜在的概念(类别). 
* 聚类过程仅能自动形成簇结构, 簇所对应的概念语义需由使用者自己来把握. 
* 聚类既能作为一个单独的过程用于寻找数据内在的分布结构, 
 也可以作为分类等其他学习任务的前驱过程. 

## 聚类算法

角度I:

* 基于原型的聚类(Prototype-based Clustering)
    - K均值聚类(K-means)
    - 学习向量量化聚类(Learning Vector Quantization)
    - 高斯混合模型聚类 (Gaussian Mixture Model)
* 基于密度的聚类 (Density-based Clustering)
    - DBSCAN (Density-Based Spatial Clustering of Application with Noise)
    - OPTICS (Ordering Points To Identify the Clustering Structure)
* 层次聚类 (Hierarchical Clustering)
* 基于模型的聚类 (Model-based Clustering)
    - 混合回归模型 (Mixture Regression Model)

[角度II](http://lchiffon.github.io/2014/12/28/cluster-analysis.html):

* 基于中心的聚类: kmeans聚类
* 基于分布的聚类: GMM聚类
* 基于密度的聚类: DBSCAN, OPTICS
* 基于连通性的聚类: 层次聚类
* 基于模型的聚类: Miture Regression Model
* 其他聚类方法: 谱聚类, Chameleon, Canopy...

## 聚类数据

> 假定样本集 `$D$` 包含 $n$ 个无标记样本:
> 
> $$D = \{x_1, x_2, \ldots, x_n\}$$
> 
> 每个样本是一个 `$p$` 维特征向量:
> 
> $$x_i=(x_{i1}; x_{i2}; \ldots; x_{ip})$$


> 聚类算法将样本集 `$D$` 划分为 `$k$` 个不相交的簇:
> 
> $$\{C_l|l=1, 2, \ldots, k\}$$
> 
> 其中: `$C_{l^{'}} \cap_{l^{'} \neq l} C_{l} = \emptyset$` 且 `$D=\cup_{l=1}^{k}C_{l}$`


> 相应的, 用
> 
> $$\lambda_{i} \in \{1, 2, \ldots, k\}$$
> 
> 表示样本 `$x_{i}$` 的"簇标记”(cluster label), 即
> 
> $$x_{i} \in C_{\lambda_{i}}$$

> 于是, 聚类的结果可用包含 `$n$` 个元素的簇标记向量表示
> 
> $$\lambda=(\lambda_{1}; \lambda_{2}, \ldots, \lambda_{n})$$


## 聚类性能度量

**聚类性能度量亦称聚类"有效性指标"(validity index).**

**设置聚类性能度量的目的:**

* 对聚类结果, 通过某种性能度量来评估其好坏；
* 若明确了最终将要使用的性能度量, 则可直接将其作为聚类过程的优化目标, 从而更好地得到符合要求的聚类结果. 

**什么样的聚类结果比较好？**

* "簇内相似度"(intra-cluster similarity)高
* "蔟间相似度"(inter-cluster similarity)低

**聚类性能度量分类:**

* "外部指标"(external index) : 将聚类结果与某个"参考模型"(reference model)进行比较
* "内部指标"(internal index): 直接考察聚类结果而不利用任何参考模型

**性能度量指标:**

**(1) 外部指标**

对数据集 `$D = \{x_1, x_2, \ldots, x_n\}$`, 
假定通过聚类给出的簇划分为 `$C=\{C_{1}, C_{2}, \ldots, C_{k}\}$`, 
参考模型给出的簇划分为 `$C^{*}=\{C_{1}^{*}, C_{2}^{*}\, \ldots, C_{k}^{*}\}$`. 
相应地, 令 `$\lambda$与$\lambda^{*}$` 分别表示与 `$C$` 和 `$C^{*}$` 对应的簇标记向量, 将样本两两配对考虑, 定义: 

$$a=|SS|, SS=\{(x_{i}, x_{j}) | \lambda_{i}=\lambda_{j}, \lambda_{i}^{*}=\lambda_{j}^{*}, i<j\}$$
$$b=|SD|, SD=\{(x_{i}, x_{j}) | \lambda_{i}=\lambda_{j}, \lambda_{i}^{*} \neq \lambda_{j}^{*}, i<j\}$$
$$c=|DS|, DS=\{(x_{i}, x_{j}) | \lambda_{i} \neq \lambda_{j}, \lambda_{i}^{*}=\lambda_{j}^{*}, i<j\}$$
$$d=|DD|, DD=\{(x_{i}, x_{j}) | \lambda_{i} \neq \lambda_{j}, \lambda_{i}^{*} \neq \lambda_{j}^{*}, i<j\}$$

其中: 

* 集合 `$SS$` 包含了在 `$C$` 中隶属于相同簇且在 `$C^{*}$` 中也隶属于相同簇的样本对；
* 集合 `$SD$` 包含了在 `$C$` 中隶属于相同簇但在 `$C^{*}$` 中隶属于不同簇的样本对；
* 集合 `$DS$` 包含了在 `$C$` 中隶属于不同簇但在 `$C^{*}$` 中隶属于相同簇的样本对；
* 集合 `$DD$` 包含了在 `$C$` 中隶属于不同簇且在 `$C^{*}$` 中也隶属于不同簇的样本对；

这样, 由于每个样本对 `$(x_{i}, x_{j})(i<j)$` 仅能出现在一个集合中, 因此有: 

$$a+b+c+d=n(n-1)/2$$

> * **Jaccard系数(Jaccard Coefficient 简称JC)**
> 
> $$JC=\frac{a}{a+b+c}$$
> 
> * **FM指数(Fowlkes and Mallows Index 简称JMI)**
> 
> $$FMI=\sqrt{\frac{a}{a+b} \cdot \frac{a}{a+c}}$$
> 
> * **Rand指数(Rand Index 简称RI)**
> 
> $$RI=\frac{2(a+d)}{n(n-1)}$$
> 
> **说明:**
> 
> 上述性能度量指标的结果值均在 `$[0,1]$` 区间, 值越大越好. 

**(2) 内部指标**

根据聚类结果的簇划分 `$C=\{C_{1}, C_{2}, \ldots, C_{k}\}$` ,定义: 

* 簇 `$C$` 内样本间的平均距离 

$$avg(C)=\frac{2}{|C|(|C|-1)}\sum_{1<i<j<|C|}dist(x_{i}, x_{j})$$

* 簇 `$C$` 内样本间的最远距离 

$$diam(C)=max_{1<i<j<|C|}dist(x_{i}, x_{j})$$

* 簇 `$C_{i}$` 与簇 `$C_{j}$` 最近样本间的距离 

$$d_{min}(C_{i}, C_{j})=min_{1<i<j<|C|}dist(x_{i}, x_{j}$$

* 簇 `$C_{i}$` 与簇 `$C_{j}$` 中心点间的距离 

$$d_{cen}(C_{i}, C_{j})=dist(\mu_{i}, \mu_{j})$$

其中: 

* `$dist(,)$` 是两个样本之间的距离
* `$\mu$` 是簇 `$C$` 的中心点 `$\mu=\frac{1}{|C|}\sum_{1<i<|C|}x_{i}$`

> * **DB指数(Davies-Bouldin Index 简称DBI)**
> 
> `$$DBI=\frac{1}{k}\sum^{k}_{i=1}\underset{j \neq i}{max}\bigg(\frac{avg(C_{i})+avg(C_{j})}{d_{cen(\mu_{i}, \mu_{j})]}}\bigg)$$`
> 
> * **Dunn指数(Dunn Index 简称DI)**
> 
> `$$DI=\underset{1 \leqslant i \leqslant k}{min}\bigg\{\underset{j \neq i}{min}\bigg(\frac{d_{min}(C_{i}, C_{j})}{max_{1 \leqslant l \leqslant k}diam(C_{l})}\bigg)\bigg\}$$`
> 
> **说明:**
> 
> * DBI的值越小越好
> * DI的值越大越好

## 聚类距离计算

**距离度量(distance measure)函数 `$dist(,)$` 需满足的基本性质:**

* **非负性**: `$dist(x_{i}, x_{j}) \geqslant 0$`
* **同一性**: `$dist(x_{i}, x_{j})=0$` 当且仅当 `$x_{i}=x_{j}$`
* **对称性**: `$dist(x_{i}, x_{j})=dist(x_{j}, x_{i})$`
* **直递性**: `$dist(x_{i}, x_{j}) \leqslant dist(x_{i}, x_{k}) + dist(x_{k}, x_{j})$` (可不满足)

**变量属性:**

* 连续属性: 闵可夫斯基距离
* 离散属性
    - 有序属性: 闵可夫斯基距离
    - 无序属性: VDM(Value Difference Metric)
* 混合属性: 闵可夫斯基距离与VDM混合距离

**(1) 闵可夫斯基距离(Minkowski distance)**

样本: 

$$x_{i}=(x_{i1}, x_{i2}, \ldots, x_{ip})$$
$$x_{j}=(x_{j1}, x_{j2}, \ldots, x_{jp})$$

* `$q \geqslant 1$`: **闵可夫斯基距离(Minkowski distance)**

$$dist_{mk}(x_{i}, x_{j})=\bigg(\sum_{u=1}^{p}|x_{iu}-x_{ju}|^{q}\bigg)^{\frac{1}{q}}, \quad q \geqslant 1$$

* `$q=1$` :**曼哈顿距离(Manhattan distance):**

$$dist_{man}(x_{i}, x_{j})=\|x_{i}-x_{j}\|_{1}=\sum^{p}_{u=1}|x_{iu}-x_{ju}|$$

* `$q=2$` :**欧氏距离(Euclidean distance):**

$$dist_{ed}(x_{i}, x_{j})=\|x_{i}-x_{j}\|_{2}=\sqrt{\sum^{p}_{u=1}|x_{iu}-x_{ju}|^{2}}$$


**(2) VDM(Value Difference Metric)**

令 `$m_{u,a}$` 表示在属性 `$u$` 上取值为 `$a$` 的样本数, 
`$m_{u, a, i}$` 表示在第 `$i$` 个样本簇中在属性 `$u$` 上取值为 `$a$` 的样本数, `$k$` 为样本簇数, 则属性 `$u$` 上两个离散值 `$a$` 与 $b$ 之间的VDM距离为: 

$$VDM_{q}(a, b)=\sum^{k}_{i=1}\bigg|\frac{m_{u,a,i}}{m_{u,a}}-\frac{m_{u,b,i}}{m_{u,b}}\bigg|^{q}$$


**(3) 闵可夫斯基距离与VDM混合距离**

假设有 `$p_{c}$` 个有序属性, `$p-p_{c}$` 个无序属性, 有序属性排列在无序属性之前: 

$$MinkovDM_{q}(x_{i}, x_{j})=\bigg(\sum^{p_{c}}_{u=1}|x_{i,u}-x_{j,u}|^{q}+\sum^{p}_{u=p_{c}+1}VDM_{q}(x_{i,u},x_{j,u})\bigg)^{\frac{1}{q}}$$

**(4) 加权闵可夫斯基距离**

当样本在空间中不同属性的重要性不同时: 

$$dist_{wmk}(x_{i}, x_{j})=(w_{1}\cdot|x_{i1}-x_{j1}|^{q}+w_{2}\cdot|x_{i2}-x_{j2}|^{q}+\ldots+w_{n}\cdot|x_{in}-x_{jn}|^{q})^{\frac{1}{q}}$$

其中: 权重 `$w_{i}\geqslant 0(i=1, 2, \ldots, p)$` 表示不同属性的重要性, 通常 `$\sum_{i=1}^{n}w_{i}=1$`.


# 聚类算法介绍及实现

**聚类算法类型:**

* **基于原型的聚类(Prototype-based Clustering)**
    - **K均值聚类(K-means )**
    - **学习向量量化聚类(Learning vector Quantization)**
    - **高斯混合聚类(Mixture-of-Gaussian)**
* **基于密度的聚类(Density-based Clustering)**
* **层次聚类(Hierarchical Clustering)**


## 基于原型的聚类

**基于原型的聚类(Prototype-based Clustering), 此类算法假设聚类结构能通过一组原形刻画. 通常情况下, 算法先对原型进行初始化, 然后对原型进行迭代更新求解, 采用不同的原型表示, 不同的求解方式,将产生不同的算法.**

* **基于原型的聚类(Prototype-based Clustering)**
    - **K均值聚类(K-means )**
    - **学习向量量化聚类(Learning vector Quantization)**
    - **高斯混合聚类(Mixture-of-Gaussian)**


### K-means

#### 算法介绍

##### 算法原理

1. 初始化数据, 选择`$k$`个对象作为中心点, 对于`$k$`的选择, 需要经过交叉验证等方法进行选取；
2. 遍历整个数据集, 计算每个点与每个中心点的距离, 将它们分配给距离中心最近的组；
3. 重新计算每个组的平均值, 作为新的聚类中心；
4. 重复上面的2-3步, 直到函数收敛, 不再有新的分组情况出现. 


##### 算法优劣性

优点: 

> 原理比较简单, 实现容易, 收敛速度快；
> 聚类效果比较好；
> 算法的可解释性较强；
> 主要的参数仅有一个`$k$`；

缺点: 

> * K-means只适用于连续型数据集；
> * `$k$`值不容易选取；
> * 如果各隐含类别的数据不平衡, 比如各隐含类别的数据量严重失衡, 或者各隐含类别的方差不同, 则聚类效果不佳；
> * 采用迭代方法, 得到的结果只是局部最优；
> * K-means对数据异常点(极端值)比较敏感；


##### 算法数学模型

给定样本集 `$D=\{x_{1}, x_{2}, \ldots, x_{n}\}$`, K-means 算法针对聚类所得簇划分`$C=\{C_{1}, C_{2}, \ldots, C_{k}\}$`, 最小化平方误差:

$$E = \sum^{k}_{i=1}\sum_{x \in C_{i}}||x-\mu_{i}||_{2}^{2}$$
其中 `$\mu_{i}$` 是簇 `$C_{i}$` 的均值向量: 
$$\mu_{i}=\frac{1}{|C_{i}|}\sum_{x \in C_{i}}x$$

直观上看, 平方误差在一定程度上刻画了簇内样本围绕均值向量的紧密程度, $E$ 值越小簇内样本相似度越高. 但最小化 $E$ 不容易, 是一个NP难问题, K-means 算法采用了贪心策略, 通过迭代优化来近似求解 $E$ 的最小值. 具体算法如下. 


##### 算法伪代码

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


#### 算法实现


##### r实现

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

##### python实现



### K-Means++


1. 从输入的数据集合中随机选择一个点作为第一个聚类中心 `$\mu_1$`
2. 对于数据集中的每一个点 `$x_i$`, 计算它与已选择的聚类中心的距离, 最小的距离
$$D(x)=arg min \sum_{i=1}^{k}||x_i-\mu_i||_{2}^{2}$$
3. 选择一个新的数据点作为新的聚类中心, 选择的原则是: `$D(x)$` 较大的点, 被选做聚类中心的概率较大；
4. 重复2-3直到选择出 `$k$`个聚类中心；
5. 利用这`$k$`个质心去初始化质心, 然后运行标准的K-Means算法；



### 距离计算优化elkan K-Means


在传统的K-Means算法中, 在每轮迭代时, 要计算所有的样本点到所有的质心的距离, 
这样会比较的耗时. 那么, 对于距离的计算有没有能够简化的地方呢？
elkan K-Means算法就是从这块入手加以改进. 
它的目标是减少不必要的距离的计算. 那么哪些距离不需要计算呢？
elkan K-Means利用了两边之和大于等于第三边,以及两边之差小于第三边的三角形性质, 来减少距离的计算. 

* 第一种规律是对于一个样本点 `$x$` 和两个质心 `$\mu_i$,$\mu_j$`. 
  如果我们预先计算出了这两个质心之间的距离 `$D(i,j)$`, 
  则如果计算发现 `$2D(x,j)≤D(i,j)$`, 我们立即就可以知道 `$D(x,i)≤D(x,j)$`. 
  此时我们不需要再计算 `$D(x,j)$`, 也就是说省了一步距离计算. 
* 第二种规律是对于一个样本点 `$x$` 和两个质心 `$\mu_i, \mu_j$`. 
  我们可以得到 `$D(x,j)≥max{0,D(x,i)−D(i,j)}$`. 这个从三角形的性质也很容易得到. 


利用上边的两个规律, elkan K-Means比起传统的K-Means迭代速度有很大的提高. 
但是如果我们的样本的特征是稀疏的, 有缺失值的话, 
这个方法就不使用了, 此时某些距离无法计算, 则不能使用该算法. 


### Mini Batch K-Means

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

### PAM (Partitioning Around Medoids)聚类

#### 算法介绍

##### 算法原理

PAM算法分为两个阶段: 

1. 第1阶段BUILD, 为初始集合S选择k个对象的集合；
2. 第2阶段SWAP, 尝试用未选择的对象, 交换选定的中心点, 来提高聚类的质量.  

PAM的工作原理: 

1. 初始化数据集, 选择k个对象作为中心；
2. 遍历数据点, 把每个数据点关联到最近中心点m；
3. 随机选择一个非中心对象, 与中心对象交换, 计算交换后的距离成本；
4. 如果总成本增加, 则撤销交换的动作；
5. 上面2-4步, 过程不断重复, 直到函数收敛, 中心不再改变为止；

##### 算法优劣性

> * PAM对噪声和异常值更稳健, 消除了k-means算法对于孤立点的敏感性；
> * PAM支持混合的数据类型, 不仅限于连续变量；
> * 比k-means的计算的复杂度要高. 
> * 与k-means一样, 必须设置k的值. 
> * 对小的数据集非常有效, 对大数据集效率不高

##### 算法数学模型

##### 算法伪代码

#### 算法实现

##### r实现

函数解释: 

```r
pam(x,                                              # 数据框或矩阵, 允许有空值(NA)
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
    trace.lev = 0)                                  # 日志打印, 默认为0, 不打印
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



##### python实现





### 学习向量量化聚类(Learning vector Quantization)

#### 算法介绍

LVQ 假设数据样本带有类别标记, 学习过程利用样本的这些监督信息来辅助聚类. 

给定样本集 `$D=\{(x_{1}, y_{1}), (x_{2}, y_{2}), \ldots, (x_{n}, y_{n})\}$`, 
`$x_{i} \in R^{p}$, $y_{i} \in \mathcal{Y}$` 是样本的类别标记. 
LVQ的目标是学得一组 `$n$` 维原型向量 `$\{p_{1}, p_{2}, \ldots, p_{q}\}$`, 
每个原型向量代表一个聚类簇, 簇标记为 `$t_{i}\in \mathcal{Y}$`.

**算法**

> **输入:**
> 
> 样本集: `$D=\{(x_{1}, y_{1}), (x_{2}, y_{2}), \ldots, (x_{n}, y_{n})\}$`;
>
> 原型向量个数 `$q$`, 各原型向量预设的类标记 `$\{t_{1}, t_{2}, \ldots, t_{q}\}$`;
>
> 学习率 `$\eta \in (0,1)$`.
> 
> **过程:**
> 
> 1. 初始化一组原型向量: `$\{p_{1}, p_{2},\ldots,p_{q}\}$`
> 2. **repeat**
> 3. 从样本集中随机选取样本 `$(x_{j}, y_{j})$`;
> 4. 计算样本 `$x_{j}$` 与 `$p_{i} (1 \leqslant i \leqslant q)$` 的距离: `$d_{ji}=\|x_{j}-p_{i}\|_{2}$`;
> 5. 找出与 `$x_{j}$` 距离最近的原型向量 `$p_{i^{*}}$`, `$i^{*}=arg min_{i \in \{1, 2, \ldots, q\}}d_{ji}$`;
> 6. **if** `$y_{j}=t_{i^{*}}$`
> 7. `$p' = p_{i^{*}} + \eta \cdot (x_{j}-p_{i^{*}})$`
> 8. **else**
> 9. `$p' = p_{i^{*}} - \eta \cdot (x_{j}-p_{i^{*}})$`
> 10. **end if**
> 11. 将原型向量 `$p_{i^{*}}$` 更新为 `$p'$`
> 12. **until** 满足停止条件
> 
> **输出:**
> 
> 原型向量 `$\{p_{1}, p_{2}, \ldots, p_{q}\}$`,

**算法解释**

* 算法第1行: 对原型向量进行初始化. 例如: 对第 `$i,i=(1,2,\ldots,q)$` 个簇,从类别标记为 `$t_{i}$` 的样本中随机选取一个作为原型向量. 
* 算法第2-12行: 对原型向量进行迭代优化. 在每一轮迭代中, 算法随机选取一个有标记训练样本, 
  找出与其距离最近的原型向量, 并根据两者的类别标记是否一致来对原型向量进行相应的更新. 
    - 第6-10行: 如何更新原型向量. 对样本 `$x_{j}$`,
        - 若距离`$x_{j}$`最近的原型向量 `$p_{i^{*}}$` 与 `$x_{j}$` 的标记相同, 
          则令 `$p_{i^{*}}$` 向 `$x_{j}$` 的方向靠拢, 此时新的原型向量为
          $$p' = p_{i^{*}} + \eta \cdot (x_{j}-p_{i^{*}})$$
          `$p^{'}$` 与 `$x_{j}$` 之间的距离为
          $$\|p'-x_{j}\|_{2}=(1-\eta) \cdot \|p_{i^{*}}-x_{j}\|_{2}$$
          原型向量 `$p_{i^{*}}$` 更新为 `$p'$` 之后将更接近 `$x_{j}$`.
        - 若距离`$x_{j}$`最近的原型向量 `$p_{i^{*}}$` 与 `$x_{j}$` 的标记不同, 
          则令 `$p_{i^{*}}$` 向 `$x_{j}$` 的方向远离, 此时新的原型向量为 
          $$p' = p_{i^{*}} - \eta \cdot (x_{j}-p_{i^{*}})$$
          `$p^{'}$` 与 `$x_{j}$` 之间的距离为
          $$\|p'-x_{j}\|_{2}=(1+\eta) \cdot \|p_{i^{*}}-x_{j}\|_{2}$$
          原型向量 `$p_{i^{*}}$` 更新为 `$p'$`之后将更远离 `$x_{j}$`.
* 算法第12行: 若算法的停止条件已满足(例如已达到最大迭代轮数, 或原型向量更新很小甚至不再更新), 则将当前原型向量作为最终结果返回. 
* 在学得一组原型向量 `$\{p_{1}, p_{2}, \ldots, p_{q}\}$` 后即可实现对样本空间 `$\mathcal{X}$` 的簇划分. 
    - 对任意样本 `$x$`, 他将被划入与其距离最近的原型向量所代表的簇中, 
      每个原型向量 `$p_{i}$` 定义了与之相关的一个区域 `$R_{i}$`, 
      该区域中每个样本与 `$p_{i}$` 的距离不大于他与其他原型向量 `$p_{i'} (i \neq i')$`, 即
      $$R_{i}=\{x \in \mathcal{X}|\|x-p_{i}\|_{2} \leqslant \|x-p_{i'}\|_{2}, i \neq i'\}$$
      由此形成了对样本空间 `$\mathcal{X}$` 的簇划分 `$\{R_{1}, R_{2}, \ldots, R_{q}\}$`, 该划分通常称为"Voronoi剖分"(Voronoi tessellation).


#### 算法实现


### 高斯混合聚类(Mixture-of-Gaussian)


#### 算法介绍

高斯混合聚类(Mixture-of-Gaussian)采用概率模型来表达聚类原型. 

##### (多元)高斯分布

对 `$n$` 维样本空间 `$\mathcal{X}$` 中的随机向量 `$x$`, 若 `$x$` 服从(多元)高斯分布, 其概率密度函数为: 

$$p(x| \mu, \Sigma)=\frac{1}{(2\pi)^{\frac{n}{2}}|\Sigma|^{\frac{1}{2}}} e^{ - \frac{1}{2} (x-\mu)^{T} \Sigma^{-1} (x-\mu)}$$

其中: `$\mu$` 是 `$n$` 维均值向量, `$\Sigma$` 是 `$n \times n$` 协方差矩阵.

##### (多元)高斯混合分布

对 `$n$` 维样本空间 `$\mathcal{X}$` 中的随机向量 `$x$`, 若 `$x$` 服从(多元)高斯混合分布, 其概率密度函数为: 

$$p_{\mathcal{M}}(x)=\sum_{i=1}^{k}\alpha_{i} \cdot p(x| \mu_{i}, \Sigma_{i})$$

该分布由 `$k$` 个混合成分组成, 每个成分对应一个(多元)高斯分布,

其中: `$\mu_{i}$`, `$\Sigma_{i}$` 是第 `$i$` 个高斯混合成分的参数, 
而 `$\alpha_{i}$` 为相应的"混合系数"(mixture coeffcient), 
`$\sum_{i=1}^{k}\alpha_{i}=1$`.

##### 样本集的生成模型

假设样本集 ``$D=\{x_{1}, x_{2}, \ldots, x_{n}\}$`` 的生成过程有高斯混合分布给出: 

* 首先: 根据 `$\alpha_{1}, \alpha_{2}, \ldots,\alpha_{k}$` 定义的先验分布选择高斯混合成分, 其中 `$\alpha_{i}$` 为选择第 `$i$` 个混合成分的概率;
* 然后: 根据被选择的混合成分的概率密度函数进行采样, 生成相应的样本.

令随机变量 `$z_{j} \in \{1, 2, \ldots,k\}$` 表示生成样本 `$x_{j}$` 的高斯混合成分, 其取值未知. 

`$z_{j}$` 的先验概率: 

$$P(z_{j}=i)=\alpha_{i} \quad (i=1, 2, \ldots,k)$$. 

由Bayesian定理得 `$z_{j}$` 的后验分布为: 

$$\begin{align*}
P_{\mathcal{M}}(z_{j}=i|x_{j}) &=\frac{P(z_{j}=i) \cdot p_{\mathcal{M}}(x_{j}|z_{j}=i)}{p_{\mathcal{M}}(x_{j})}\\
                               &=\frac{\alpha_{i}\cdot p(x_{j}|\mu_{i},\Sigma_{i})}{\sum^{k}_{l=1}\alpha_{l}\cdot p(x_{j}|\mu_{l},\Sigma_{l})}
\end{align*}$$

$P_{\mathcal{M}}(z_{j}=i|x_{j})$ 给出了样本 `$x_{j}$` 由第 `$i$` 个高斯混合成分生成的后验概率, 记: 

$$\gamma_{ji}=P_{\mathcal{M}}(z_{j}=i|x_{j}) \quad (i=1, 2, \ldots,k)$$


##### 高斯混合聚类策略

* 若(多元)高斯混合分布 `$p_{\mathcal{M}}(x)$` 已知, 高斯混合聚类将把样本集 `$D$` 划分为 `$k$` 个簇 

$$C=\{C_{1}, C_{2}, \ldots,C_{k}\}$$

每个样本 `$x_{j}$` 的簇标记 `$\lambda_{j}$` 为:

$$\lambda_{j}=arg \underset{i \in \{1, 2, \ldots,k\}}{max}\gamma_{ji}$$


* (多元)高斯混合分布 `$p_{\mathcal{M}}(x)$` 参数 `$\{(\alpha_{i},\mu_{i}, \Sigma_{i})|1 \leqslant i \leqslant k\}$` 的求解采用极大似然估计(MLE):

给定样本集 `$D$`, 最大化(对数)似然函数: 

$$\begin{align*}
LL(D)&=ln\Big(\prod^{n}_{j=1}p_{\mathcal{M}}(x_{j})\Big)\\
&=\sum^{n}_{j=1}\Big(\sum^{k}_{i=1}\alpha_{i}\cdot p(x_{j}|\mu_{i}, \Sigma_{i})\Big)
\end{align*}$$

MLE解为: 

$$\mu_{i}=\frac{\sum^{n}_{j=1}\gamma_{ji}x_{j}}{\sum^{n}_{j=1}\gamma_{ji}}$$
$$\Sigma_{i}=\frac{\sum^{n}_{j=1}\gamma_{ji}(x_{j}-\mu_{i})(x_{j}-\mu_{i})^{T}}{\sum^{n}_{j=1}\gamma_{ji}}$$
$$\alpha_{i}=\frac{1}{n}\sum^{n}_{j=1}\gamma_{ji}$$


##### 算法

> **输入:**
> 
> 样本集: ``$D=\{x_{1}, x_{2}, \ldots, x_{n}\}$``;
>
> 高斯混合成分个数 `$k$`.
> 
> **过程:**
> 
> 1. 初始化高斯混合分布的模型参数 : `$\{(\alpha_{i},\mu_{i}, \Sigma_{i})|1 \leqslant i \leqslant k\}$`
> 2. **repeat**
> 3. **for** `$j=1, 2, \ldots, n$` **do**
> 4. 根据()计算 `$x_{j}$` 由各混合成分生成的后验概率, 即 `$\gamma_{ji}=p_{\mathcal{M}}(z_{j}=i|x_{j})(1 \leqslant i \leqslant k)$`
> 5. **end for** 
> 6. **for** `$i=1, 2, \ldots, k$` **do**
> 7. 计算新均值向量: `$\mu_{i}'=\frac{\sum^{n}_{j=1}\gamma_{ji}x_{j}}{\sum^{n}_{j=1}\gamma_{ji}}$`;
> 8. 计算新协方差矩阵: `$\Sigma_{i}'=\frac{\sum^{n}_{j=1}\gamma_{ji}(x_{j}-\mu_{i}')(x_{j}-\mu_{i}')^{T}}{\sum^{n}_{j=1}\gamma_{ji}}$`;
> 9. 计算新混合系数: `$\alpha_{i}'=\frac{\sum^{n}_{j=1}\gamma_{ji}}{n}$`;
> 10. **end for**
> 11. 将模型参数 `$\{(\alpha_{i},\mu_{i}, \Sigma_{i})|1 \leqslant i \leqslant k\}$` 更新为 `$\{(\alpha_{i}',\mu_{i}', \Sigma_{i}')|1 \leqslant i \leqslant k\}$`
> 12. **until** 满足停止条
> 13. `$C_{i}=\emptyset (1 \leqslant i \leqslant k)$`
> 14. **for** `$j=1, 2, \ldots, n$` **do**
> 15. 根据()确定 `$x_{j}$` 的簇标记 `$\lambda_{j}$`;
> 16. 将 `$x_{j}$` 划入相应的簇: `$C_{\lambda_{j}}=C_{\lambda_{j}} \cup \{x_{j}\}$`
> 17. **end for**
> 
> **输出:**
> 
> 簇划分 `$C=\{C_{1}, C_{2}, \ldots, C_{k}\}$`




#### 算法实现



## 基于密度的聚类 (Density-based Clustering)

### 基于密度的聚类算法介绍

基于密度的聚类(Density-based Clustering)假设聚类结构能通过样本分布的紧密程度确定.密度聚类算法从样本密度的角度来考察样本之间的可连续性, 并基于可连续样本不断扩展聚类簇以获得最终的聚类结果.


* **基于密度的聚类(Density-based Clustering)**
    - **[DBSCAN](https://en.wikipedia.org/wiki/DBSCAN)** (Density-Based Spatial Clustering of Application with Noise)
    - **[OPTICS](https://en.wikipedia.org/wiki/OPTICS_algorithm)** (Ordering Points To Identify the Clustering Structure)


### DBSCAN (Density-Based Spatial Clustering of Application with Noise)

#### 算法介绍


##### 算法原理

> * DBSCAN (Density-Based Spatial Clustering of Application with Noise): 基于一组"邻域 (neighborhood)"参数 $(\epsilon, MinPts)$ 来刻画样本分布的紧密程度.
> * DBSCAN有两个重要参数: 
>    - `$\epsilon$`: `$\epsilon$` 定义了点 `$x$` 附近的领域半径, 被称为 `$x$` 的最邻居；
>    - `$MinPts$`: `$MinPts$` 是 `$\epsilon$` 半径内的最小邻居数；
> * DBSCAN算法将数据点分为三类: 
>    - 核心点: 在半径 `$\epsilon$` 内含有超过 `$MinPts$` 数目的点. 
>    - 边界点: 在半径 `$\epsilon$` 内点的数量小于使用DBSCAN进行聚类的时候, 不需要预先指定簇的个数, 最终的簇的个数不确定. minPts`$MinPts$`,但是落在核心点的邻域内的点. 
>    - 噪音点: 既不是核心点也不是边界点的点



##### 算法优劣性

* 不需要预先确定簇的数量；
* 能将异常点识别为噪声数据；
* 能够很好地找到任意大小和形状的簇；
* 当数据簇密度不均匀时, 效果不如其他算法好, 这是因为当密度变化时, 用于识别邻近点的距离阈值ε和minPoints的设置将随着簇而变化. 在处理高维数据时也会出现这种缺点, 因为难以估计距离阈值eps；


##### 算法数学模型

给定数据集 ``$D=\{x_{1}, x_{2}, \ldots, x_{n}\}$``,  定义: 

* **`$\epsilon$`-邻域**
    - 对 `$x_{j}\in D$`, 其 `$\epsilon$`-邻域包含样本集 `$D$` 中与 `$x_{j}$` 的距离不大于 `$\epsilon$` 的样本, 即 `$N_{\epsilon}(x_{j})=\{x_{i} \in D|dist(x_{i},x_{j})\leqslant \epsilon\}$`；
* **核心对象 (core object)**
    - 若 `$x_{j}$` 的 `$\epsilon$`-邻域至少包含 `$MinPts$` 个样本, 即 `$|N_{\epsilon}(x_{j})|\geqslant MinPts$`, 则 `$x_{j}$` 是一个核心对象；
* **密度直达 (directly density-reachable)**
    - 若 `$x_{j}$` 位于 `$x_{i}$` 的 `$\epsilon$`-邻域中, 且 `$x_{i}$` 是核心对象, 则称 `$x_{j}$` 由 `$x_{i}$` 密度直达；
* **密度可达 (density-reachable)**
    - 对 `$x_{i}$` 与 `$x_{j}$`, 若存在样本序列 `$p_{1}, p_{2}, \ldots, p_{n}$`, 其中 `$p_{1}=x_{i}$`, `$p_{n}=x_{j}$` 且 `$p_{i+1}$` 由 `$p_{i}$` 密度直达, 则称 `$x_{j}$` 由 `$x_{i}$` 密度可达； 
* **密度相连(density-connected)**
    - 对 `$x_{i}$` 与 `$x_{j}$`, 若存在 `$x_{k}$` 使得 `$x_{i}$` 与 `$x_{j}$` 均由 `$x_{k}$` 密度可达, 则称`$x_{i}$` 与 `$x_{j}$` 密度相连.

基于这些概念, DBSCAN 将"簇”定义为: 由密度可达关系导出的最大的密度相连样本集合. 形式化的说: 给定邻域参数 `$(c, MinPts)$`, 簇 `$C \subseteq D$` 是满足以下性质的非空样本子集: 

* **连接性 (connectivity)**
    - `$x_{i} \in C$`, `$x_{j} \in C$ $\Longrightarrow$` `$x_{i}$` 与 `$x_{j}$` 密度相连 (一个聚类簇内的所有点都是密度相连的)
* **最大性 (maximality)**
    - `$x_{i} \in C$`, `$x_{j}$` 由 `$x_{i}$` 密度可达 `$\Longrightarrow$ $x_{j} \in C$` (如果一个点对于聚类簇中每个点都是密度可达的,那这个点就是这个类中的一个)
    

**从数据集 `$D$` 中找出满足以上性质的聚类簇:**

若 `$x$` 为核心对象, 由 `$x$` 密度可达的所有样本组成的集合记为 `$X=\{x' \in D | x'由x密度可达\}$`, 则不难证明 `$X$` 即为满足连续性与最大性的簇.


##### 算法伪代码

> **输入:**
> 
> 样本集: ``$D=\{x_{1}, x_{2}, \ldots, x_{n}\}$``;
>
> 领域参数 `$(c, MinPts)$`.
> 
> **过程:**
> 
> 1. 初始化核心对象集合: `$\Omega=\emptyset$`
> 2. **for** `$j=1, 2, \ldots, n$` **do**
> 3. 确定样本 `$x_{j}$` 的 `$\epsilon$`-领域 `$N_{\epsilon}(x_{j})$`
> 4. **if** `$|N_{\epsilon}(x_{j})|\geqslant MinPts$` **then**
> 5. 将样本 `$x_{j}$` 加入核心对象集合: `$\Omega=\Omega\cup\{x_{j}\}$`
> 6. **end if**
> 7. **end for**
> 8. 初始化聚类簇数: `$k=0$`
> 9. 初始化未访问样本集合 `$\Gamma=D$`
> 10. **while** `$\Omega\neq\emptyset$` **do**
> 11. 记录当前未访问样本集合 `$\Gamma_{old}=\Gamma$`;
> 12. 随机选取一个核心对象 `$o\in\Omega$`,初始化队列 `$Q=<o>$`;
> 13. `$\Gamma=\Gamma \setminus \{o\}$`
> 14. **while** `$Q\neq\emptyset$` **do**
> 15. 取出队列 `$Q$` 中的首个样本 `$q$`;
> 16. **if** `$|N_{\epsilon}(q)|\geqslant MinPts$` **then**
> 17. 令 `$\Delta=N_{\epsilon}(q) \cap \Gamma$`;
> 18. 将 `$\Delta$` 中的样本加入队列 `$Q$`;
> 19. `$\Gamma=\Gamma \setminus \Delta$`;
> 20. **end if**
> 21. **end while**
> 22. `$k=k+1$`, 生成聚类簇 `$C_{k}=\Gamma_{old} \setminus \Gamma$`;
> 23. `$\Omega=\Omega \setminus C_{k}$`
> 24. **end while**
> 
> **输出:**
> 
> 簇划分 `$C=\{C_{1}, C_{2}, \ldots, C_{k}\}$`

**算法说明:**

* DBSCAN算法先任选数据集中的一个核心对象为 "种子 (seed)",再由此出发确定相应的聚类簇；
* 第1-7行: 先根据给定的邻域簇 `$(c, MinPts)$` 找出所有核心对象；
* 第10-24行: 以任一核心对象为出发点, 找出由其密度可达的样本生成聚类簇, 直到所有核心对象均被访问过为止.

#### 算法实现

> * R中提供了`dbscan`包, `dbscan`底层使用C++编程, 并建立kd树的数据结构进行更快的K最近邻搜索, 从而实现加速. 该包提供了基于密度的有噪声聚类算法的快速实现, 包括: 
      - DBSCAN(基于密度的具有噪声的应用的空间聚类)
      - OPTICS(用于识别聚类结构的排序点)
      - HDBSCAN(分层DBSCAN)
      - LOF(局部异常因子算法)
  * Python中提供了

##### R实现聚类

函数列表: 

* dbscan(), 实现DBSCAN算法
* optics(), 实现OPTICS算法
* hdbscan(), 实现带层次DBSCAN算法
* sNNclust(), 实现共享聚类算法
* jpclust(), Jarvis-Patrick聚类算法
* lof(), 局部异常因子得分算法
* extractFOSC(),集群优选框架, 可以通过参数化来执行聚类. 
* frNN(), 找到固定半径最近的邻居
* kNN(), 最近邻算法, 找到最近的k个邻居
* sNN(), 找到最近的共享邻居数量
* pointdensity(), 计算每个数据点的局部密度
* kNNdist(), 计算最近的k个邻居的距离
* kNNdistplot(), 画图, 最近距离
* hullplot(), 画图, 集群的凸壳

```r
if(!require(dbscan)) install.packages("dbscan")
```

##### Python实现聚类


### OPTICS (Ordering Points To Identify the Clustering Structure)

OPTICS和DBSCAN聚类相同,都是基于密度的聚类,但是,OPTICS的好处在于可以处理不同密度的类,结果有点像基于连通性的聚类,不过还是有些区别的. 上段伪代码:

#### 算法介绍


#### 算法实现



## 层次聚类(Hierarchical Clustering)

### 层次聚类算法介绍

层次聚类(Hierarchical Clustering) 也称为基于连通性的聚类. 这种算法试图在不同层次对数据进行划分, 从而形成树形的聚类结构. 

数据集的划分采用不同的策略会生成不同的层次聚类算法: 

* "自底向上"的聚合策略
    - AGNES(Agglomerative Nesting)
* "自顶向下"的分拆策略

### AGNES(Agglomerative Nesting)层次聚类

#### 算法介绍

AGNES(Agglomerative Nesting)是一种采用自底向上聚合策略的层次聚类算法, 算法的步骤为: 

1. 先将数据集中的每个样本当做是一个初始聚类簇;
2. 然后在算法运行的每一步中找出距离最近的两个点(聚类簇)进行合并为一个聚类簇;
3. 上述过程不断重复, 直至所有的样本点合并为一个聚类簇或达到预设的聚类簇个数.  最终算法会产生一棵树, 称为树状图(dendrogram), 树状图展示了数据点是如何合并的.

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
$$d_{max}(C_{i}, C_{j})=\underset{x\in C_{i},y\in C_{j}}{max}dist(x, y)$$
    - **Single-linkage clustering(单链接)**: Find the minimum possible distance between points belonging to two different clusters.
$$d_{min}(C_{i}, C_{j})=\underset{x\in C_{i},y\in C_{j}}{min}dist(x, y)$$
    - **Mean-linkage clustering(平均链接 UPGMA)**: Find all possible pairwise distances for points belonging to two different clusters and then calculate the average.
$$d_{avg}(C_{i}, C_{j})=\frac{1}{|C_{i}||C_{j}|}\sum_{x\in C_{i}}\sum_{y\in C_{j}}dist(x, y)$$
    - **Centroid-linkage clustering(中心链接 UPGMC)**: Find the centroid of each cluster and calculate the distance between centroids of two clusters.
$$d_{cen}(C_{i}, C_{j})=dist(x, y),\quad x,y分别是C_{i},C_{j}的中心$$
    - **Minimum energy clustering**
$$d_{ene}(C_{i}, C_{j})=\frac{2}{nm}\sum_{i=1}^{n}\sum_{j=1}^{m}\|x_{i}- y_{j}\|_{2} - \frac{1}{n^2}\sum_{i=1}^{n}\sum_{j=1}^{n}\|x_{i}-x_{j}\|_{2}-\frac{1}{m^2}\sum_{i=1}^{m}\sum_{j=1}^{m}\|y_{i}-y_{j}\|_{2}$$

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


#### 算法实现

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

