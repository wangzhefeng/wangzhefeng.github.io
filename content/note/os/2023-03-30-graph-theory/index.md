---
title: 网络流与图
author: 王哲峰
date: '2023-03-30'
slug: graph-theory
categories:
  - optimizer algorithm
tags:
  - algorithm
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

- [图](#图)
  - [图介绍](#图介绍)
  - [图表示](#图表示)
    - [邻接矩阵](#邻接矩阵)
    - [邻接列表](#邻接列表)
</p></details><p></p>

# 图

## 图介绍

图(Graph)是图论的研究对象，图论是欧拉在研究哥尼斯堡七桥问题过程中，创造出来的新数学分支

网络(Graph / Network)视为一个系统，以 `$G(N, E)$` 表示，由两种元素组成：
顶点/节点(Vertex/Node)，以 `$N$` 表示，和边/链接(Edge/Link)，以 `$E$` 表示。
顶点和边具有属性(Attribute)，边可能有方向(有向图 Directed Graph)。
社交网络中，人是顶点，人和人之间的关系是边，人/顶点的属性比如年龄、性别、职业、爱好等构成了一个向量，类似的，边也可用向量来表示

![[img](https://sites.google.com/a/cs.christuniversity.in/discrete-mathematics-lectures/graphs/directed-and-undirected-graph)](images/graph.png)

* 微信好友是双向的，你我互为好友(不考虑拉黑/屏蔽)，对应于无向图(Undirected Graph)
* 微博的关注是单向的，我粉你，你未必粉我，对应有向图，也可以将无向图视为双向的有向图

## 图表示

图本身也具有表达其自身的全局属性，来描述整个图

### 邻接矩阵

如何用数学表示图中顶点的关系呢？最常见的方法是邻接矩阵(Adjacency Matrix)，
下图中 A 和 B、C、E 相连，故第一行和第一列对应的位置为 1，其余位置为 0

![img](images/adjacency_matrix.png)

如果将图片的像素表达为图，下左图表示图片的像素值，深色表示 1，浅色表示 0，
右图为该图片对应的图，中间为对应的邻接矩阵，蓝色表示 1，白色表示 0。
随着图的顶点数(`$n$`)增多，邻接矩阵矩阵的规模(`$n^{2}$`)迅速增大，
一张百万(`$10^{6}$`)像素的照片，
对应的邻接矩阵的大小就是(`$10^{6} \times 10^{6} = 10^{12}$`)，
计算时容易内存溢出，而且其中大多数值为 0，很稀疏

![img](images/graph_to_tensor_image.png)

文本也可以用邻接矩阵表示，但是问题也是类似的，很大很稀疏：

![img](images/graph_to_tensor_text.png)

### 邻接列表

也可以选用边来表示图，即邻接列表(Adjacency List)，这可以大幅减少对空间的消耗，因为实际的边比所有可能的边(邻接矩阵)数量往往小很多

![img](images/adjacency_list.png)

类似的例子有很多：

* CNN(局部连接)和全连接神经网络的关系；
* 大脑 860 亿个神经元，每个神经元大约与 1000 个神经元相连(而不是 860 亿个)
* 你真正保持联系的人并不太多，邓巴数告诉我们：一个人经常联系的人数大约 150 个，
  这是人类脑容量所决定的，不可能也没必要和 70 亿人都产生直接联系，小世界理论(6 度理论)又说，
  只要不超过 6 个人，你就可以连接上世界上的任何人。2016年，Facebook，不对，应该叫 Meta 了，
  研究发现社交网络使这个间隔降低到 4.57，这也可以理解，社交网络上可能有些你不太熟悉的人，
  你的微信好友大概率不止 150，但其中很多人联系并不多，联系的频率符合幂律分布，
  这是复杂系统的特点。随着 COVID-19 的载毒量下降，死亡率接近千分之一，
  疫苗接种普遍，蜕变为大号流感，国外的一个段子说，如果你的朋友圈没有人感染 COVID-19，
  那说明你没有朋友。社交网络和病毒传播路径均可以图来表示
