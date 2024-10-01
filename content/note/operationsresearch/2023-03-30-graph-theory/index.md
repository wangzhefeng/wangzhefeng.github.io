---
title: 图与网络分析
subtitle: Graph and Network
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

- [图的基本概念](#图的基本概念)
    - [无向图和有向图](#无向图和有向图)
    - [图表示方法](#图表示方法)
- [图的矩阵表示](#图的矩阵表示)
    - [邻接矩阵](#邻接矩阵)
    - [旅行商问题](#旅行商问题)
    - [邻接列表](#邻接列表)
- [最小生成树](#最小生成树)
- [图问题](#图问题)
    - [最短路径问题](#最短路径问题)
    - [网络最大流问题](#网络最大流问题)
    - [路径规划](#路径规划)
    - [车辆路径问题](#车辆路径问题)
</p></details><p></p>

# 图的基本概念

图(Graph)是图论的研究对象，图论是欧拉在研究哥尼斯堡七桥问题过程中，创造出来的新数学分支。

## 无向图和有向图

没有方向区分的子图称为无向图，如交通网络；有方向区分的子图为有向图，流程任务图就是很经典的有向图。
例如：微信好友是双向的，你我互为好友(不考虑拉黑/屏蔽)，对应于无向图(Undirected Graph)；
微博的关注是单向的，我粉你，你未必粉我，对应有向图，也可以将无向图视为双向的有向图。

![[img](https://sites.google.com/a/cs.christuniversity.in/discrete-mathematics-lectures/graphs/directed-and-undirected-graph)](images/graph.png)

## 图表示方法

图/网络(Graph / Network)视为一个系统，以 `$G(V, E)$` 表示，由两种元素组成：

* 顶点/节点(Vertex/Node)，以 `$V$` 表示，表示顶点的集合
* 边/链接(Edge/Link)，以 `$E$` 表示，表示边的集合
    - 对于有向图通常称有方向的边为弧(arc)，用尖括号表示弧的方向，
      如 `<A, B>` 表示弧的方向是 `$A \rightarrow B$`。
      有向图的边或弧具有与它相关数字，这种与图的边或弧相关的数字叫作**权**，
      在实际问题中，权通常是两个城市之间的距离、两个地点之间的运输费用等。

顶点和边具有属性(Attribute)，边可能有方向(有向图 Directed Graph)。
社交网络中，人是顶点，人和人之间的关系是边，人/顶点的属性比如年龄、性别、职业、爱好等构成了一个向量，
类似的，边也可用向量来表示。

图的其他概念：

* 连通图：若图中任意两个顶点之间至少有一条路径连接起来
* 连通分量：在无向图中极大连通子图称为连通分量，注意连通分量的概念：
    - 首先它是子图
    - 其次子图是连通的，连通子图具有极大顶点数
    - 最后，具有极大顶点数的连通子图包含依附于这些顶点的所有边
* 强连通分量：在有向图中，如果对于每一对 `$v_{i}$` 和 `$v_{j}$`，`$v_{i} \neq v_{j}$`，
  无论从 `$v_{i}$` 到 `$v_{j}$` 还是从 `$v_{j}$` 到 `$v_{i}$` 都存在路径，则称为强连通图。
  有向图中的极大强连通子图称为有向图的强连通分量。

# 图的矩阵表示

图本身也具有表达其自身的全局属性，来描述整个图

## 邻接矩阵

如何用数学表示图中顶点的关系呢？最常见的方法是邻接矩阵(Adjacency Matrix)。

邻接矩阵是一个二维矩阵，其中矩阵的行和列表表示顶点编号，值表示顶点之间的边信息。
设图 G 有 `$n$` 个顶点，则邻接矩阵是一个 `$n\times n$` 的方阵，
用 `$\text{arch}$` 表示，方阵的元素 `$\text{arch}[i, j]$` 定义为：

`$$\text{arch}[i,j]=\begin{cases}
1, \text{if} \space (v_{i}, v_{j}) \in E \\
0, \text{otherwise} 
\end{cases}$$`

注意：对于有向图，`$\text{arch}[i, j] \neq \text{arch}[j, i]$`，
而无向图则是 `$\text{arch}[i, j] = \text{arch}[j, i]$`。在有向图中，
通常用 `$1$` 表示 `$<v_{i}, v_{j}>$` 连通，用 `$0$` 表示 `$<v_{i}, v_{j}>$` 不连通。
在实际问题中，还可以用 `$\text{arch}[i, j]$` 表示城市之间的距离，运输的费用等。

下图中 A 和 B、C、E 相连，故第一行和第一列对应的位置为 1，其余位置为 0

![img](images/adjacency_matrix.png)

## 旅行商问题

> Traveling Saleman Problem, TSP

以 TSP 问题讲解邻接矩阵在图分析中的应用。TSP 问题假设有一个旅行商人要拜访 `$n$` 个城市，
他必须选择所要走的路径，路径的限制是每个城市只能拜访一次，而且最后要回到原来出发的城市。

问题：设有 20 个城市，要使任意两个城市之间能够直接通达，需要规划一条路线，使每个城市都能访问一次，

## 邻接列表

也可以选用边来表示图，即邻接列表(Adjacency List)，这可以大幅减少对空间的消耗，
因为实际的边比所有可能的边(邻接矩阵)数量往往小很多。

![img](images/adjacency_list.png)

# 最小生成树

# 图问题

## 最短路径问题

## 网络最大流问题

## 路径规划

## 车辆路径问题

> Vehicle Routing Problem, VRP, 车辆路径问题
