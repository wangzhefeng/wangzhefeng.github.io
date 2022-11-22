---
title: 高斯混合聚类
author: 王哲峰
date: '2022-11-22'
slug: ml-clustering-mixture-of-gaussian
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

- [算法介绍](#算法介绍)
  - [(多元)高斯分布](#多元高斯分布)
  - [(多元)高斯混合分布](#多元高斯混合分布)
  - [样本集的生成模型](#样本集的生成模型)
  - [高斯混合聚类策略](#高斯混合聚类策略)
  - [算法](#算法)
- [算法实现](#算法实现)
</p></details><p></p>

# 算法介绍

高斯混合聚类(Mixture-of-Gaussian)采用概率模型来表达聚类原型. 

## (多元)高斯分布

对 `$n$` 维样本空间 `$\mathcal{X}$` 中的随机向量 `$x$`, 若 `$x$` 服从(多元)高斯分布, 其概率密度函数为: 

$$p(x| \mu, \Sigma)=\frac{1}{(2\pi)^{\frac{n}{2}}|\Sigma|^{\frac{1}{2}}} e^{ - \frac{1}{2} (x-\mu)^{T} \Sigma^{-1} (x-\mu)}$$

其中: `$\mu$` 是 `$n$` 维均值向量, `$\Sigma$` 是 `$n \times n$` 协方差矩阵.

## (多元)高斯混合分布

对 `$n$` 维样本空间 `$\mathcal{X}$` 中的随机向量 `$x$`, 若 `$x$` 服从(多元)高斯混合分布, 其概率密度函数为: 

$$p_{\mathcal{M}}(x)=\sum_{i=1}^{k}\alpha_{i} \cdot p(x| \mu_{i}, \Sigma_{i})$$

该分布由 `$k$` 个混合成分组成, 每个成分对应一个(多元)高斯分布,

其中: `$\mu_{i}$`, `$\Sigma_{i}$` 是第 `$i$` 个高斯混合成分的参数, 
而 `$\alpha_{i}$` 为相应的"混合系数"(mixture coeffcient), 
`$\sum_{i=1}^{k}\alpha_{i}=1$`.

## 样本集的生成模型

假设样本集 ``$D=\{x_{1}, x_{2}, \ldots, x_{n}\}$`` 的生成过程有高斯混合分布给出: 

* 首先: 根据 `$\alpha_{1}, \alpha_{2}, \ldots,\alpha_{k}$` 定义的先验分布选择高斯混合成分, 
  其中 `$\alpha_{i}$` 为选择第 `$i$` 个混合成分的概率;
* 然后: 根据被选择的混合成分的概率密度函数进行采样, 生成相应的样本.

令随机变量 `$z_{j} \in \{1, 2, \ldots,k\}$` 表示生成样本 `$x_{j}$` 的高斯混合成分, 其取值未知. 

`$z_{j}$` 的先验概率: 

$$P(z_{j}=i)=\alpha_{i} \quad (i=1, 2, \ldots,k)$$. 

由 Bayesian 定理得 `$z_{j}$` 的后验分布为: 

$$\begin{align*}
P_{\mathcal{M}}(z_{j}=i|x_{j}) &=\frac{P(z_{j}=i) \cdot p_{\mathcal{M}}(x_{j}|z_{j}=i)}{p_{\mathcal{M}}(x_{j})}\\
                               &=\frac{\alpha_{i}\cdot p(x_{j}|\mu_{i},\Sigma_{i})}{\sum^{k}_{l=1}\alpha_{l}\cdot p(x_{j}|\mu_{l},\Sigma_{l})}
\end{align*}$$

$P_{\mathcal{M}}(z_{j}=i|x_{j})$ 给出了样本 `$x_{j}$` 由第 `$i$` 个高斯混合成分生成的后验概率, 记: 

$$\gamma_{ji}=P_{\mathcal{M}}(z_{j}=i|x_{j}) \quad (i=1, 2, \ldots,k)$$


## 高斯混合聚类策略

* 若(多元)高斯混合分布 `$p_{\mathcal{M}}(x)$` 已知, 高斯混合聚类将把样本集 `$D$` 划分为 `$k$` 个簇 

$$C=\{C_{1}, C_{2}, \ldots,C_{k}\}$$

每个样本 `$x_{j}$` 的簇标记 `$\lambda_{j}$` 为:

$$\lambda_{j}=arg \underset{i \in \{1, 2, \ldots,k\}}{max}\gamma_{ji}$$


* (多元)高斯混合分布 `$p_{\mathcal{M}}(x)$` 参数 `$\{(\alpha_{i},\mu_{i}, \Sigma_{i})|1 \leqslant i \leqslant k\}$` 的求解采用极大似然估计(MLE):

给定样本集 `$D$`, 最大化(对数)似然函数: 

`$$\begin{align*}
LL(D)&=ln\Big(\prod^{n}_{j=1}p_{\mathcal{M}}(x_{j})\Big)\\
&=\sum^{n}_{j=1}\Big(\sum^{k}_{i=1}\alpha_{i}\cdot p(x_{j}|\mu_{i}, \Sigma_{i})\Big)
\end{align*}$$`

MLE 解为: 

`$$\mu_{i}=\frac{\sum^{n}_{j=1}\gamma_{ji}x_{j}}{\sum^{n}_{j=1}\gamma_{ji}}$$`
`$$\Sigma_{i}=\frac{\sum^{n}_{j=1}\gamma_{ji}(x_{j}-\mu_{i})(x_{j}-\mu_{i})^{T}}{\sum^{n}_{j=1}\gamma_{ji}}$$`
`$$\alpha_{i}=\frac{1}{n}\sum^{n}_{j=1}\gamma_{ji}$$`


## 算法

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

# 算法实现


