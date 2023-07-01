---
title: GBDT、Logistic Regression CTR
author: 王哲峰
date: '2023-03-16'
slug: rs-ctr-gbdt-lr
categories:
  - recommended system
tags:
  - model
---

# 简介

协同过滤和矩阵分解存在的劣势就是仅利用了用户与物品相互行为信息进行推荐，忽视了用户自身特征，
物品自身特征以及上下文信息等，导致生成的结果往往会比较片面。而这次介绍的这个模型是 2014 年由 Facebook 提出的 GBDT+LR 模型，
该模型利用 GBDT 自动进行特征筛选和组合，进而生成新的离散特征向量，再把该特征向量当做 LR 模型的输入，
来产生最后的预测结果，该模型能够综合利用用户、物品和上下文等多种不同的特征，生成较为全面的推荐结果，
在 CTR 点击率预估场景下使用较为广泛

# GBDT 结合 Logistic Regression 模型

2014年，Facebook 提出了一种利用 GBDT 自动进行特征筛选和组合，进而生成新的离散特征向量，再把该特征向量当做 LR 模型的输入，
来产生最后的预测结果，这就是著名的 GBDT+LR 模型了。GBDT+LR 使用最广泛的场景是 CTR 点击率预估，即预测当给用户推送的广告会不会被用户点击

![img](images/)

训练时，GBDT 建树的过程相当于自动进行的特征组合和离散化，然后从根结点到叶子节点的这条路径就可以看成是不同特征进行的特征组合，
用叶子节点可以唯一的表示这条路径，并作为一个离散特征传入 LR 进行二次训练

比如上图中，有两棵树，`$x$` 为一条输入样本，遍历两棵树后，`$x$` 样本分别落到两颗树的叶子节点上，
每个叶子节点对应 LR 一维特征，那么通过遍历树，就得到了该样本对应的所有 LR 特征。
构造的新特征向量是取值 0/1 的。比如左树有三个叶子节点，右树有两个叶子节点，
最终的特征即为五维的向量。对于输入 `$x$`，假设他落在左树第二个节点，编码 `$[0,1,0]$`，落在右树第二个节点则编码 `$[0,1]$`，
所以整体的编码为 `$[0,1,0,0,1]$`，这类编码作为特征，输入到线性分类模型（LR or FM）中进行分类

预测时，会先走 GBDT 的每棵树，得到某个叶子节点对应的一个离散特征(即一组特征组合)，
然后把该特征以 one-hot 形式传入 LR 进行线性加权预测

这个方案应该比较简单了，下面有几个关键的点我们需要了解：

1. 通过 GBDT 进行特征组合之后得到的离散向量是和训练数据的原特征一块作为逻辑回归的输入，而不仅仅全是这种离散特征
2. 建树的时候用 ensemble 建树的原因就是一棵树的表达能力很弱，不足以表达多个有区分性的特征组合，
   多棵树的表达能力更强一些。GBDT 每棵树都在学习前面棵树尚存的不足，迭代多少次就会生成多少棵树
3. RF 也是多棵树，但从效果上有实践证明不如 GBDT。且 GBDT 前面的树，特征分裂主要体现对多数样本有区分度的特征；
   后面的树，主要体现的是经过前 N 颗树，残差仍然较大的少数样本。优先选用在整体上有区分度的特征，
   再选用针对少数样本有区分度的特征，思路更加合理，这应该也是用 GBDT 的原因
4. 在 CRT 预估中，GBDT 一般会建立两类树(非 ID 特征建一类，ID 类特征建一类)，AD，ID 类特征在 CTR 预估中是非常重要的特征，
   直接将 AD，ID 作为 feature 进行建树不可行，故考虑为每个 AD，ID 建 GBDT 树
    - 非 ID 类树：不以细粒度的 ID 建树，此类树作为 base，即便曝光少的广告、广告主，仍可以通过此类树得到有区分性的特征、特征组合
    - ID 类树：以细粒度的 ID 建一类树，用于发现曝光充分的 ID 对应有区分性的特征、特征组合

# 参考

* [逻辑回归 + GBDT模型融合实战](https://mp.weixin.qq.com/s/XP5z_BEeFr6oJp9VmVJRqQ)
