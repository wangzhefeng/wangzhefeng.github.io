---
title: 读统计之都《因果推断简介》系列文章
author: 王哲峰
date: '2023-07-09'
slug: statistics-causal-inference
categories:
  - 数学、统计学
tags:
  - article
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

- [Yule-Simpson's Paradox](#yule-simpsons-paradox)
  - [因果推断的教材](#因果推断的教材)
  - [Yule-Simpson's Paradox](#yule-simpsons-paradox-1)
  - [吸烟是否导致肺癌和健康工人效应](#吸烟是否导致肺癌和健康工人效应)
  - [结语](#结语)
- [Rubin Causal Model 和随机化试验](#rubin-causal-model-和随机化试验)
- [R.A.Fisher 和 J.Neyman 的分歧](#rafisher-和-jneyman-的分歧)
- [观察性研究，可忽略性和倾向得分](#观察性研究可忽略性和倾向得分)
- [因果图](#因果图)
- [参考](#参考)
</p></details><p></p>

# Yule-Simpson's Paradox

> 统计还能研究因果？

## 因果推断的教材

目前市面上能够买到的因果推断(causal inference)相关教科书：

* 2011 年图灵奖得主 Judea Pearl 的《Causality: Models, Reasoning, and Inference》

目前还没写完的：

* Harvard 的统计学家 Donald Rubin 和 计量经济学家 Guido Imbens 合著的教科书历时多年仍尚未完成
    - Donald Rubin 对 Judea Pearl 提出的因果图模型（causal diagram）非常反对，他的教科书中杜绝使用因果图模型
* Harvard 的流行病学家 James Robins 和他的同事也在写一本因果推断的教科书，本书目前只完成了第一部分，还未出版
* 统计之都系列文章的作者丁鹏的新书：[A First Course in Causal Inference](https://arxiv.org/abs/2305.18793)
    - 目前中文版还未翻译、出版，名字大概叫《因果推断基础教程》

## Yule-Simpson's Paradox

在高维列联表分析中， 有一个很有名的例子，叫做 Yule-Simpson’s Paradox。
有文献称，Karl Pearson 很早就发现了这个悖论——也许这正是他反对统计因果推断的原因。
此悖论表明，存在如下的可能性：`$X$` 和 `$Y$` 在边缘上正相关；但是给定另外一个变量 `$Z$` 后，
在 `$Z$` 的每一个水平上，`$X$` 和 `$Y$` 都负相关。Table 1 是一个数值的例子，取自 Pearl(2000)。

![img](images/yule_simpsons_paradox.png)

Table 1 中，第一个表是整个人群的数据：接受处理和对照的人都是 40 人，处理有较高的存活率，因此处理对整个人群有 “正作用”。
第二个表和第三个表是将整个人群用性别分层得到的，因为第一个表的四个格子数，分别是下面两个表对应格子数的和：

`$$20 = 18 + 2$$`
`$$20 = 12 + 8$$`
`$$16 = 7 + 9$$`
`$$24 = 3 + 21$$`

奇怪的是，处理对男性有 “负作用”，对女性也有 “负作用”。一个处理对男性和女性都有 “负作用”，但是他对整个人群却有 “正作用”：悖论产生了！

> 个人理解：
> * **Treatment** 和 **Control** 这两种对人群的处理方式是变量 `$X$`
> * **Survive Rate**(存活率)就是指变量 `$Y$`
> * 性别指的就是是变量 `$Z$`，**Male** 和  **Female** 是变量 `$Z$` 的两个水平

有人可能会认为这种现象是由于随机性或者小样本的误差导致的。但是这个现象与样本量无关，与统计的误差也无关。
比如，将上面的每个格子数乘以一个巨大的正数，上面的悖论依然存在。

纯数学的角度，上面的悖论可以写成初等数学；

`$$\frac{a}{b} < \frac{c}{d}，\frac{a'}{b'}<\frac{c'}{d'}，\frac{a+a'}{b+b'}>\frac{c+c'}{d+d'}$$`

这并无新奇之处。但是在统计上，这具有重要的意义——变量之间的相关关系可以完全的被第三个变量 “扭曲”。
更严重的问题是，我们的收集的数据可能存在局限性，忽略潜在的“第三个变量” 可能改变已有的结论，而我们常常却一无所知。
鉴于 Yule-Simpson 悖论的潜在可能，不少人认为，统计不可能用来研究因果关系。

上面的例子是人工构造的，在现实中，也存在不少的实例正是 Yule-Simpson’s Paradox：

* UC Berkeley 的著名统计学家 Peter Bickel 教授 1975 年在 Science 上发表文章，
  报告了 Berkeley 研究生院男女录取率的差异。他发现，总体上，男性的录取率高于女性，
  然而按照专业分层后，女性的录取率却高于男性 (Bickel 等 1975)
* 在流行病学的教科书 (如 Rothman 等 2008) 中，都会讲到 “混杂偏倚”（confounding bias），
  其实就是 Yule-Simpson’s Paradox，书中列举了很多流行病学的实际例子

## 吸烟是否导致肺癌和健康工人效应

由于有 Yule-Simpson’s Paradox 的存在，观察性研究中很难得到有关因果的结论，除非加上很强的假定。

一个很经典的问题：吸烟是否导致肺癌？由于我们不可能对人群是否吸烟做随机化试验，
我们得到的数据都是观察性的数据：即吸烟和肺癌之间的相关性 （正如 Table 1 的合并表）。
此时，即使我们得到了吸烟与肺癌正相关，也不能断言 “吸烟导致肺癌”。这是因为可能存在一些未观测的因素，
他既影响个体是否吸烟，同时影响个体是否得癌症。比如，某些基因可能使得人更容易吸烟，同时容易得肺癌；
存在这样基因的人不吸烟，也同样得肺癌。此时，吸烟和肺癌之间相关，却没有因果作用。

相反的，我们知道放射性物质对人体的健康有很大的伤害，但是铀矿的工人平均寿命却不比常人短；
这是流行病学中有名的 “健康工人效应”（healthy worker effect）。这样一来，似乎是说铀矿工作对健康没有影响。
但是，事实上，铀矿的工人通常都是身强力壮的人，不在铀矿工作寿命会更长。此时，在铀矿工作与否与寿命不相关，
但是放射性物质对人的健康是有因果作用的。

## 结语

这里举了一个悖论，但没有深入的阐释原因。阐释清楚这个问题的根本原因，其实就讲清楚了什么是因果推断。

# Rubin Causal Model 和随机化试验 

> Rubin Causal Model，RCM

因果推断用的最多的模型是 Rubin Causal Model (RCM; Rubin 1978) 和 Causal Diagram (Pearl 1995)。
Pearl (2000) 中介绍了这两个模型的等价性，但是就应用来看，RCM 更加精确，而 Causal Diagram 更加直观，
后者深受计算机专家们的推崇。这里主要讲了 RCM。

设 `$Z_{i}$` 表示个体 `$i$` 接受处理与否，处理取 `$1$`，对照取 `$0$`(这部分的处理变量都讨论二值的，多值的可以做相应的推广)；
`$Y_{i}$` 表示个体 `$i$` 的结果变量。另外记 `$\{Y_{i}(1), Y_{i}(0)\}$` 表示个体 `$i$` 接受处理或者对照的潜在结果(potential outcome)，
那么 `$Y_{i} - Y_{0}$` 表示个体 `$i$` 接受治疗的个体因果作用。不幸的是，每个个体要么接受处理，要么接受对照，
`$\{Y_{i}(1), Y_{i}(0)\}$` 中必然缺失一半，个体的作用是不可识别的。观测的结果是 `$Y_{i} = Z_{i}Y_{i}(1)+(1-Z_{i})Y_{i}(0)$`。
但是，在 `$Z$` 做随机化的前提下，我们可以识别总体的平均因果作用(Average Causal Effect, ACE)：

`$$ACE(Z \rightarrow Y)=E[Y_{i}(1) - Y_{i}(0)]$$`

这是因为：

`$$\begin{align}
ACE(Z \rightarrow Y) &= E[Y_{i}(1)] - E[Y_{i}(0)] \
&= E[Y_{i}(1) | Z_{i} = 1] - E[Y_{i}(0) | Z_{i} = 0] \
&= E[Y_{i} | Z_{i}=1]
\end{align}$$`

最后一个等式表明 ACE 可以由观测的数据估计出来。其中第一个等式用到了期望算子的线性性（非线性的算子导出的因果度量很难被识别）；
第二个式子用到了随机化，即：

# R.A.Fisher 和 J.Neyman 的分歧 

完全随机化试验下的 Fisher randomization test 和 Neyman repeated sampling procedure。
简单地说，前者是随机化检验，或者如很多教科书讲的 Fisher 精确检验（Fisher exact test）；
后者是 Neyman 提出的置信区间 （confidence interval）理论。




# 观察性研究，可忽略性和倾向得分 



# 因果图

> 因果图，Causal Diagram







# 参考

* [统计之都-因果推断系列文章](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MjM5NDQ3NTkwMA==&action=getalbum&album_id=2515660787328253953&scene=173&from_msgid=2650147649&from_itemidx=1&count=3&nolastread=1#wechat_redirect)
