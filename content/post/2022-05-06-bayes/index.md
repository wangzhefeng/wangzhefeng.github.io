---
title: 贝叶斯统计分析
author: wangzf
date: '2022-05-06'
slug: bayes
categories:
  - 数学、统计学
tags:
  - book
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
img {
    pointer-events: none;
}
</style>


<details><summary>目录</summary><p>

- [贝叶斯统计](#贝叶斯统计)
    - [贝叶斯公式](#贝叶斯公式)
        - [条件概率](#条件概率)
        - [乘法公式](#乘法公式)
        - [全概率公式](#全概率公式)
        - [贝叶斯公式](#贝叶斯公式-1)
    - [贝叶斯估计](#贝叶斯估计)
        - [统计推断的基础](#统计推断的基础)
        - [贝叶斯公式的密度函数形式](#贝叶斯公式的密度函数形式)
        - [贝叶斯估计](#贝叶斯估计-1)
        - [共轭先验分布](#共轭先验分布)
    - [统计决策理论与贝叶斯分析](#统计决策理论与贝叶斯分析)
        - [统计决策问题和损失函数](#统计决策问题和损失函数)
        - [决策函数和风险函数](#决策函数和风险函数)
        - [贝叶斯决策准则和 贝叶斯分析](#贝叶斯决策准则和-贝叶斯分析)
    - [贝叶斯定理](#贝叶斯定理)
- [贝叶斯建模](#贝叶斯建模)
- [贝叶斯思维](#贝叶斯思维)
    - [建模和近似](#建模和近似)
    - [代码利用](#代码利用)
    - [统计计算](#统计计算)
        - [分布](#分布)
        - [曲奇饼问题](#曲奇饼问题)
        - [Monty Hall 难题](#monty-hall-难题)
        - [M\&M 豆问题](#mm-豆问题)
    - [估计](#估计)
- [参考](#参考)
</p></details><p></p>

# 贝叶斯统计

> 在统计学中有两个大的学派: **频率学派** (也称经典学派) 和 **贝叶斯学派**

## 贝叶斯公式

### 条件概率

> 将坚持概率的贝叶斯解释（Bayes interpretation），即根据概率描述对事件的信念程度，
> 并使用数据来增强、更新或削弱这些信念程度。在这种形式化中，信念程度被赋予某种语言的命题（能判断真或假的句子）。
> 根据概率演算规则，这些信念程度可进行组合和操作。

条件概率的三个公式中: 

* **乘法公式** 是求事件交的概率
* **全概率公式** 是求一个复杂事件的概率
* **贝叶斯公式** 是求一个条件概率

所谓条件概率，它是指在某事件 `$B$`  发生的条件下，求另一件事 `$A$`  的概率，
记为 `$P(A|B)$`，它与 `$P(A)$` 是不同的两类概率.  条件概率是两个无条件概率之商，
这就是条件概率的定义。

> **定义**

设 `$A$` 与 `$B$` 是样本空间 `$\Omega$` 中的两个事件，若 `$P(B) > 0$`，
则称下面的公式为"在 `$B$` 发生下 `$A$` 的条件概率"，简称**条件概率**

`$$P(A|B) = \frac{P(AB)}{P(B)}$$` 

> **性质**

条件概率是概率，即若设 `$P(B) > 0$`，则

1. `$P(A|B) \geq 0, A \in \mathcal{F}$`
2. `$P(\Omega|B) = 1$`
3. 若 `$\mathcal{F}$` 中的 `$A_1, A_2, \cdots, A_n, \cdots$` 互不相容，则: 

`$$P\Bigg(\bigcup_{n=1}^{\infty}A_{n}|B\Bigg) = \sum_{n = 1}^{\infty}P(A_{n}|B)$$`

### 乘法公式

1. 若 `$P(B)>0$`，则

`$$P(AB)=P(B)P(A|B)=P(A)P(B|A)$$`

2. 若 `$P(A_1, A_2, \cdots, A_{n-1})>0$`，则

`$$P(A_1, A_2, \cdots, A_{n})=P(A_{1})P(A_{2}|A_{1})P(A_{3}|A_{1}A_{2}) \cdots P(A_{n}|A_{1} \cdots A_{n-1})$$`

### 全概率公式

设 `$B_{1}, B_{2}, \cdots, B_{n}$` 为样本空间 `$\Omega$` 的一个分割，
即 `$B_{1}, B_{2}, \cdots, B_{n}$` 互不相容，且 `$\bigcup_{i=1}^{n}B_{i} = \Omega$`，
如果 `$P(B_{i})>0,i=1,2,\cdots,n$`，则对任一事件 `$A$` 有

`$$P(A)=\sum_{n = 1}^{\infty}P(B_{i})P(A|B_{i})$$`

全概率公式的最简单形式。假如 `$0<P(B)<1$`，则

`$$P(A) = P(B)P(A|B) + P(\bar{B})P(A|\bar{B})$$`

> 条件 `$B_{1}, B_{2}, \cdots, B_{n}$` 为样本空间的一个分割，
> 可改成 `$B_{1}, B_{2}, \cdots, B_{n}$` 互不相容，
> 且 `$A \subset \bigcup_{i=1}^{n}B_{i}$`，性质中的定理仍然成立

> TODO：
> 
> 在贝叶斯形式化中，信念测度服从概率演算的三个基本公理：
> 
> `$$0 \leq P(A) \leq 1$$`
> 
> `$$P(\text{确定命题}) = 1$$`
> 
> `$$\text{如果 A 与 B 互斥，则} P(A 或 B) = P(A) + P(B)$$`
> 
> 第三条公理表明，任何一组事件的信念是其非相交成分的信念的总和。
> 由于事件 `$A$` 都可以写成联合事件 `$(A \wedge B)$` 与 `$(A \wedge \neg B)$` 的并，
> 因此它们对应的概率可写为：
> 
> `$$P(A) = P(A, B) + P(A, \neg B)$$`
> 
> 其中，`$P(A, B)$` 是 `$P(A \wedge B)$` 的缩写。
> 
> 更一般地，如果 `$B_{i} (i=1,2,\cdots,n)$` 是一组完备的互斥命题（称为划分或变量），
> 那么 `$P(A)$` 可通过对 `$P(A, B_{i}) (i=1,2,\cdots,n)$` 求和得到：
> 
> `$$P(A) = \underset{i}{\sum}P(A, B_{i})$$`
> 
> 该式称为“全概率公式”。在所有 `$B_{i}$` 上的概率求和操作也称为“边缘化于 `$B$`”，
> 获得的概率 `$P(A)$` 称为 `$A$` 的边缘概率。

### 贝叶斯公式

设 `$B_{1}, B_{2}, \cdots, B_{n}$` 为样本空间 `$\Omega$` 的一个分割，
即 `$B_{1}, B_{2}, \cdots, B_{n}$` 互不相容，且 `$\bigcup_{i=1}^{n}B_{i} = \Omega$`，
如果 `$P(A) > 0, P(B_{i})>0,i=1,2,\cdots,n$`，则

`$$P(B_{i}|A)=\frac{P(B_{i})P(A|B_{i})}{\sum_{j=1}^{n}P(B_{j})P(A|B_{j})}, i=1,2,\cdots,n.$$`

在贝叶斯公式中，如果称 `$P(B_{i})$` 为事件 `$B_{i}$` 的 **先验概率**，
称 `$P(B_{i}|A)>0$` 为事件 `$B_{i}$` 的 **后验概率**，则贝叶斯公式是专门用于计算后验概率的，
也就是通过事件 `$A$` 的发生这个新信息，来对 `$B_{i}$` 的概率作出修正。

`$$P(B_{i}|A)=P(B_{i}) \cdot \frac{P(A|B_{i})}{\sum_{j=1}^{n}P(B_{j})P(A|B_{j})}, i=1,2,\cdots,n.$$`

## 贝叶斯估计

### 统计推断的基础

**经典统计学派** 对统计推断的规定如下: 统计推断是根据样本信息对总体分布或总体的特征数进行推断，
这里的统计推断使用到了两种信息: **总体信息** 和 **样本信息**。

**贝叶斯学派** 则认为: 统计推断除了总体信息、样本信息以外，还应该使用第三种信息: **先验信息**。

* 总体信息 
    - 总体信息即总体分布或总体所属分布族提供的信息
* 样本信息 
    - 样本信息即抽取样本所得观测值提供的信息
* 先验信息 
    - 如果把抽取样本看作做一次试验，则样本信息就是试验中得到的信息. 
      实际上，人们在试验之前要对要做的的问题在经验上和资料上总是有所了解的，
      这些信息对统计推断是有益的
    - 先验信息即是抽样(试验)之前有关统计问题的一些信息
    - 一般来说，先验信息来源于经验和历史资料

**贝叶斯统计学**

基于上述三种信息进行统计推断的统计学称为 **贝叶斯统计学**。它与经典统计学的差别就在于是否利用先验信息。

贝叶斯统计在重视使用总体信息和样本信息的同时，还注意先验信息的收集、挖掘和加工，使它数量化，形成先验分布，参加到
统计推断中来，以提高统计推断的质量。忽视先验信息的利用，有时是一种浪费，有时还会导出不合理的结论。

贝叶斯学派的基本观点是：**任一未知量 `$\theta$` 都可看作随机变量，可用一个概率分布去描述，这个分布称为先验分布**；
在获得样本之后，总体分布、样本与先验分布通过贝叶斯公式结合起来得到一个关于未知量 `$\theta$` 的新分布：后验分布；
任何关于 `$\theta$` 的统计推断都应该基于 `$\theta$` 的后验分布进行。

> 关乎未知量是否可看作随机变量在经典学派与贝叶斯学派间争论了很长时间。因为任一未知量都有不确定性，
> 而在表述不确定的程度时，概率与概率分布是最好的语言，因此把它看成随机变量是合理的。
> 如今经典学派已不反对这一观点：著名的美国统计学家莱曼（Lehmann,E.L.）在他的《点估计理论》一书中写道：
> “把统计问题中参数看作随机变量的实现要比看作位置参数更合理一些”。
> 如今两派的争论焦点是：**如何利用各种先验信息合理地确定先验分布**。
> 这在有些场合是容易解决的，但在很多场合是相当困难的。

### 贝叶斯公式的密度函数形式

贝叶斯公式的事件形式已经在上面介绍过，这里用随机变量的概率函数再一次叙述贝叶斯公式，
并从中介绍贝叶斯学派的一些具体想法。

1. 总体依赖于参数 `$\theta$` 的概率函数在经典统计中记为 `$p(x;\theta)$`，
   它表示参数空间 `$\Theta$` 中不同的 `$\theta$` 对应不同的分布。在贝叶斯统计中应记为 `$p(x|\theta)$`，
   它表示在随机变量 `$\theta$` 取某个给定值时总体的条件概率函数。
2. 根据参数 `$\theta$` 的先验信息确定先验分布 `$\pi(\theta)$`。
3. 从贝叶斯观点看，样本 `$X=(x_{1}, \cdots, x_{n})$` 的产生要分两步进行。
   首先设想从先验分布 `$\pi(\theta)$` 产生一个样本 `$\theta_{0}$`。这一步是“老天爷”做的，人们是看不到的，
   故用“设想”二字。第二步从 `$p(X|\theta_{0})$` 中产生一组样本。
   这时样本 `$X=(x_{1}, \cdots, x_{n})$` 的联合条件概率函数为

   `$$p(X|\theta_{0})=p(x_{1},\cdots,x_{n}|\theta_{0})=\prod_{i=1}^{n}p(x_{i}|\theta_{0})$$`

   这个分布综合了总体信息和样本信息。

4. 由于 `$\theta_{0}$` 是设想出来的，仍然是未知的，它是按先验分布 `$\pi(\theta)$` 产生的。
   为把先验信息综合进去，不能只考虑 `$\theta_{0}$`，对 `$\theta$` 的其他值发生的可能性也要加以考虑，
   故要用 `$\pi(\theta)$` 进行综合。这样一来，样本 `$X$` 和参数 `$\theta$` 的联合分布为

    `$$h(X,\theta)=p(X|\theta)\pi(\theta)$$`

    这个联合分布把总体信息、样本信息和先验信息三种可用信息都综合进去了。

5. 我们的目的是要对未知参数 `$\theta$` 作统计推断。在没有样本信息时，我们只能依据先验分布对 `$\theta$` 做出推断。
   在有了样本观测值 `$X=(x_{1},\cdots,x_{n})$` 之后，我们应依据 `$h(X, \theta)$` 对 `$\theta$` 作出推断。
   若把 `$h(X, \theta)$` 作如下分解：

   `$$h(X, \theta)=\pi(\theta|X)m(X)$$`

   其中 `$m(X)$` 是 `$X$` 的边际概率函数

   `$$m(X)=\int_{\theta}h(X, \theta)d\theta=\int_{\Theta}p(X|\theta)\pi(\theta)d\theta$$`

   它与 `$\theta$` 无关，或者说 `$m(X)$` 中不含 `$\theta$` 的任何信息，
   因此能用来对 `$\theta$` 作出推断的仅是条件分布 `$\pi(\theta|X)$`，它的计算公式是

   `$$\pi(\theta|X)=\frac{h(X, \theta)}{m(X)}=\frac{p(X|\theta)\pi(\theta)}{\int_{\Theta}p(X|\theta)\pi(\theta)d\theta}$$`

    这个条件分布称为 `$\theta$` 的后验分布，它集中了总体、样本和先验中有关 `$\theta$` 的一切信息。
    上式就是用密度函数表示的贝叶斯公式，它也是用总体和样本对先验分布 `$\pi(\theta)$` 作调整的结果，
    它要比 `$\pi(\theta)$` 更接近 `$\theta$` 的实际情况。

### 贝叶斯估计

由后验分布 `$\pi(\theta|X)$` 估计 `$\theta$` 有三种常用的方法： 

* 使用 **后验分布的密度函数最大值点** 作为 `$\theta$` 的点估计的 **最大后验估计**。
* 使用 **后验分布的中位数** 作为 `$\theta$` 的点估计的 **后验中位数估计**。
* 使用 **后验分布的均值** 作为 `$\theta$` 的点估计的 **后验期望估计**。用的最多的是后延期望估计，
  它一般称为 **贝叶斯估计**，记为 `$\hat{\theta}_{B}$`。

### 共轭先验分布

从贝叶斯公式可以看出，整个贝叶斯统计推断只要先验分布确定后就没有理论上的困难。
关于先验分布的确定有多种途径，最常用的先验分布类为 **共轭先验分布**。

> **定义**

设 `$\theta$` 是总体分布 `$p(x;\theta)$` 中的参数， `$\pi(\theta)$` 是其先验分布，
若对任意来自 `$p(x;\theta)$` 的样本观测值得到的后验分布 `$\pi(\theta|x)$` 与 `$\pi(\theta)$` 属于同一个分布族，
则称该分布族是 `$\theta$` 的共轭先验分布（族）。

## 统计决策理论与贝叶斯分析

- 经典统计学
    - (待补充)
- 统计决策理论
    - **统计决策理论** 是著名统计学家 A.Wald(1902-1950) 在 20 世纪 40 年代建立起来的，
      它与经典统计学的差别在于是否涉及后果. 经典统计学着重在推断上
      而不考虑在何处和效益如何. 而统计决策理论引入*损失函数*，用来度量效益大小，
      评价统计推断结果的优劣. 
- 贝叶斯分析、非决策的 贝叶斯分析、贝叶斯决策分析
    - **贝叶斯分析** 是英国学者 T.Bayes(1702-1761) 首先提出，在 20 世纪后半叶发展迅速. 
      它与经典统计学的差别在于 **是否使用先验信息(经验与历史资料)**. 经典统计学只用样本信息，而
      贝叶斯分析把先验信息与样本信息结合起来用于推断之中，形成 **非决策的 贝叶斯分析**. 
      若再使用后果信息，就形成 **贝叶斯决策分析**. 

### 统计决策问题和损失函数

### 决策函数和风险函数

### 贝叶斯决策准则和 贝叶斯分析

## 贝叶斯定理

联合概率满足乘法交换律: 

`$$P(AB)  = P(BA)$$` 

乘法公式: 

* `$P(AB) = P(A)P(B|A)$`
* `$P(BA) = P(B)P(A|B)$`
* `$P(A)P(B|A) = P(B)P(A|B)$` 

贝叶斯定理: 

`$$P(A|B) = \frac{P(A)P(B|A)}{P(B)}$$` 

> * 对于涉及条件概率的很多问题，贝叶斯定理提供了一个分而治之的策略。 
> * 如果 `$P(A|B)$` 难以计算，或难以用实验衡量，可以检查计算贝叶斯定理中的其他项是否更容易，
>   如 `$P(B|A)$`，`$P(A)$` 和 `$P(B)$`。  

# 贝叶斯建模

贝叶斯建模过程可以总结为以下三步：

1. 给定一些数据以及这些数据是如何生成的假设，然后通过一些概率分布来设计模型。
   大多数情况下，这些模型是粗略的近似，不过正是我们所需要的。
2. 根据贝叶斯定理将数据添加到模型里，然后把数据和假设结合起来推导初逻辑结果，这就是根据数据调整模型。
3. 检查模型是否有意义可以根据不同的标准，包括数据、我们在这方面的专业知识，有时还通过比较几个模型来评价模型。

贝叶斯模型是基于概率构建的，因此也称作 **概率模型**。为什么基于概率？因为概率这个数学工具能够很好地模拟不确定性。

# 贝叶斯思维

使用 Python 代码实现的 贝叶斯方法不是数学，离散近似而不是连续数学，
结果就是原本需要积分的地方变成了求和，概率分布的大多数操作变成了简单的循环. 

## 建模和近似

在应用任何分析方法前，必须决定真实世界中的哪些部分可以被包括进模型，而哪些细节可以被抽象掉. 

在解决问题的过程中，明确建模过程作为其中一部分是重要的，因为这会提醒我们考虑建模误差(也就是建模当中简化和假设带来的误差). 

本书中很多方法都基于离散分布，这让一些人担心数值误差，但对于真实世界的问题，数值误差几乎从来都小于建模误差. 
再者，离散方法总能允许较好的建模选择，我宁愿要一个近似的良好的模型也不要一个精确但却糟糕的模型. 

从另一个角度看，连续方法常在性能上有优势，比如能以常数时间复杂度的解法替换掉线性或者平方时间复杂度的解法. 

总的来说，推荐这些步骤的一个通用流程如下: 

1. 当研究问题时，以一个简化模型开始，并以清晰、好理解、实证无误的代码实现它. 
   注意力集中在好的建模决策而不是优化上
2. 一旦简化模型有效，再找到最大错误来源. 这可能需要增加离散近似过程当中值的数量，
   或者增加蒙特卡洛方法中的迭代数，或者增加模型细节. 
3. 如果对你的应用而言性能就已经足够了，则没必要优化. 但如果要做，有两个方向可以考虑: 
   评估你的代码以寻找优化空间，例如，如果你缓存了前面的计算结果，你也许能避免重复冗余的计算; 
   或者可以去发现找到计算捷径的分析方法. 

## 代码利用

书信息: 

- https://greenteapress.com/wp/think-bayes/
- https://github.com/wangzhefeng/ThinkBayes2
- https://github.com/rlabbe/ThinkBayes

环境配置: 

```bash
$ git clone git@github.com:wangzhefeng/ThinkBayes2.git
$ cd ThinkBayes2
$ pip install .
$ python -m pip install numpy scipy matplotlib jupyter pandas jupyterlab
$ python install_test.py
```

## 统计计算

### 分布

在统计学中，分布是一组值及其对应的概率. 

使用示例: 

- 建立一个 `$Pmf$` 来表示六面筛骰子的结果分布

```python
from thinkbayes2 import Pmf

pmf = Pmf()
for x in [1, 2, 3, 4, 5, 6]:
    pmf.Set(x, 1/6.0)

print(pmf.Prob(1))
print(pmf.Prob(7))
```

- 计算每个单词在一个词序列中出现的次数

```python
from thinkbayes2 import Pmf

pmf = Pmf()
word_list = ["the", "wang", "zhe", "feng", "the"]
for word in word_list:
    pmf.Incr(word, 1)

print(pmf.Prob("the"))
print(pmf.Prob("the") / pmf.Normalize())
```

### 曲奇饼问题

假设是事件 `$B_1$` 和 `$B_2$`

```python
from thinkbayes2 import Pmf

# 先验分布
pmf = Pmf()
pmf.Set("Bowl1", 0.5)
pmf.Set("Bowl2", 0.5)

# 更新基于数据(拿到一块香草曲奇饼) 后的分布, 将先验分别乘以对应的似然度
pmf.Mult("Bowl1", 0.75)
pmf.Mult("Bowl2", 0.5)

# 分布归一化
pmf.Normalize()

# 后验分布
print(pmf.Prob("Bowl1"))
print(pmf.Prob("Bowl2"))
```

- 贝叶斯框架:

```python
from thinkbayes2 import Pmf
class Cookie(Pmf):
    def __init__(self, hypos):
        Pmf.__init__(self)
        for hypo in hypos:
            self.Set(hypo, 1)
        self.Normalize()
    
    def Update(self, data):
        for hypo in self.Values():
            like = self.Likelihood(data, hypo)
            self.Mult(hypo, like)
        self.Normalize()
    
    mixes = {
        "Bowl1": dict(
            vanilla = 0.75, chocolate = 0.25
        ),
        "Bowl2": dict(
            vanilla = 0.5,  chocolate = 0.5
        ),
    }
    def Likelihood(self, data, hypo):
        mix = self.mixes(hypo)
        like = mix[data]
        return like

hypos = ["Bowl1", "Bowl2"]
pmf = Cookie(hypos)

# 更新
pmf.Update("vanilla")

# 打印每个假设的后验概率
for hypo, prob in pmf.Items():
    print(hypo, prob)
# 推广到从同一个碗取不只一个曲奇饼(带替换)的情形
dataset = ["vanilla", "chocolate", "vanilla"]
for data in dataset:
    pmf.Update(data)
```

### Monty Hall 难题

Monty Hall 类实现: 

```python
class Monty(Pmf):

    def __init__(self, hypos):
        Pmf.__init__(self)
        for hypo in hypos:
            self.Set(hypo, 1)
        self.Normalize()

    def Update(self, data):
        for hypo in self.Values():
            like = self.Likelihood(data, hypo)
            self.Mult(hypo, like)
        self.Normalize()

    def Likelihood(self, data, hypo):
        if hypo == data:
            return 0
        elif hypo == "A":
            return 0.5
        else:
            return 1

hypos = "ABC"
pmf = Monty(hypos)
data = "B"
pmf.Update(data)

for hypo, prob in pmf.Items():
    print(hypo, prob)
```

封装框架:

```python
class Suite(Pmf):
    """代表一套假设及其概率"""

    def __init__(self, hypo = tuple()):
        """初始化分配"""
        pass
    
    def Update(self, data):
        """更新基于该数据的每个假设"""
        pass

    def Print(self):
        """打印出假设和它们的概率"""
        pass
```

```python
from thinkbayes2 import Suite

class Monty(Suite):
    
    def Likelihood(self, data, hypo):
        if hypo == data:
            return 0
        elif hypo == "A":
            return 0.5
        else:
            return 1

suite = Monty("ABC")
suite.Update("B")
suite.Print()
```

### M&M 豆问题

```python
mix94 = dict(brown = 30, yellow = 20, red = 20, green = 10, orange = 10, tan = 10)
mix96 = dict(blue = 24, green = 20, orange = 16, yellow = 14, red = 13, brown = 13)
hypoA = dict(bag1 = mix94, bag2 = mix96)
hypoB = dict(bag1 = mix94, bag2 = mix94)

hypotheses = dict(A = hypoA, B = hypoB)

class M_and_M(Suite):

    def Likelihood(self, data, hypo):
        bag, color = data
        mix = self.hypotheses[hypo][bag]
        like = mix[color]
        return like

suite = M_and_M("AB")
suite.Update(("bag1", "yellow"))
suite.Update(("bag2", "green"))
suite.Print()
```

> * Suite 是一个抽象类(abstract type)，这意味着它定义了 Suite 应该有额接口，但并不提供完整的实现. 
>   Suite 接口包括了 Update 和 Likelihood 方法，但只提供了 Update 的实现，没有 Likelihood 的实现. 
> * 具体类(concrerte type)是继承自抽象父类的类，而且提供了缺失方法的实现. 

## 估计

* TODO

# 参考

* 《概率论与数理统计教程-第二版(茆诗松 程依明 濮晓龙)》
* 《高等数理统计-(茆诗松 程依明 濮晓龙)》 
