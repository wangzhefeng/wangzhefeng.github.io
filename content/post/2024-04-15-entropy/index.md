---
title: 熵 Entropy
author: wangzf
date: '2024-04-15'
slug: entropy
categories:
  - 整理
tags:
  - article
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
h3 {
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

- [熵的概念](#熵的概念)
    - [熵的基本概念](#熵的基本概念)
    - [熵的计算方法](#熵的计算方法)
    - [通俗解释](#通俗解释)
    - [抽象解释](#抽象解释)
        - [性质 1](#性质-1)
        - [性质 2](#性质-2)
        - [性质 3](#性质-3)
        - [性质 4](#性质-4)
    - [熵的衍生物](#熵的衍生物)
- [参考](#参考)
</p></details><p></p>

# 熵的概念

## 熵的基本概念

通俗来讲，熵一般有两种解释：（1）<span style='border-bottom:1.5px dashed red;'>熵是不确定性的度量</span>；
（2）<span style='border-bottom:1.5px dashed red;'>熵是信息的度量</span>。看上去说的不是一回事，
其实它们说的就是同一个意思。

熵是不确定性的度量，它衡量着我们对某个事物的 “无知程度”。
熵为什么又是信息的度量呢？既然熵代表了我们对事物的无知，
那么当我们从 “无知” 到 “完全认识” 这个过程中，就会获得一定的信息量，
我们越开始无知，那么到达 “完全认识” 时，获得的信息量就越大。

因此，作为不确定性的度量的熵，也可以看作是信息的度量，
说准确点，是我们能从中获得的最大的信息量。

## 熵的计算方法

> 如何衡量不确定性？也就是说，如何计算熵？

可以想到，用<span style='border-bottom:1.5px dashed red;'>概率</span>来描述不确定性，
可以给出系统出现某种状态的概率（或者概率密度，这里不做区分），
只要概率不为 1，就意味着出现不确定性。因此，熵必定与概率分布有关。
对于具体怎么计算熵？这里尝试从两个角度给出熵的公式的来源。

熵的基本计算公式为：

`$$S=-\sum_{x}p(x)\log p(x)\tag{1}$$`

其中：

* `$\log$` 可以是自然对数，也可以是以 2 为低的对数，事实上，
  任意大于 1 的底都是成立的，因为换底只不过相当于换了信息的单位；
* `$p(x)$` 是量 `$X$` 取值为 `$x$` 的概率。

对于连续的概率分布，熵类似地定义为（把求和换成积分）：

`$$S=-\int p(x) \log p(x) dx\tag{2}$$`

其中：

* `$p(x)$` 是 `$X$` 的概率密度函数

> 注意：上面两个公式的几个特征是：对数 `$\log$`、求和 `$\sum_{x}$` 以及前面的负号。
> 下面将着重解释这几个特征，从两个方面，一个是比较通俗的，一个是比较抽象的（数学化的）。

## 通俗解释

```
一个例子：世界杯结束后，大家都很关心谁会是冠军。假如我错过了看世界杯，赛后我问
一个知道比赛结果的观众“哪支球队是冠军”？他不愿意直接告诉我，而要让我猜，并且我
每猜一次，他要收一元钱才肯告诉我是否猜对了，那么我需要付给他多少钱才能知道谁是
冠军呢？我可以把球队编上号，从 1 到 32，然后提问：“冠军的球队在 1-16 号中吗？”
假如他告诉我猜对了，我会接着问：“冠军在 1-8 号中吗？” 假如他告诉我猜错了，我
自然知道冠军队在 9-16 中。这样只需要 5 次，我就能知道哪支球队是冠军。所以，谁
是世界杯冠军这条消息的信息量只值 5 块钱。

出自：吴军老师的著作《数学之美》中的 “谈谈最大熵模型” 这一章
```

以上例子成立的前提是：“我” 对足球一无所知：“我”既没有研究足球，
也没有听说过任何关于足球的消息。在这种情况下，我只能靠猜，
最朴素的方法就是一支支队伍地猜：是中国吗？是巴西吗？是日本吗？
... 这样我们有可能问到第 31 个问题才得到最终结果。显然，这不是最高效率的方法。
在例子中，我们可以先对球队编号（在数据处理时，我们称之为建立“索引”），
然后用二分法查找算法，该算法的计算复杂度是 `$\mathscr{O}(\log_2 N)$`。

这给我们两个启示：<span style='border-bottom:1.5px dashed red;'>一是建立索引，
并通过二分法，能够大大加快查找的速度，不过这跟本文没多大关系；第二就是 `$\log_{2}N$`，
对数出现了！</span>它可以改写为：

`$$\log_{2} N = -\log_{2}\frac{1}{N} = -\sum_{N 支队伍}\frac{1}{N} \log_{2}\frac{1}{N}$$`

这正好是上面式 `(1)` 的形式，这里因为我们对足球一无所知，所以每支队伍得冠军的概率都是一样的，
也就是 `$p=\frac{1}{N}$`。

可能这个例子太特殊，结果没什么代表性。确实，这只不过是一个感性的认知。
从更抽象的、更精准的数学角度来理解也是可以的，这时候，我们的思路是反过来的。

## 抽象解释

首先，我们希望构造出一个公式来表示信息量，或者等价地，不确定的程度，
叫做 “熵”。

> 注意，这里是我们希望根据我们所想的用途去构造一个量，
> 而不是我们已经得到了这个量，然后才证明它有这样的用途。
> 这里边的逻辑刚好是反过来的，是我们需要什么，我们就去构造什么。

既然 “熵” 用来表示信息量，那么它应该具有下面的简单的性质：

### 性质 1

熵是概率分布 `$p(x)$` 的函数，记作 `$S[p(x)]$`，为了研究的方便，
还希望它是一个光滑函数。

### 性质 2

熵具有可加性，这意味着熵具有形式：

`$$S[p(x)] = \sum_{x}f\Big(p(x)\Big)\tag{3}$$`

现在的问题是，`$f$` 的具体形式是什么，我们需要一些额外的信息来确定它。
比如，假如 `$X$`，`$Y$` 是两个独立的随机变量，它们的概率分布分别是 `$p(x)$` 和 `$p(y)$`，
那么 `$X$`，`$Y$` 的联合概率是 `$p(x)p(y)$`。因为两个随机变量是独立的，
那么它们的联合分布 `$p(x)p(y)$` 和单独的两个分布 `$p(x)$`、
`$p(y)$` 所具有的信息量是等价的（从联合密度分布可以算得单个的密度分布，
从单个的密度分布，可以算得联合密度分布），也就是下面的性质 3。

### 性质 3

当 `$X$`、`$Y$` 是独立随机变量时，有：

`$$S[p(x)p(y)] = S[p(x)] + S[p(y)]\tag{4}$$`

事实上，上述三个性质就可以确定熵的表达式。为了从 (4) 式确定 `$f$` 的形式，
我们只需要从最简单的二元分布出发，假设 `$X$`、`$Y$` 都是二元随机变量，`$X$` 的概率分布为 `$p$`、`$1−p$`，
`$Y$` 的概率分布为 `$q$`、`$1−q$`，那么联合分布为 `$pq$`、`$p(1−q)$`、`$(1−p)q$`、`$(1−p)(1−q)$`，
根据 (4) 式就有：

`$$\begin{aligned}
&f(pq)+f\big(p(1-q)\big)+f\big((1-p)q\big)+f\big((1-p)(1-q)\big) \\ 
=&f(p)+f(1-p)+f(q)+f(1-q)
\end{aligned}\tag{5}$$`

这是关于 `$f$` 的一个函数方程，只要加上适当的合理的限制，那么它就具有唯一解。
这里我们尝试求解它，但不去证明解的唯一性。求解的过程是试探性的，我们发现，
左边是自变量的积的形式，如 `$pq$`，右边是单个的自变量的形式，如 `$p$`，回想数学中的概念，
我们可以想起来，能够把乘积变为求和的运算是取对数，所以我们不妨设 `$f(x)=h(x)\ln x$`，
得到：

`$$\begin{aligned}
&h(p)\ln p+h(1-p) \ln (1-p)+h(q)\ln q+h(1-q)\ln (1-q)\\ 
=&h(pq)\ln p+h(pq)\ln q \\ 
&+ h\big(p(1-q)\big)\ln p + h\big(p(1-q)\big)\ln(1-q)\\ 
&+h\big((1-p)q\big)\ln(1-p)+h\big((1-p)q\big)\ln q \\ 
&+ h\big((1-p)(1-q)\big)\ln(1-p)+h\big((1-p)(1-q)\big)\ln(1-q)
\end{aligned}$$`

把相同对数的项合并起来，比如 `$lnp$` 项，是

`$$\big[h(p)-h(pq)-h(p(1-q))\big]\ln p\tag{7}$$`

剩余三项也类似。我们发现，如果 `$h$` 取线性函数，那么上式刚好是 0，剩余三项也是 0，
等式自动满足！所以，我们就找到了一个解：

`$$f(x)=\alpha x\ln x\tag{8}$$`

所以我们有了熵的表达式：

`$$S=\sum_{x}\alpha p(x)\ln p(x)\tag{9}$$`

最后，还要确定 `$\alpha$`，当然，`$\alpha$` 本身的值不重要，它只不过是信息的单位而已，
但是 `$\alpha$` 的符号是很重要的。我们要求熵有下面的性质。

### 性质 4

信息量越大，熵越大。

第 4 点是为了符合我们的习惯而已，如果你喜欢，你也可以定义“信息量越大，熵越小”。
既然这样定义，我们就知道，确定性事件的熵肯定比不确定的事件的熵要小（不确定的事情蕴含的信息越大），
确定性事件的概率分布也就是恒等于 1，对应的熵是 0，而不确定性事件，我们还是取二元分布，
概率分布为 `$p$`，`$1-p$`，那么必然有：

`$$\alpha p \ln p + \alpha (1-p) \ln(1-p) > 0\tag{10}$$`

因此只有 `$\alpha < 0$`。

根据以上讨论，我们已经发现，熵已经不再是物理概念的抽象，而已经是一个完全独立的对象。
熵来源于物理，但基本上已经脱胎于物理，成为了一个能够贯穿信息、物理、生物等领域的强大工具。
事实上，作为物理方面的应用，我们可以反过来，从后面要谈到的最大熵原理出发，建立起物理定律，
这时候，熵不仅不是衍生物，还成为了物理定律的来源。

## 熵的衍生物

在有了熵的定义式 `(1)` 和 `(2)`，就可以得到一些 “衍生品” 了，
比如 “联合熵”，这是一元熵的等价推广罢了：

`$$S[p(x, y)] = -\sum_{x}\sum_{y}p(x,y) \ln p(x, y)$$`

为了后面要讲到的最大熵模型，需要引入一个条件熵，它跟条件分布 `$p(y|x)$` 有关。
我们已经知道，条件分布就是在联合分布 `$p(x,y)$` 的基础上，已经知道 `$p(x)$` 的分布，
求 `$X$` 确定时，`$Y$` 的分布情况。那么条件熵自然是在联合熵的基础上，再引入 `$X$` 的熵，
所剩下的熵值：

`$$S(Y|X)=S[p(x,y)]-S[p(x)]\tag{14}$$`

说白了，条件熵就是说本来不确定性有 `$S[p(x,y)]$` 这么多，
然后 `$p(x)$` 能带来量为 `$S[p(x)]$` 的信息，
减少掉一定的不确定性，剩下的不确定性，就是条件熵。

# 参考

* [“熵”不起：从熵、最大熵原理到最大熵模型（一）](https://kexue.fm/archives/3534)
* [“熵”不起：从熵、最大熵原理到最大熵模型（二）](https://kexue.fm/archives/3552)
* [“熵”不起：从熵、最大熵原理到最大熵模型（三）](https://kexue.fm/archives/3567)
* [熵的社会学意义](https://www.ruanyifeng.com/blog/2013/04/entropy.html)
