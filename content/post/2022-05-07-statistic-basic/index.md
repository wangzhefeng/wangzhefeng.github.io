---
title: 统计学
author: 王哲峰
date: '2022-05-07'
slug: statistic-basic
categories:
  - 数学、统计学
tags:
  - note
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

- [内容](#内容)
- [大数定律、中心极限定理](#大数定律中心极限定理)
- [统计推断理论](#统计推断理论)
  - [参数估计](#参数估计)
  - [假设检验](#假设检验)
    - [统计显著性](#统计显著性)
    - [假设检验的基本思想](#假设检验的基本思想)
    - [假设检验的基本步骤](#假设检验的基本步骤)
  - [抽样分布](#抽样分布)
- [不确定性](#不确定性)
  - [非传递性骰子](#非传递性骰子)
  - [医疗概率](#医疗概率)
  - [混沌](#混沌)
  - [社会选择与阿罗定理](#社会选择与阿罗定理)
  - [纽科姆悖论](#纽科姆悖论)
- [偏度、峰度](#偏度峰度)
  - [偏度](#偏度)
  - [峰度](#峰度)
  - [Python 实现](#python-实现)
- [data](#data)
- [data scatter plot](#data-scatter-plot)
- [Linear Regression](#linear-regression)
- [omnibus 检验](#omnibus-检验)
- [jarque_bera 检验](#jarque_bera-检验)
- [回归分析](#回归分析)
  - [回归分析简介](#回归分析简介)
  - [回归分析理论](#回归分析理论)
  - [回归分析实现](#回归分析实现)
- [方差分析](#方差分析)
  - [方差分析简介](#方差分析简介)
  - [方差分析理论](#方差分析理论)
    - [单因子方差分析](#单因子方差分析)
  - [方差分析实现](#方差分析实现)
</p></details><p></p>



整理一下统计学中常用的概念、方法论. 作为一个统计学出身的人, 遇到这些问题时希望不要被难倒


# 内容

- 大数定律、中心极限定理
- 贝叶斯公式、贝叶斯定理
- 参数估计
   - 点估计、区间估计
- 最大似然估计与EM算法
- 假设检验
   - A/B test
- 方差分析
- 回归分析
- 主成分分析
- 因子分析
- 聚类分析
- 统计显著性


# 大数定律、中心极限定理

- [维基百科](https://zh.wikipedia.org/wiki/%E5%A4%A7%E6%95%B0%E5%AE%9A%E5%BE%8B)

在统计学中, 大数定律又称大数法则、大数率, 是描述相当多次数重复实验的结果的定律; 
根据这个定律, 样本数量越多, 则其算术平均值就有越高的概率接近期望值. 

- **定义1 大数定律**

若 `$\xi_1, \xi_2,...,\xi_n,...$` 是随机变量序列, 令

$$\eta_{n} = \frac{\xi_1+\xi_2+...+\xi_n}{n}$$

若存在常数序列 `$a_1,a_2,...,a_n,...$` 对任何的正数 `$\epsilon$`, 恒有

$$\lim\limits_{n \to \infty}P(|\eta_n-a_n|<\epsilon)=1$$

则称序列 `${\epsilon_n}$` 服从 **大数定律**(或**大数法则**). 

- **切比雪夫(Chebyishev)不等式**

- **切比雪夫定理的特殊情况**

- **伯努利大数定理**

- **辛钦大数定理**

- **定义2 中心极限定理**

对于独立随机变量序列 `$\xi_1, \xi_2,...,\xi_n,...$`, 假定 `$E(\xi_n)$` 和 `$D(\xi_n)$` 都存在, 令

$$\zeta_n=\frac{\sum_{i=1}^{n}\xi_i-\sum_{i=1}^{n}E(\xi_i)}{\sqrt{\sum_{i=1}^{n}}}$$

若

$$\lim\limits_{n \to \infty}P(\zeta_n < x)=\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{x}e^{\frac{-t^2}{2}}dt$$
   
则称序列 `${\xi_n}$` 服从 **中心极限定理(Central Limit Theorem)**. 

- **同分布的中心极限定理**
- **德莫佛－拉普拉斯定理**

# 统计推断理论

- 参数估计
    - 点估计
    - 区间估计
- 假设检验
    - 参数假设检验问题
    - 非参数假设检验问题
- 抽样分布

## 参数估计

## 假设检验

假设检验(hypothesis test)是由K.Pearson于20世纪初提出的, 
之后由费希尔进行了细化, 并最终由Neyman和E.Pearson提出了较完整的假设检验理论. 

### 统计显著性

显著性、统计显著性(Statistical significance): 是指零假设为真的情况下拒绝零假设所要承担的风险水平,
又叫概率水平, 或者显著水平. 

- 显著性的含义是指两个群体的态度之间的任何差异是由于系统因素而不是偶然因素的影响. 
  假定控制了可能影响两个群体之间差异的所有其他因素, 
  因此, 余下的解释就是我们所推断的因素, 而这个因素不能够 100% 保证, 
  所以有一定的概率值, 叫显著性水平(Significant level). 
- 总的来说, 它表示群体之间得以相互区别的能力. 在统计假设检验中, 
  公认的小概率事件的概率值被称为统计假设检验的显著性水平, 
  对同一量, 进行多次计量, 然后算出平均值. 

### 假设检验的基本思想

假如实验结果与原假设H发生矛盾就拒绝原假设 `$H$`, 否则就接受原假设 `$H$`

假如对某个指标 `$\theta$` 进行检测, 检测的临界值为 `$\theta_0$`, 
即如果 `$\theta \geq \theta_0$` 则认为指标
`$\theta$` 合格, 如果 `$\theta < \theta_0$`, 
则认为指标 `$\theta$` 不合格; 

因此命题 "`$\theta \geq \theta_0$`" 将涉及如下两个参数集合: 

$$\Theta_0 = \{\theta:\theta \geq \theta_0\}$$
    
$$\Theta_1 = \{\theta:\theta < \theta_0\}$$

- 在统计学中, 这两个非空不相交的参数集合都称作 $`$`统计假设$`$`, 简称 $`$`假设$`$`
- 通过样本对一个假设作出"对"或"不对"的具体判断规则就称为该假设的一个 $`$`检验$`$` 或 $`$`检验法则$`$`
- 检验的结果若是否定该命题, 则称 $`$`拒绝这个假设$`$` , 否则就称为 $`$`接受该假设$`$`
- 若假设可用一个参数的集合表示, 该假设检验问题称为 $`$`参数假设检验$`$` , 否则称为 $`$`非参数假设检验问题$`$`, 
  上面的问题就是一个参数假设检验问题, 而对假设"总体为正太分布"作出检验的问题就是一个非参数假设检验问题

### 假设检验的基本步骤

- 建立假设
    - 原假设(零假设, null hypothesis):  `$H_0:\theta\in \Theta_0$`
    - 对立假设(备择假设, alternative hypothesis): `$H_1:\theta\in \Theta_1$`
    - 简单原假设: `$H_0:\theta = \theta_0$`
       - 双侧(双边)假设:          
          - `$H_0:\theta = \theta_0$` vs `$H_0:\theta \neq \theta_0$`
       - 单侧(单边)假设:             
          - `$H_0:\theta = \theta_0$` vs `$H_0:\theta < \theta_0$` 或者
            `$H_0:\theta = \theta_0$` vs `$H_0:\theta > \theta_0$`
    - 复杂(复合)原假设:          
       - `$H_0:\theta \geq \theta_0$` 或者 `$H_0:\theta < \theta_0$`
    - 在假设检验中, 通常将不宜轻易加以否定的假设作为原假设; 
- 选择检验统计量, 给出拒绝域形式
  - 拒绝域: 对于一个假设的检验就是指这样一个法则: 当有了具体的样本后, 
    按照该法则就可以决定是接受  `$H_0$`\还是拒绝  `$H_0$`, 
    即检验就等价于把样本空间划分为两个互不相交的部分  `$W$`\和  `$\bar{W}$`, 
    当样本属于  `$W$`\时, 拒绝  `$H_0$`\; 否则接受  `$H_0$`, 
    于是, 称  `$W$`\为该检验的拒绝域, 而称  `$\bar{W}$`\为接受域; 
  - 检验统计量: 由样本对原假设进行检验总是通过一个统计量完成的, 该统计量称为检验统计量; 
- 选择显著性水平
  - 由于样本是随机的, 当应用某种检验做判断时, 可能做出正确的判断, 也可能做出错误的判断; 
  - 两种错误: 
     - 当  `$\theta \in \Theta_0$`\时, 样本由于随机性却落入了拒绝域  `$W$`, 
       于是采取了拒绝原假设 `$H_0$` 的错误决策, 称这样的错误为第一类错误(type I error); 
     - 当  `$\theta \in \Theta_1$`\时, 样本由于随机性却落入了接受域  `$\bar{W}$`, 
       于是采取了接受原假设 `$H_0$` 的错误决策, 称这样的错误为第一类错误(type II error); 
  - 犯两种错误的概率: 由于检验结果受样本的影响, 具有随机性, 于是, 可用总体分布定义犯第一类, 第二类错误的概率如下
     - 犯第一类错误概率: `$\alpha=P_{\theta}(X \in W), \theta \in \Theta_0$`
     - 犯第二类错误概率: `$\beta=1-\alpha=P_{\theta}(X \in \bar{W}), \theta \in \Theta_1$`
  - 势函数, 功效函数(power function): 每一个检验都无法避免犯错误的可能, 
     但无法找到一个检验, 使其犯两种错误的概率都尽可能地小
     `$g(\theta) = P_{\theta}(X \in W), \theta \in \Theta_0 \cup \Theta_1$`
     显然, 势函数 `$g(\theta)$` 是定义在参数空间  `$\Theta$` 上的一个函数, 当 `$\theta\in\Theta_0$`\时, 
     `$g(\theta)=\alpha=\alpha(\theta)$`, 当  `$\theta\in\Theta_1$`\时, 
     `$g(\theta)=1-\beta=1-\beta(\theta)$`, 犯两类错误的概率都是参数  `$\theta$`\的函数: 

$$
g(\theta)=\left\{
\begin{array}{l}
\alpha(\theta)    & & {\theta\in\Theta_0}\\
1-\beta(\theta)   & & {\theta\in\Theta_1}\\
\end{array} \right.
$$

即

$$
\left\{
\begin{array}{l}
\alpha(\theta)=g(\theta)    & & {\theta\in\Theta_0}\\
\beta(\theta)= 1-g(\theta)  & & {\theta\in\Theta_1}\\
\end{array} \right.
$$

上面的函数说明, 在样本量给定的条件下, `$\alpha$`\与 `$\beta$`\中一个减小必导致另一个增大, 
既然不能同时控制一个检验的犯第一类、第二类错误的概率, 只能采取折中方案, 
通常的做法是仅限制犯第一类错误的概率, 也就是费希尔的显著性检验. 

- 显著性水平
   - 对检验问题 `$H_0:\theta\in\Theta_0$` vs `$H_1:\theta\in\Theta_1$`, 
     如果一个检验满足对任意的 `$\theta\in\Theta_0$` 都有: `$g(\theta)\leq\alpha$`, 
     则称该检验是显著性水平为 `$\alpha$` 的显著性检验, 简称水平为  `$\alpha$` 的检验. 
   - 提出显著性检验的概念就是要控制犯第一类错误的概率 `$\alpha$`, 但也不能使得 `$\alpha$` 
     过小(`$\alpha$` 过小会导致 `$\beta$` 过大), 在适当控制 `$\alpha$` 中制约 `$\alpha$`, 
     最常用的选择是 `$\alpha=0.05$`, `$\alpha=0.10$`, `$\alpha=0.01$`
- 给出拒绝域
  - `$g(\theta_0)=\alpha$`
  - 显著性水平越小, 拒绝域越小; 
- 做出判断
  - 在有了明确的拒绝域  `$W$` 后, 根据样本观测值就可以做出判断; 
  - 由样本观测值计算检验统计量, 由样本统计量是否属于拒绝域做出判断; 
  - 可能会出现这样的情况: 在一个较大的显著性水平下得到了拒绝原假设的结论, 
    而在一个较小的显著性水平下却得到了接受原假设的结论; 
  - 检验的  `$p$` 值(  `$p-value$`)
     - 在一个假设检验问题中, 利用样本观测值能够做出拒绝原假设的最小显著性水平称为检验的p值, 
       这样由检验的p值与人们心中的显著性水平  `$\alpha$` 进行比较就可以很容易做出检验的结论: 
        - 如果 `$\alpha\geq p$`, 则在显著性水平  `$\alpha$` 下拒绝原假设; 
        - 如果 `$\alpha< p$`, 则在显著性水平  `$\alpha$` 下接受原假设; 
     - 假设检验可以从两个方面进行: 其一是建立拒绝域, 考察样本观测值是否落在拒绝域中, 
       其二是根据样本观测值计算检验的p值, 通过将p值与事先设定的显著性水平进行比较

## 抽样分布

# 不确定性

## 非传递性骰子

## 医疗概率

## 混沌

## 社会选择与阿罗定理

## 纽科姆悖论

# 偏度、峰度

## 偏度

- 偏度(skewness)又称偏态、偏态系数, 是描述数据分布偏斜方向和程度的度量, 
  其是衡量数据分布非对称程度的数字特征. 
  对于随机变量 `$X$`, 其偏度是样本的三阶标准化矩:

$$Skew(x) = E[(\frac{(X-\mu)^{3}}{\sigma})] = \frac{E(X^{3})-3\mu \sigma^{2} - \mu^{3}}{\sigma^{3}}$$

- 偏度的衡量是相对于正态分布来说, 正态分布的偏度为0. 因此我们说:
  - 若数据分布是对称的, 偏度为0
  - 若偏度 > 0, 则可认为分布为右偏, 也叫正偏, 即分布有一条长尾在右
  - 若偏度 < 0, 则可认为分布为左偏, 也叫负偏, 即分布有一条长尾在左

## 峰度

- 峰度(Kurtosis)是描述数据分布陡峭或平滑的统计量, 通过对峰度的计算, 
 我们能够判定数据分布相对于正态分布而言是更陡峭还是平缓. 对于随机变量 `$X$`, 
 其峰度为样本的四阶标准中心矩
      
$$Kurt(x) = E[(\frac{(X-\mu)^{4}}{\sigma})] = \frac{E[(X-\mu)^4]}{(E[[(X-\mu)^2]])^2}$$

- 当峰度系数 > 0, 从形态上看, 它相比于正态分布要更陡峭或尾部更厚
- 峰度系数 < 0, 从形态上看, 则它相比于正态分布更平缓或尾部更薄
- 在实际环境当中, 如果一个分部是厚尾的, 这个分布往往比正态分布的尾部具有更大的"质量", 即含又更多的极端值
- 我们常用的几个分布中, 正态分布的峰度为 0, 均匀分布的峰度为 -1.2, 指数分布的峰度为 6

## Python 实现

$`$`$`python
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms
import statsmodels.formula.api as smf
from statsmodels.compat import lzip
from statsmodels.graphics.tsaplots import plot_acf

# data
df = pd.read_excel(file)

# data scatter plot 
fig, ax = plt.subplots(figsize = (8, 6))
plt.ylabel("Loss")
plt.xlabel("Distance")
plt.plot(df["distance"], df["loss"], "bo-", label = "loss")
plt.legend()
plt.show()

# Linear Regression
expr = "loss `$ distance"
results = smf.ols(expr, df).fit()
print(results.summary())

# omnibus 检验
omnibus_label = ["Omnibus K-squared test", "Chi-squared(2) p-value"]
omnibus_test = sms.omni_normtest(results.resid)
omnibus_results = lzip(omnibus_label, omnibus_test)
print(jb_results)

# jarque_bera 检验
jb_label = ["Jarque-Bear test", "Chi-squared(2) p-value", "Skewness", "Kurtosis"]
jb_test = sms.jarque_bera(results.resid)
jb_results = lzip(jb_label, jb_test)
print(jb_results)
$`$`$`

# 回归分析

## 回归分析简介


## 回归分析理论


## 回归分析实现


# 方差分析

## 方差分析简介

对于多个总体均值的比较问题, 处理这类问题通常采用所谓的方差分析方法

## 方差分析理论

### 单因子方差分析

问题描述：

通常, 在单因子试验中, 记因子为 `$A$` , 
设其有 `$r$` 个水平, 记为 `$A_{1}, A_{2}, \cdots, A_{r}$` , 
在每一个水平下考察的指标可以看成一个总体, 现有 `$r$` 个水平, 
故有 `$r$` 个总体, 假定如下，并且这三个假定都可以用统计方法进行验证:

- (1) 每一个总体均为正态总体, 记为 `$N(\mu_{i}, \sigma_{i}^{2}), i=1,2, \cdots, r$`. 利用正态性验证成立.
- (2) 各总体的方差相同, 记为 `$\sigma_{1}^{2} = \sigma_{2}^{2} = \cdots = \sigma_{r}^{2} = \sigma^{2}$`. 利用方差齐性检验验证成立.
- (3) 从每一总体中抽取的样本是相互独立的, 即所有的试验结果 `$y_{ij}$` 都相互独立. 可由随机化实现, 这里的随机化是指所有试验按随机次序进行.

接下来要做的工作就是比较各水平下的均值是否相同，即要对如下的一个假设进行检验:

$$H_{0}: \mu_{1} = \mu_{2} = \cdots = \mu_{r}.$$

其备择假设为:

$$H_{1}: \mu_{1}, \mu_{2}, \cdots, \mu_{r} 不全相等.$$

> 在不会引起误会的情况下 `$H_{1}$` 通常可省略不写.

- 问题讨论：
   - 如果 `$H_{0}$` 成立，因子 `$A$` 的 `$r$` 个水平均值相同，
     称因子 `$A$` 的 `$r$` 个水平间没有显著差异，简称因子 `$A$` 不显著;
   - 反之，当 `$H_{0}$` 不成立时，因子 `$A$` 的 `$r$` 个水平均值不全相同，
     这时称因子 `$A$` 的不同水平间有显著差异，简称因此 `$A$` 显著.
- 进行试验:
   - 为了对假设 `$H_{0}$` 进行试验，需要从每一水平下的总体抽取样本，
     设从第 `$i$` 个水平下的总体获得 `$m$` 个试验结果，
     记 `$y_{ij}$` 表示第 `$i$` 个总体的第 `$j$` 次重复试验结果，
     共得如下 `$r \times m$` 个试验结果:

$$y_{ij}, i = 1, 2, \cdots, r, j=1, 2, \cdots, m.$$

- 其中: `$r$` 为水平数，`$m$` 为重复数，`$i$` 为水平编号，`$j$` 为重复序号.
   - 在水平 `$A_{i}$` 下的试验结果 `$y_{ij}$` 与该水平下的指标均值 `$\mu_{i}$` 一般总是有差距的，
     记 `$\epsilon_{ij} = y_{ij} - \mu_{i}$` , `$\epsilon_{ij}$` 称为随机误差，于是有:

$$y_{ij} = \mu_{i} + \epsilon_{ij}$$

- 上式称为试验结果 `$y_{ij}$` 的数据结构式, 把三个假定用于数据结构式就可以写出单因子方差分析的统计模型.
- 单因子方差分析的统计模型

$$
\begin{cases}
y_{ij} = \mu_{i} + \epsilon_{ij}, i = 1, 2, \cdots, r, j=1, 2, \cdots, m. \\
诸 \epsilon_{ij} 相互独立，且都服从 N(0, \sigma^{2})
\end{cases}
$$
 
> 为了更好地描述数据，常在方差分析中引入总均值与水平效应的概念，
  称诸 `$\mu_{i}$` 的平均(所有试验结果的均值的平均)为总均值，也称一般平均。

$$\mu = \frac{1}{r}(\mu_{1} + \mu_{2} + \cdots + \mu_{r}) = \frac{1}{r}\sum_{i=1}^{r}\mu_{i}$$

称第 `$i$` 水平下的均值 `$\mu_{i}$` 与总均值 `$\mu$` 的差为因子 `$A$` 的第 `$i$` 水平的主效应，
简称为 `$A_{i}$` 的水平效应:

$$a_{i}=\mu_{i} - \mu, i = 1, 2, \cdots, r.$$

容易看出第 `$i$` 个总体均值是由总均值与该水平效应叠加而成的:

$$
\begin{cases}
\sum_{i=1}^{r}a_{i}=0 \\
\mu_{i} = \mu + a_{i}
\end{cases}
$$

$$
\begin{cases}
y_{ij} = \mu + a_{i} + \epsilon_{ij}, i = 1, 2, \cdots, r, j=1, 2, \cdots, m. \\
\sum_{i=1}^{r}a_{i}=0, \\
\epsilon_{ij} 相互独立，且都服从 N(0, \sigma^{2}).
\end{cases}
$$

- 单因子方差分析的假设

原假设:

$$H_{0}: a_{1} = a_{2} = \cdots = a_{r}.$$

其备择假设为:

$$H_{1}: a_{1}, a_{2}, \cdots, a_{r} 不全相等.$$

## 方差分析实现

