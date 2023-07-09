---
title: 统计假设检验
author: 王哲峰
date: '2023-07-08'
slug: python-stats-hypothetical-test
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

- [假设检验](#假设检验)
  - [统计显著性](#统计显著性)
  - [假设检验的基本思想](#假设检验的基本思想)
  - [假设检验的基本步骤](#假设检验的基本步骤)
    - [建立假设](#建立假设)
    - [选择检验统计量并给出拒绝域形式](#选择检验统计量并给出拒绝域形式)
    - [选择显著性水平](#选择显著性水平)
  - [抽样分布](#抽样分布)
- [常用假设检验](#常用假设检验)
  - [正态性检验](#正态性检验)
    - [Shapiro-Wilk Test](#shapiro-wilk-test)
    - [D'Agostino's K2 Test](#dagostinos-k2-test)
- [Anderson-Darling Test](#anderson-darling-test)
  - [相关性检验](#相关性检验)
    - [Pearson 相关系数](#pearson-相关系数)
  - [参数检验](#参数检验)
  - [非参数检验](#非参数检验)
  - [时间序列平稳性检验](#时间序列平稳性检验)
  - [Q-Q 图](#q-q-图)
- [参考](#参考)
</p></details><p></p>

# 假设检验

假设检验(Hypothesis Test)是由 K.Pearson 于 20 世纪初提出的，
之后由费希尔进行了细化，并最终由 Neyman 和 E.Pearson 提出了较完整的假设检验理论。

假设检验作为统计学的一项重要模块，是用来判断样本与总体的差异是由抽样误差引起还是本质差别造成的统计推断方法。

假设检验需要事先对总体参数或分布形式做出某种假设，然后利用样本信息来判断原假设是否成立。
一般假设检验分为参数检验和非参数检验，其运用逻辑上的反证法，依据统计上的小概率原理来实现，
假设检验分为原假设 `$H0$` 和备择假设 `$H1$`

## 统计显著性

显著性、统计显著性(Statistical Significance)是指零假设为真的情况下拒绝零假设所要承担的风险水平，
又叫概率水平，或者显著水平(Significant Level)。

显著性的含义是指两个群体的态度之间的任何差异是由于系统因素而不是偶然因素（采样等）的影响。
假定控制了可能影响两个群体之间差异的所有其他因素，因此，余下的解释就是所推断的因素，
而这个因素不能够 100% 保证，所以有一定的概率值，叫显著性水平。

总的来说，它表示群体之间得以相互区别的能力。在统计假设检验中，公认的小概率事件的概率值被称为统计假设检验的显著性水平，
对同一量，进行多次计量，然后算出平均值。

## 假设检验的基本思想

假如实验结果与原假设 `$H$` 发生矛盾就拒绝原假设 `$H$`，否则就接受原假设 `$H$`。

假如对某个指标 `$\theta$` 进行检测，检测的临界值为 `$\theta_0$`，
即如果 `$\theta \geq \theta_0$` 则认为指标 `$\theta$` 合格；
如果 `$\theta < \theta_0$`，则认为指标 `$\theta$` 不合格。

因此命题 "`$\theta \geq \theta_0$`" 将涉及如下两个参数集合: 

`$$\Theta_0 = \{\theta:\theta \geq \theta_0\}$$`

`$$\Theta_1 = \{\theta:\theta < \theta_0\}$$`

在统计学中，这两个非空不相交的参数集合都称作统计假设，简称假设。
通过样本对一个假设作出"对"或"不对"的具体判断规则就称为该假设的一个检验或检验法则。

检验的结果若是否定该命题，则称拒绝这个假设，否则就称为接受该假设。
若假设可用一个参数的集合表示，该假设检验问题称为参数假设检验，
否则称为非参数假设检验问题，上面的问题就是一个参数假设检验问题，
而对假设"总体为正态"作出检验的问题就是一个非参数假设检验问题。

## 假设检验的基本步骤

### 建立假设

假设的基本形式：

* 原假设(零假设，null hypothesis)：`$H_0:\theta\in \Theta_0$`
* 对立假设(备择假设，alternative hypothesis)：`$H_1:\theta\in \Theta_1$`

假设的类别：

* 简单原假设：`$H_0:\theta = \theta_0$`
    - 双侧(双边)假设
        - `$H_0:\theta = \theta_0$` vs `$H_0:\theta \neq \theta_0$`
    - 单侧(单边)假设
        - `$H_0:\theta = \theta_0$` vs `$H_0:\theta < \theta_0$` 或者
          `$H_0:\theta = \theta_0$` vs `$H_0:\theta > \theta_0$`
- 复杂(复合)原假设
    - `$H_0:\theta \geq \theta_0$` 或者 `$H_0:\theta < \theta_0$`

在假设检验中，通常将不宜轻易加以否定的假设作为原假设 

### 选择检验统计量并给出拒绝域形式

拒绝域：对于一个假设的检验就是指这样一个法则：当有了具体的样本后，
按照该法则就可以决定是接受  `$H_0$` 还是拒绝 `$H_0$`，
即检验就等价于把样本空间划分为两个互不相交的部分 `$W$` 和 `$\bar{W}$`，
当样本属于 `$W$` 时，拒绝  `$H_0$`；否则接受 `$H_0$`，
于是，称 `$W$` 为该检验的拒绝域，而称 `$\bar{W}$` 为接受域 

检验统计量：由样本对原假设进行检验总是通过一个统计量完成的，该统计量称为检验统计量 

### 选择显著性水平

由于样本是随机的，当应用某种检验做判断时，可能做出正确的判断，也可能做出错误的判断 

* 两种错误
    - 当 `$\theta \in \Theta_0$` 时，样本由于随机性却落入了拒绝域 `$W$`，
      于是采取了拒绝原假设 `$H_0$` 的错误决策，称这样的错误为第一类错误(type I error)
    - 当 `$\theta \in \Theta_1$` 时，样本由于随机性却落入了接受域  `$\bar{W}$`， 
      于是采取了接受原假设 `$H_0$` 的错误决策，称这样的错误为第一类错误(type II error)
* 犯两种错误的概率：由于检验结果受样本的影响，具有随机性，于是，可用总体分布定义犯第一类，第二类错误的概率如下
    - 犯第一类错误概率：`$\alpha=P_{\theta}(X \in W), \theta \in \Theta_0$`
    - 犯第二类错误概率：`$\beta=1-\alpha=P_{\theta}(X \in \bar{W}), \theta \in \Theta_1$`
- 势函数、功效函数(power function)：每一个检验都无法避免犯错误的可能，但无法找到一个检验，使其犯两种错误的概率都尽可能地小 

  `$$g(\theta) = P_{\theta}(X \in W), \theta \in \Theta_0 \cup \Theta_1$$`

  显然，势函数 `$g(\theta)$` 是定义在参数空间  `$\Theta$` 上的一个函数，当 `$\theta\in\Theta_0$` 时，
  `$g(\theta)=\alpha=\alpha(\theta)$`，当  `$\theta\in\Theta_1$` 时，
  `$g(\theta)=1-\beta=1-\beta(\theta)$`，犯两类错误的概率都是参数  `$\theta$` 的函数: 

`$$g(\theta)=\left\{
\begin{array}{l}
\alpha(\theta)    & & {\theta\in\Theta_0}\\
1-\beta(\theta)   & & {\theta\in\Theta_1}\\
\end{array} \right.$$`

即

`$$\left\{
\begin{array}{l}
\alpha(\theta)=g(\theta)    & & {\theta\in\Theta_0}\\
\beta(\theta)= 1-g(\theta)  & & {\theta\in\Theta_1}\\
\end{array} \right.$$`

上面的函数说明，在样本量给定的条件下，`$\alpha$` 与 `$\beta$` 中一个减小必导致另一个增大，
既然不能同时控制一个检验的犯第一类、第二类错误的概率，只能采取折中方案，
通常的做法是仅限制犯第一类错误的概率，也就是费希尔的显著性检验. 

* 显著性水平
    - 对检验问题 `$H_0:\theta\in\Theta_0$` vs `$H_1:\theta\in\Theta_1$`，
      如果一个检验满足对任意的 `$\theta\in\Theta_0$` 都有: `$g(\theta)\leq\alpha$`，
      则称该检验是显著性水平为 `$\alpha$` 的显著性检验，简称水平为  `$\alpha$` 的检验. 
    - 提出显著性检验的概念就是要控制犯第一类错误的概率 `$\alpha$`，但也不能使得 `$\alpha$` 
      过小(`$\alpha$` 过小会导致 `$\beta$` 过大)，在适当控制 `$\alpha$` 中制约 `$\alpha$`，
      最常用的选择是 `$\alpha=0.05$`，`$\alpha=0.10$`，`$\alpha=0.01$`
* 给出拒绝域
    - `$g(\theta_0)=\alpha$`
    - 显著性水平越小，拒绝域越小； 
* 做出判断
    - 在有了明确的拒绝域  `$W$` 后，根据样本观测值就可以做出判断； 
    - 由样本观测值计算检验统计量，由样本统计量是否属于拒绝域做出判断； 
    - 可能会出现这样的情况: 在一个较大的显著性水平下得到了拒绝原假设的结论，
      而在一个较小的显著性水平下却得到了接受原假设的结论； 
    - 检验的 `$p$` 值(`$p-value$`)
        - 在一个假设检验问题中，利用样本观测值能够做出拒绝原假设的最小显著性水平称为检验的 `$p$` 值，
          这样由检验的 `$p$` 值与人们心中的显著性水平  `$\alpha$` 进行比较就可以很容易做出检验的结论: 
            - 如果 `$\alpha\geq p$`，则在显著性水平  `$\alpha$` 下拒绝原假设； 
            - 如果 `$\alpha< p$`，则在显著性水平  `$\alpha$` 下接受原假设； 
        - 假设检验可以从两个方面进行: 其一是建立拒绝域，考察样本观测值是否落在拒绝域中，
        其二是根据样本观测值计算检验的 `$p$` 值，通过将 `$p$` 值与事先设定的显著性水平进行比较

## 抽样分布

# 常用假设检验

## 正态性检验

* Shapiro-Wilk Test (W 检验)
* D'Agostino's K2 Test (Normal Test)
* Anderson-Darling Test

### Shapiro-Wilk Test

Shapiro-Wilk Test 用于检验样本数据是否来自服从某个正态分布的总体。
在实际应用中，Shapiro-Wilk Test 被认为是一个可靠的正态性检验，
但是也有人认为该检验更适用于较小的数据样本（数千个观测值以内）。

* 使用前提
    - 各样本观测值为独立同分布的(iid)
* 原假设
    - 样本数据服从正态分布
* 结果解释
    - 当 `$p$` 值小于某个显著性水平 `$\alpha$`，则认为样本不是来自正态分布的总体，否则承认样本来自正态分布的总体
* 方法库
    - `from scipy.stats import shapiro`

```python
from scipy.stats import shapiro

# 置信水平
alpah = 0.05
# 数据
data = [
    0.873, 2.817, 0.121, -0.945, -0.055, 
    -1.436, 0.360, -1.478, -1.637, -1.869,
]
# 假设检验
stat, p = shapiro(data)
print(f"stat={stat}%.3f, p={p}%.3f")

# 做出决定
if p > alpah:
    print("不能拒绝原假设，样本数据服从正态分布")
else:
    print("拒绝原假设，样本数据不服从正态分布")
```

### D'Agostino's K2 Test

D'Agostino's K2 Test，用于检验样本数据是否来自服从正态分布的总体。是通过计算样本数据的峰度和偏度，
来判断其分布是否偏离正态分布。偏度是对数据分布对称性的测度，衡量数据分布是否左偏或右偏。
峰度是对数据分布平峰或尖峰程度的测度，它是一种简单而常用的正态性统计检验量。

* 使用前提
    - 各样本观测值为独立同分布的(iid)
* 原假设
    - 样本数据服从整天分布
* 结果解释
    - 当 `$p$` 值小于某个显著性水平 `$\alpha$`，则认为样本不是来自正态分布的总体，否则承认样本来自正态分布的总体
* 方法库
    - `from scipy.stats import normaltest`

```python
from scipy.stats import normaltest

# 执行水平
alpha = 0.05
# 数据
data = [
    0.873, 2.817, 0.121, -0.945, -0.055, 
    -1.436, 0.360, -1.478, -1.637, -1.869,
]
# 假设检验
stat, p = normaltest(data)
print(f"stat={stat}, p={p}")

if p > alpha:
    print("不能拒绝原假设，样本数据服从正态分布")
else:
    print("拒绝原假设，样本数据不服从正态分布")
```

# Anderson-Darling Test

Anderson-Darling Test，用于检验样本数据是否服从某一已知分布。
该检验修改自一种更复杂的非参数的拟合良好的检验统计（Kolmogorov-Smirnov Test）。
SciPy 中的 `anderson()` 函数实现了Anderson-Darling Test，函数参数为样本数据及要检验的分布名称，
默认情况下，为 `'norm'`正态分布，还支持对 `'expon'` 指数分布、`'logistic'` 分布，以及 `'gumbel'` 耿贝尔分布的检验，
它会返回一个包含不同显著性水平下的p值的列表，而不是一个单一的 `$p$` 值，因此这可以更全面地解释结果

* 使用前提
    - 各样本观测值为独立同分布的(iid)
* 原假设
    - 样本数据服从某一已知分布
* 结果解释
    - 当 `$p$` 值小于某个显著性水平 `$\alpha$`，则认为样本不是来自某一分布的总体，否则承认样本来自该分布的总体
* 方法库
    - `from scipy.stats import anderson`

```python
from scipy.stats import anderson

# 显著性水平
alpha = 0.05
# 样本数据
data = [
    0.873, 2.817, 0.121, -0.945, -0.055, 
    -1.436, 0.360, -1.478, -1.637, -1.869
]
# 假设检验
result = anderson(data)
pritn(f"stat={result.statistic}, p={result.critical_values}")

for i in range(len(result.critical_values)):
    sl, cv = result.significance_level[i], result.critical_values[i]
    if result.statistic < result.critical_values[i]:
        print(f"显著性水平为 {sl/100:.3f}时，p 值为 {cv:.3f}，不能拒绝原假设，样本数据服从正态分布")
    else:
        print(f"显著性水平为 {sl/100:.3f}时，p 值为 {cv:.3f}，拒绝原假设，样本数据不服从正态分布")
```

## 相关性检验

* Pearson 相关系数
* Spearman 等级相关系数
* Kendall 等级相关系数
* Chi-square Test (卡方检验)

### Pearson 相关系数

Pearson 相关系数，用于检验两样本数据之间线性关系的强度。
该检验将两个变量之间的协方差进行归一化处理以给出可解释的分数，为一个介于 -1 到 1 之间的值，
-1 表示完全负相关，1 表示完全正相关，0 表示没有相关性。

* 使用前提
    - 各样本观测值为独立同分布的(iid)
    - 样本数据服从正态分布
    - 每个样本观测值的方差相等
    - 所有变量都是连续变量
* 原假设
    - 两个变量相互独立（或不相关）
* 结果解释
    - 当 `$p$` 值小于某个显著性水平 `$\alpha$`，则拒绝原假设，认为两个变量是相关的。
      否则认为是不相关的。这里的相关仅为统计学意义上的相关性，并不能理解为实际因果关系
* 方法库
    - `from scipy.stats import anderson`



## 参数检验

* 单样本 t 检验
* 两样本 t 检验 (方差齐性使用 Levene 检验)
* 配对样本 t 检验
* 方差分析(Anasis Of Variance, ANOVA)

## 非参数检验

* Mann-Whitney U Test (曼惠特尼 U 检验)
* Wilcoxon Signed-Rank Test
* Kruskal-Wallis H Test (K-W 检验)
* Friedman Test (弗里德曼检验)
* Kolmogorv-Smirnov Test (K-S 检验)

## 时间序列平稳性检验

* Augmented Dickey-Fuller Unit Root Test (单位根检验)
* Kwiatkowski-Phillips-Schmidt-Shin Test

## Q-Q 图





# 参考

* [scipy 正态性检验]https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html
