---
title: 时间序列回归模型
author: wangzf
date: '2024-03-22'
slug: forecasting-regression-model
categories:
  - timeseries
tags:
  - book
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

- [线性模型](#线性模型)
    - [简单线性回归](#简单线性回归)
    - [多元线性回归](#多元线性回归)
    - [假设条件](#假设条件)
- [最小二乘估计](#最小二乘估计)
    - [模型拟合](#模型拟合)
    - [拟合值](#拟合值)
    - [拟合优度](#拟合优度)
    - [回归的标准误差](#回归的标准误差)
- [回归模型的评估](#回归模型的评估)
    - [残差的性质](#残差的性质)
    - [残差时序图](#残差时序图)
    - [残差的自相关函数图](#残差的自相关函数图)
    - [残差直方图](#残差直方图)
    - [预测变量与残差的关系图](#预测变量与残差的关系图)
    - [拟合值与残差的关系图](#拟合值与残差的关系图)
    - [异常值点和强影响点](#异常值点和强影响点)
    - [伪回归](#伪回归)
- [回归模型特征构建](#回归模型特征构建)
    - [趋势](#趋势)
    - [虚拟变量](#虚拟变量)
    - [季节性虚拟变量](#季节性虚拟变量)
    - [干预变量](#干预变量)
    - [交易日](#交易日)
    - [分布滞后](#分布滞后)
    - [复活节](#复活节)
    - [傅里叶级数](#傅里叶级数)
- [预测变量的筛选](#预测变量的筛选)
    - [模型精度指标](#模型精度指标)
        - [调整的可决系数](#调整的可决系数)
        - [交叉检验](#交叉检验)
        - [赤池信息准则](#赤池信息准则)
        - [修正的赤池信息准则](#修正的赤池信息准则)
        - [施瓦茨的贝叶斯信息准则](#施瓦茨的贝叶斯信息准则)
        - [准则的选择](#准则的选择)
    - [最佳回归子集](#最佳回归子集)
        - [逐步回归](#逐步回归)
    - [筛选预测变量之后的推断](#筛选预测变量之后的推断)
- [回归预测](#回归预测)
    - [事前预测与事后预测](#事前预测与事后预测)
    - [基于不同情景的预测](#基于不同情景的预测)
    - [建立预测回归模型](#建立预测回归模型)
    - [预测区间](#预测区间)
- [非线性回归](#非线性回归)
    - [非线性回归模型](#非线性回归模型)
        - [对数变换](#对数变换)
        - [分段回归](#分段回归)
    - [非线性趋势预测](#非线性趋势预测)
- [相关关系、因果关系和预测](#相关关系因果关系和预测)
    - [相关性不是因果关系](#相关性不是因果关系)
    - [混淆变量](#混淆变量)
    - [多重共线性及预测](#多重共线性及预测)
        - [多重共线性问题](#多重共线性问题)
        - [多重共线性下的预测](#多重共线性下的预测)
</p></details><p></p>

线性回归模型的核心思路是：我们预测时间序列 `$y$` 时假设它与其它时间序列 `$x$` 之间存在线性关系。
例如，我们可以通过广告总花费 `$x$` 来预测月度销量 `$y$`；同样的，
我们可以通过气温数据 `$x_{1}$` 和星期数据 `$x_{2}$` 来预测日耗电量 `$y$`。

* 被预测变量 `$y$` 有时还称作回归变量、因变量或被解释变量。
* 预测变量 `$x$` 有时也叫作回归量、自变量或解释变量。这里我们称它们为“被预测变量”和“预测变量”。

# 线性模型

## 简单线性回归

最简单的线性回归模型假设被预测变量 `$y$` 和单个预测变量 `$x$` 之间存在如下线性关系：

`$$y_{t} = \beta_{0} + \beta_{1}x_{t} + \varepsilon_{t}$$`

观测值并不全部落在回归线上，而是分布在回归线的周围。我们可以这样理解：
每个观测值 `$y_{t}$` 都包含可解释部分 `$\beta_{0}+\beta_{1}x_{t}$` 和随机误差项 `$\varepsilon_{t}$`。
随机误差项并不意味着错误，而是指观测值与线性模型的偏差。
它捕捉到了除 `$x_{t}$` 外其他影响 `$y_{t}$` 的信息。

## 多元线性回归

当预测变量有两个甚至更多时，模型被称为多元线性回归模型。多元线性回归模型的一般形式如下：

`$$y_{t}=\beta_{0}+\beta_{1}x_{1，t}+\beta_{2}x_{2，t}+\cdots+\beta_{k}x_{k，t}+\varepsilon_{t}$$`

其中，`$y$` 是被预测变量，`$x_{1}，\cdots，x_{k}$` 是 `$k$` 个预测变量，
每个预测变量都必须为数值型变量。系数 `$\beta{1}，\cdots，\beta_{k}$` 分别衡量了在保持其他所有预测变量不变的情况下，
该预测变量对被预测变量的影响程度。因此，系数衡量了对应预测变量对被预测变量的边际影响。

## 假设条件

当我们想要使用线性回归模型时，需要对变量做出一些基本假设。

1. 首先，我们假设线性模型是对现实情况的合理近似；也就是说，预测变量和被预测变量之间的关系基本满足这个线性方程。
2. 其次，我们对误差项 `$(\varepsilon_{1}，\cdots，\varepsilon_{T})$` 做出如下假设：
    * 期望为零；否则预测结果会产生系统性偏差。
    * 随机误差项彼此不相关；否则预测效果会很差，因为这表明数据中尚有很多可用信息没有包含在模型中。
    * 与预测变量不相关；若误差项与预测变量相关，则表明模型的系统部分中应该包含更多信息。
    * 为了方便得到预测区间，我们还需要假设随机误差项服从方差为 `$\sigma^{2}$` 的正态分布。

线性回归模型还有一个重要的假设是 <span style='border-bottom:1.5px dashed red;'>预测变量 `$x$` 不是随机变量</span>。在进行模拟实验时，
我们可以控制每个 `$x$` 的值（所以 `$x$` 不会是随机的）并观察 `$y$` 的结果值。
但在实际生活中，我们只能得到观察数据（包括商业和经济学中的大多数数据），
而不能控制 `$x$` 的值。因此，我们需要做出如上假设。

# 最小二乘估计

## 模型拟合

在实际问题中，有一系列的观察值，
但是我们不知道模型系数 `$\beta_{0}, \beta_{1}, \cdots, \beta_{k}$` 的具体值。
因此，我们需要利用模型对这些参数进行估计。

最小二乘估计方法通过最小化残差平方和来确定模型的各个参数。
也就是说，我们通过最小化下式来确定 `$\beta_{0}, \beta_{1}, \cdots, \beta_{k}$`的估计值：

`$$\sum_{t=1}^{T}\varepsilon_{t}^{2}=\sum_{t=1}^{T}(y_{t} - \beta_{0} - \beta_{1}x_{1,t}-\beta_{2}x_{2,t} - \cdots - \beta_{k}x_{k,t})^{2}$$`

由于它的目标是最小化残差平方和，因此被称为最小二乘估计。寻找最优参数的过程，一般被称为“拟合”模型，
或者被称为模型的“学习”或者“训练”。参数估计值，
一般用 `$\hat{\beta}_{0}, \hat{\beta}_{1}, \cdots, \hat{\beta}_{k}$` 来表示。

## 拟合值

可以利用回归方程中的估计系数并将误差项设置为零来预测 `$y$`。我们通常将模型写成如下形式：

`$$\hat{y}_{t}=\hat{\beta}_{0}+\hat{\beta}_{1}x_{1,t}+\cdots+\hat{\beta}_{k}x_{k,t}$$`

将训练样本中 `$x_{1,t}, \cdots, x_{k,t}$`（其中 `$t=1,\cdots, T$`）的值代入模型中，
我们将会得到 `$y_{t}$` 的预测值，即为模型的拟合值。需要注意的是，这是模型估计得到的训练样本的预测值，
而不是 `$y$` 未来真实值的预测值。

## 拟合优度

一般用可决系数 `$R^{2}$` 评价线性回归模型对数据的拟合程度。
它可以通过计算观测值 `$y$` 和预测值 `$\hat{y}$` 之间的相关性来得出。或者，通过下式计算：

`$$R^{2}=\frac{\sum(\hat{y}_{t}-\bar{y})^{2}}{\sum(y_{t}-\bar{y})^{2}}$$`

可决系数反映了回归模型所能解释的被预测变量的变异占被预测变量总变异的比例。

在简单线性回归模型中，`$R^{2}$` 也等于 `$y$` 和 `$x$` 的相关系数的平方（假设存在截距项）。
预测值越接近于真实值，`$R^{2}$` 则会越接近于 `$1$`。
相反，若预测值和真实值不相关，则 `$R^{2}=0$` （假设存在截距项）。
在其它情况下，`$R^{2}$` 的值会处在 `$0$` 和 `$1$` 之间。

但是仅仅利用 `$R^{2}$` 来衡量模型是远远不够的。因为当增加解释变量的个数时，
`$R^{2}$` 值将会不断增加，但这并不意味着更好的模型效果。目前并不存在衡量 `$R^{2}$` 值好坏的规则， 
`$R^{2}$` 值的有效性需要视具体情况而定。
因此，利用模型在测试集上的预测结果来衡量模型好坏比直接根据 `$R^{2}$` 大小来衡量模型更加有效。

## 回归的标准误差

另外一个衡量模型拟合效果的指标是残差的标准偏差，通常称之为“残差标准误差”。
它可以通过下式来计算：

`$$\hat{\sigma}_{e}=\sqrt{\frac{1}{T-k-1}\sum_{t=1}^{T}{e_t^2}}$$`

其中：

* `$k$` 是模型中预测变量的个数
* `$T$` 是样本数量
* `$e_{t}^{2}$` 是模型拟合值 `$\hat{y}$` 和实际观测值 `$y$` 之间的差值，被称为训练误差或“残差”

需要注意的是，由于需要估计的参数个数为 `$k+1$`（截距项和 `$k$` 个解释变量），
因此上式中分母为 `$T-k-1$`。

模型的标准误差和平均误差有一定联系。我们可以将标准误与 `$y$` 的均值或标准差做对比，
得到一些关于模型精度的结论。在生成被预测变量的预测区间时，标准误差将十分有用。

# 回归模型的评估

## 残差的性质

模型拟合值 `$\hat{y}$` 和实际观测值 `$y$` 之间的差值，被称为训练误差或“残差”，其表达式如下：

`$$
\begin{align*}
  e_t &= y_t - \hat{y}_t \\
      &= y_t - \hat\beta_{0} - \hat\beta_{1} x_{1,t} - \hat\beta_{2} x_{2,t} - \cdots - \hat\beta_{k} x_{k,t}
\end{align*}$$`

其中：

* `$t=1,\cdots, T$`
* 每个残差 `$e_{t}$` 都是观测值中不可预测的部分 

残差项有两个非常有用的性质：

`$$\sum_{t=1}^{T}e_{t}=0 \quad and \quad \sum_{t=1}^{T}x_{k,t}e_{t}=0 \quad \text{for all} \quad k.$$` 

从以上两式可以明显看出：

1. 残差的均值为零；
2. 残差项和预测变量之间相关性为零。（当模型中没有截距项时，零相关假设不一定成立。）

在选择回归变量并拟合回归模型之后，有必要绘制残差图以检查模型的假设是否已经满足。
此外应该生成一系列图表，以检查拟合模型的不同方面和基本假设是否成立。下面我们将逐个分析。

## 残差时序图

残差时序图显示了不同时间下的残差的变化，如果残差存在异方差性，这种异方差性会导致预测区间的不准确。

## 残差的自相关函数图

> 残差的自相关函数，简称 ACF

对于时间序列数据而言，在当前时间段观测到的变量值很可能与历史时段的变量值很相似。
因此，当采用回归模型拟合时间序列数据时，残差经常会出现自相关效应。
此时，模型违背了残差中无序列自相关的假设，并会导致模型的预测效率低下。
为了获得更为准确的预测值，在模型中应该考虑更多的信息。当残差项存在序列自相关时，
模型的预测结果仍然是无偏的，但此时得到预测区间范围通常会比我们需要的预测区间范围更大。
因此我们应当重点关注模型残差的 ACF 图。

另一个用于检验残差自相关的效果较好的检验方法是 Breusch-Godfrey 检验，
也被称为 LM （拉格朗日乘数）检验。假如 `$p$` 值小于一个特定值（例如 0.05），
则表明残差中存在显著的自相关性。Breusch-Godfrey 检验类似于 Ljung-Box 检验，
但它是专门用于回归模型的残差检验。

## 残差直方图

检查残差是否服从正态分布也是很有必要的。正如之前我们所解释的一样，
它对预测值并不重要，但它可以让我们更加容易的确定预测区间。

残差直方图显示了残差分布是否存在左偏或右偏，如果存在，则可能影响预测区间的准确度。

## 预测变量与残差的关系图

我们期望残差是随机分布的并且不显示任何规律，
一个简单快捷的检验方法是查看每个预测变量与残差的散点图。
如果这些散点图表现出明显的规律，则该关系可能是非线性的，
并且需要相应地修改模型，可能需要使用非线性模型。

此外，还需要对没有加入到模型中的预测变量绘制其与残差的散点图。
如果某个残差图显示出明显的规律，
则需要将对应的预测变量加入到模型之中（可能以非线性形式加入）。

## 拟合值与残差的关系图

残差与拟合值之间也应没有明显规律。如果观察到明显规律，
则残差中可能存在“异方差性”，这意味着残差的方差不是固定的。
如果出现异方差性，可能需要对预测变量做对数或者平方根变换。

## 异常值点和强影响点

与大多数数据相差甚远的点被称为“异常值点”。
对模型的参数估计有重大影响的观测点被称为“强影响点”。
通常情况下，强影响点在 `$x$` 方向也是极端的异常值。

异常值的一个来源是不正确的数据录入。简单的数据描述性统计可以识别出异常的最小值和最大值。
如果识别出这样的观察结果，则应立即对样本进行校正或删除。

当某些观测点完全不同时，也会出现异常值点。在这种情况下，将这些观测点全部删除是不可取的。
如果观察结果已被确定为异常值，则必须对其进行研究并分析其背后的可能原因。
删除或保留观察可能是一个艰难决定（特别是当异常值是有影响力的观察时）。
因此，可以分别对删除观测值和保留观测值做分析。

## 伪回归

时间序列数据一般都是“不平稳的”；也就是说，时间序列数据没有固定的均值和方差。
因此我们需要解决非平稳数据对回归模型的影响，后面我们会详细讨论时间序列的平稳性。
在这里，我们需要强调非平稳数据对回归模型的影响。

不平稳的时间序列会导致伪回归。
伪回归的特点是高 `$R^{2}$` 值和高残差自相关共存。
伪回归模型似乎可以给出合理的短期预测，
但在长期时间中，伪回归是无效的。

# 回归模型特征构建

> 原来的题目：一些有用的预测变量

## 趋势

很多时间序列存在趋势。当存在简单的线性趋势时，可以直接使用 `$x_{1,t}=t$` 作为预测变量：

`$$y_{t} = \beta_{0} + \beta_{1}t + \varepsilon_{t}$$`

其中：

* `$t = 1, 2, \cdots, T$`

## 虚拟变量

目前，我们讨论的每个预测变量都是数值型变量。
但是当某个预测变量为分类变量且只有两个取值时（例如，“是”或“否”）应当怎么处理？
例如，当你想要预测日销量时，
你想把当天是否为 <span style='border-bottom:1.5px dashed red;'>法定节假日</span> 考虑进来。
此时，则需要引入一个预测变量，当天为法定节假日时该变量取值为“是”，否则取值为“否”。

在这种情况下，我们可以通过在多元模型中添加“虚拟变量”来进行处理。
当虚拟变量的取值为 1 时，代表“是”；取值为 0 时代表“否”。
虚拟变量通常也被称为“指示变量”。

虚拟变量也可以用来处理数据中的 <span style='border-bottom:1.5px dashed red;'>离群点</span> 。虚拟变量不会省略异常值，而是会消除其效果。当该观测值是离群点时，虚拟变量取值为 1，
在其他观测值处，虚拟变量取值均为 0。虚拟变量也可以表示特殊事件是否发生。

如果有两个以上的类别，则可以使用多个虚拟变量（需要注意的是，虚拟变量个数应比类别数少1）对变量进行编码。

## 季节性虚拟变量

假设我们想要预测日度数据，且想把星期数（周一、周二等等）作为预测变量。我们可以构造如下的虚拟变量：

|            | `$d_{1,t}$` | `$d_{2,t}$` | `$d_{3,t}$` | `$d_{4,t}$` | `$d_{5,,t}$` | `$d_{6,t}$`| 
|------------|-------------|-------------|-------------|-------------|--------------|------------|
| 周一        | 1           | 0           | 0           | 0           | 0            | 0          |
| 周二        | 0           | 1           | 0           | 0           | 0            | 0          |
| 周三        | 0           | 0           | 1           | 0           | 0            | 0          |
| 周四        | 0           | 0           | 0           | 1           | 0            | 0          |
| 周五        | 0           | 0           | 0           | 0           | 1            | 0          |
| 周六        | 0           | 0           | 0           | 0           | 0            | 1          |
| 周日        | 0           | 0           | 0           | 0           | 0            | 0          |
| 周一        | 1           | 0           | 0           | 0           | 0            | 0          |
| `$\cdots$` | `$\cdots$`  | `$\cdots$`  | `$\cdots$`  | `$\cdots$`  | `$\cdots$`   | `$\cdots$` |

值得注意的是，编码七个类别只需要六个虚拟变量。当所有虚拟变量都取0时，即可表示第七类（上例中的周日）。

许多初学者会给第七类添加第七个虚拟变量，这会导致模型预测变量之间出现完全共线性，
一般被称为“虚拟变量陷阱”，它会导致回归失败。因此如果定性变量有 `$m$` 个类别，
只需要引入 `$m-1$` 个虚拟变量。例如对于季度数据，需要引入 3 个虚拟变量；
对于月度数据，需要引入 11 个虚拟变量；对于日度数据，需要引入 6 个虚拟变量。

与虚拟变量相关的每个系数的解释是 该类别相对于忽略的类别对模型的影响程度 。
在上例中，“周一”的系数 `$d_{1,t}$` 即是与“周日”相比，“周一”对被预测变量的影响。

## 干预变量

建模时，我们通常需要考虑可能对被预测变量的产生影响的干预因素。
例如竞争对手的活动、广告支出、工业行动等等都会对被预测变量产生影响。

* 当干预因素的影响仅持续一个时期时，
  我们可以使用 <span style='border-bottom:1.5px dashed red;'>“尖峰”变量</span> 来描述。
  尖峰变量的处理方法和处理离群点非常相似，也是构造一个虚拟变量，
  在干预因素作用期间取值1，在其他地方取 0。
* 干预因素的影响还可能是长期或永久的。
    - 如果干预因素导致水平偏移（即序列的值从干预时间点之后突然且永久地改变），
      那么我们使用 <span style='border-bottom:1.5px dashed red;'>“阶梯”变量</span>。
      阶梯变量在干预产生之前取值为 0，从干预产生之后取值为 1。
    - 干预因素的另一种长远影响是斜率的变化。
      此时需采取 <span style='border-bottom:1.5px dashed red;'>分段处理</span>，
      在干预因素产生影响前后斜率是不同的，因此模型是非线性的。

## 交易日

一个月的交易日数可能会有很大差异，并会对销售数据产生重大影响。
为此，可以将每个月的交易日数作为预测变量。

对于月度或季度数据，可以计算出每个时期内的交易日数。可以引入 7 个解释变量，每个解释变量定义如下：

`$$x_{1} = \text{当月中周一的数目}$$`
`$$x_{2} = \text{当月中周二的数目}$$`
`$$\cdots$$`
`$$x_{7} = \text{当月中周日的数目}$$`

## 分布滞后

通常情况下，把广告支出作为解释变量会十分有效。但是，广告效应往往会具有滞后性。
因此，我们可以使用如下变量：

`$$x_{1} = \text{一个月前的广告支出}$$`
`$$x_{2} = \text{两个月前的广告支出}$$`
`$$\cdots$$`
`$$x_{m} = \text{m 个月前的广告支出}$$`

一般情况下，系数随着滞后阶数的增加而减小。

## 复活节

复活节与其他大多数的假期不同，因为它不是每年在同一天举行，并且其影响可持续一段时间。
在这种情况下，在复活节特定的时间段内，虚拟变量取值为 1，在其他时间段内取值为 0。
当复活节从 3 月开始到 4 月结束时，虚拟变量在月份之间按比例分配。

当数据为月度数据时，若复活节在三月份，那么虚拟变量在三月份时取 1；
同样的，若复活节在四月份时，虚拟变量在四月份时取 1。
当复活节从 3 月开始到 4 月结束时，虚拟变量在月份之间按比例分配。

## 傅里叶级数

对于季节性虚拟变量，尤其是长季节周期，通常可以采用傅里叶级数。
让·巴普蒂斯·约瑟夫·傅里叶是一位出生于 18 世纪的法国数学家。
他表明一定频率的一系列正弦和余弦项可以逼近任何周期函数。
我们可以把傅里叶级数用于季节模式中。

当序列的季节周期为 `$mn$` 时，其傅里叶级数的前几项为：

`$$x_{1,t} = sin\Big(\frac{2\pi t}{m}\Big)$$`
`$$x_{2,t} = cos\Big(\frac{2\pi t}{m}\Big)$$`
`$$x_{3,t} = sin\Big(\frac{4\pi t}{m}\Big)$$`
`$$x_{4,t} = cos\Big(\frac{4\pi t}{m}\Big)$$`
`$$x_{5,t} = sin\Big(\frac{6\pi t}{m}\Big)$$`
`$$x_{6,t} = cos\Big(\frac{6\pi t}{m}\Big)$$`

如果数据中存在月度季节性，那么我们使用这些预测变量中的前 11 个，
我们将得到与使用 11 个虚拟变量完全相同的预测。

当采用傅里叶级数时，尤其当 `$m$` 值很大时，
我们通常可以用较少的预测变量得到与采用虚拟变量相同的预测结果。
例如，周度数据中 `$m\approx 52$`，因此这对于周度数据非常有效。
对于短季节周期数据（例如，季度数据），使用傅里叶级数相比于季节性虚拟变量几乎没有优势。

如果仅使用前两个傅立叶项（`$x_{1,t}$` 和 `$x_{2,t}$`），
此时季节性模式将遵循简单的正弦波。由于连续的傅里叶项表示前两个傅立叶项的谐波，
因此包含傅里叶项的回归模型通常称为 <span style='border-bottom:1.5px dashed red;'>谐波回归</span>。

# 预测变量的筛选

当存在很多备选的预测变量时，我们需要从中筛选出一部分较好的预测变量供回归模型使用。

一个常见的但是不推荐的方法是画出被预测变量和特定的预测变量之间的关系图，
如果不能看出明显的相关关系，则删除该预测变量。但这个方法常常会失效，
尤其在未考虑其他预测变量时，散点图并不总能正确的反映两个变量之间的关系。

另一种常见的无效方法是对所有预测变量进行多元线性回归，并删除所有 `$p$` 值大于 0.05 的所有变量。
统计显著性并不总能表示预测变量的预测价值。因为当两个或者多个预测变量相互关联时，
`$p$` 值可能会是错误的结果。

因此，我们通过计算 <span style='border-bottom:1.5px dashed red;'>模型精度</span> 来筛选变量。

## 模型精度指标

### 调整的可决系数

在生成一个模型之后，我们通常会计算出模型的 `$R^{2}$` 值。
但是，`$R^{2}$` 并不能准确的反应模型的优劣。我们假想一个模型，
该模型的预测输出值总为真实值的 20%，
那么此时 `$R^{2}$` 值将会等于 1（仅从 `$R^{2}$` 值来看的话模型拟合效果非常好）。
但显然，这是一个极其糟糕的模型。此外，`$R^{2}$` 也没有考虑到“自由度”的影响。
在模型中增加任意一个变量（即便该变量与被预测变量无关）都会导致 `$R^{2}$` 值增大。
因此，`$R^{w}$` 值不应该作为衡量模型优劣的指标。

等效的方法是选择最小平方误差和（SSE）的模型，定义如下

`$$SSE=\sum_{t=1}^{T}e_{t}^{2}$$`

最小化 SSE 等效于最大化 `$R^{2}$`，该方法偏向于选择具有更多变量的模型，
因此也不是选择有效预测变量的方法。

采用调整的可决系数可以解决以上问题：

`$$\bar{R}^{2}=1-(1-R^{2})\frac{T-1}{T-k-1}$$`

其中：

* `$T$` 是观测点的个数
* `$k$` 是预测变量的个数

调整的可决系数 `$\bar{R}^{2}$` 是 `$R^{2}$` 的拓展，它不会随着预测变量数的增加而增加。
因此，可以选择出 `$\bar{R}^{2}$` 值最大的模型作为最优模型。
其中，最大化 `$\bar{R}^{2}$` 等同于最小化 `$\hat{\sigma}_{e}$`。
最大化 `$\bar{R}^{2}$` 在筛选变量时一般比较有效，但当变量数目太多时，效果往往比较差。

### 交叉检验

在之前的评估预测精度一节中我们曾介绍过时间序列的交叉检验，
它是用来确定模型预测能力的一个通用方法。对于回归模型，
我们也可以对选择的预测变量使用经典的留一法交叉验证。
这样可以更快且更有效地使用数据，该过程步骤如下：

1. 将 `$t$` 时刻的观测值从数据集中移出，用剩下的数据拟合出模型。
   然后计算 `$t$` 时刻观测值和预测值之间的误差 `$e_{t}^{*}=y_{t}-\hat{y}_{t}$`（此处的误差与残差不同，
   因为 `$t$` 时刻观测值的数据在估计 `$\hat{y}_{t}$` 时并没有被使用）。
2. 分别令 `$t=1,\cdots, T$`，重复步骤 1。
3. 计算 `$e_{1}^{*}, \cdots, e_{T}^{*}$` 的 MSE 。我们称之为 CV。

虽然这看起来是一个非常耗时的过程，但计算 CV 有较为快速的方法，
因此留一法交叉验证耗时相比对整个数据集拟合模型来说并不算多。
后面我们会介绍快速计算 CV 值的方法。在此标准下，最优模型是具有最小 CV 值的模型。

### 赤池信息准则

赤池信息准则也是一个重要的方法，我们通常称之为 AIC 准则，其定义如下：

`$$\text{AIC} = T\log\left(\frac{\text{SSE}}{T}\right) + 2(k+2),$$`

其中：

* `$T$` 是观测点的个数
* `$k$` 是预测变量的个数。

AIC 的计算公式中还有关于 `$k+2$` 的部分，这是因为模型中有 `$k+2$` 个待估参数：
`$k$` 个预测变量系数，截距项和残差的方差。AIC 方法的思路是添加模型参数个数的惩罚项。

在此标准下，最优模型是具有最小 AIC 值的模型。当 `$T$` 很大时，最小化 AIC 等同于最小化 CV 值。


### 修正的赤池信息准则

当 `$T$` 值较小时，AIC 准则总是倾向于选择更多的预测变量，因此出现了修正的 AIC 准则，其定义如下：

`$$\text{AIC}_{\text{c}} = \text{AIC} + \frac{2(k+2)(k+3)}{T-k-3}.$$`

与 AIC 准则相同，AICc 准则也应该被最小化。

### 施瓦茨的贝叶斯信息准则

施瓦茨的贝叶斯信息准则，通常被称为 BIC、SBIC 或 SC ，也是一个重要的方法，其定义如下：

`$$\text{BIC} = T\log\left(\frac{\text{SSE}}{T}\right) + (k+2)\log(T)$$`

与 AIC 一样，最小化 BIC 可以选择得到最佳模型。
BIC 准则选择出的模型比 AIC 准则选择出的模型具有更少的预测变量。
这是因为 BIC 准则对参数个数的惩罚力度更强。
当 `$T$` 很大时，最小化 BIC 与留 `$v$` 交叉检验非常相似 (`$v = T[1-1/(\log(T)-1)]$`)。

### 准则的选择

`$\bar{R}^{2}$` 被广泛用于模型筛选，但是由于它倾向于选择具有很多预测变量的模型，因此它并不适用于预测。

在给定足够多数据情况下，BIC 准则能找出真正完美拟合这些数据的模型，
因此统计学家更喜欢用 BIC 准则筛选模型。但是，很少有数据存在完美的模型，
即使存在，该模型的预测结果也不一定准确（因为参数估计结果可能不准确）。

因此，我们建议使用 AICc、AIC 或 CV 准则的其中一个，因为他们都是以预测数据为目标。
当 `$T$` 足够大时，它们会选择出相同的模型。

## 最佳回归子集

首先，应列出所有有可能存在的模型，并根据以上讨论的准则来选择最佳模型，
这被称为“最佳回归子集”或“所有可能回归子集”。

### 逐步回归

当预测变量的个数很多时，例如预测变量个数为 40 个，此时有 `$2^{40}$` 个备选模型，
将每个备选模型都列出来是不可能的。因此，需要其他方式来筛选模型中的预测变量。

一种高效的方法为 向后逐步回归法 ：

1. 首先用所有预测变量进行建模。
2. 每次删除一个预测变量。如果模型精度有所改进，则保留修改之后的模型。
3. 迭代直至模型精度不再提升。

当解释变量个数太多时， 向后逐步回归 效果较差，此时我们更倾向于使用 向前逐步回归。
此过程从仅包含截距的模型开始，每次添加一个预测变量，并将能提高模型预测精度的预测变量保留下来，
重复该过程直至模型精度不再提高。

对于向后逐步回归法和向前逐步回归法而言，起始模型都可以仅包含一部分预测变量。
在这种情况下， 对于向后逐步回归法，我们还应考虑在每个步骤中添加预测变量，
对于前向逐步回归法，我们还应考虑在每个步骤中删除预测变量。因此该方法也被称为混合过程。

需要注意的是，每个逐步回归法都不能得到一个最佳的模型，但是可以得到效果较好的模型。

## 筛选预测变量之后的推断

在本书中，我们不讨论预测变量的统计推断（例如查看每个预测变量的 `$p$` 值）。
如果你非常想要查看预测变量的统计显着性，请注意选择预测变量的任何过程都会使 `$p$` 值背后的假设无效。
当模型用于预测时，我们建议的用于选择预测变量的过程是有效的；
如果你希望研究任意一个预测变量对被预测变量的影响，那么我们建议的方法是无效的。

# 回归预测

可以用以下模型来预测 `$y$` 值：

`$$\hat{y_t} = \hat\beta_{0} + \hat\beta_{1} x_{1,t} + \hat\beta_{2} x_{2,t} + \cdots + \hat\beta_{k} x_{k,t}$$`

它包括估计系数并忽略了残差项。将 `$x_{1,t}, \cdots, x_{k,t}$` (其中 `$t=1, \cdots, T$`) 分别代入上式，
得到 `$y$` 的拟合值，然而我们所关注的时预测 `$y$` 的未来值。

## 事前预测与事后预测

当我们将回归模型用于时间序列数据时，根据假设的不同，模型可以生成不同类型的预测值。

* <span style='border-bottom:1.5px dashed red;'>事前预测</span> 是仅使用预先提供的信息进行预测。
    - 为了生成事前预测，模型需要预测变量的预测值。为获得预测变量的预测值，
      我们可以使用之前介绍的均值法、朴素方法、季节朴素方法、漂移法，
      或采用更复杂的纯时间序列方法，如 ARIMA。
      或者还可以采用其他来源的预测值。
* <span style='border-bottom:1.5px dashed red;'>事后预测</span> 是使用后来的预测变量信息进行的预测。
    - 例如，消费的事后预测可以使用预测变量的实际观察进行预测。
      这些不是真正的预测值，但对研究预测模型的行为很有用。

不应使用预测期内的数据估计事后预测模型。也就是说，
事后预测应该假设预测变量 `$x$` 已知，而被预测变量 `$y$` 未知。

对事前预测和事后预测的比较评估有助于区分预测不确定性的来源。
这将表明预测误差是由于预测变量的预测偏差还是由于预测模型效果不佳所引起。

## 基于不同情景的预测

在此类预测问题中，预测者对在预测变量的不同情况下模型的预测值比较关注。
例如，美国的政客可能会比较关心在失业率不发生变化的条件下，
收入和储蓄分别保持 1％ 和 0.5％ 的固定增长与分别保持 1％ 和 0.5％ 的固定下降两种情况下消费支出的变化。
需要注意的是，基于情景预测的预测值的预测区间不会包括与预测变量未来值相关的不确定性。
我们应该注意到，该方法中假设预测变量的值是事先知道的。

## 建立预测回归模型

回归模型可以清晰的表现出预测变量和被预测变量之间的关系。为了得到 <span style='border-bottom:1.5px dashed red;'>事前被预测变量的估计值</span>，
需要事先得到 <span style='border-bottom:1.5px dashed red;'>每个预测变量的未来值</span>。
那么若着重关注 <span style='border-bottom:1.5px dashed red;'>基于不同场景下的预测</span>，
则模型将十分有效。而若主要关注 <span style='border-bottom:1.5px dashed red;'>事前预测</span>，
那么建模难点可能是获得预测变量的预测值（在许多情形下，生成预测变量的预测值比生成被预测变量的预测值更具有挑战）。

令一种方法是用预测变量进行滞后预测。假设我们想要生成被预测变量的向前 `$h$` 步预测：

`$$y_{t+h} = \beta_{0}+\beta_{1}x_{1,t}+\cdots+\beta_{k}x_{k,t} + \varepsilon_{t+h}$$`


其中，`$h=1,2,\cdots$`。

由上式可看出，直接由 `$t$` 时期的各个预测变量生成 `$t+h$` 期的被预测变量 `$y$`。
因此该模型可以用来估计未来被预测变量值，即当超出观测点个数 `$T$` 时，模型仍然有效。

预测变量的滞后预测不仅使模型便于操作，而且使模型具有更加直观的吸引力。
例如，以增加产量为目的的政策变化可能不会对消费支出产生即时影响，很有可能有滞后效应。

## 预测区间

这里给出了计算简单回归的预测区间的情况，其中可以使用下面等式生成预测：

`$$\hat{y}=\hat{\beta}_{0} + \hat{\beta}_{1}x$$`


假设误差项 `$\varepsilon$` 服从正态分布，则 `$95\%$` 的预测区间为：

`$$\hat{y} \pm 1.96 \hat{\sigma}_e\sqrt{1+\frac{1}{T}+\frac{(x-\bar{x})^2}{(T-1)s_x^2}}$$`

其中：

* `$T$` 为观测点的个数
* `$\bar{x}$` 是观测样本 `$x$` 的平均值
* `$s_{x}$` 是 `$x$` 的标准差
* `$\hat{\sigma}_{e}$` 是回归标准误差

类似地，通过将 `$1.96$` 替换为 `$1.28$` 可以获得 `$80%$` 的预测区间。

上式表明，当 `$x$` 与 `$\bar{x}$` 相差较大时，生成的预测区间会更宽。
也就是说，当预测变量值接近样本均值时，模型的预测结果更可靠。

# 非线性回归

## 非线性回归模型

在此之前，我们都假设预测变量和被预测变量之间为线性关系，但是在某些情况下，
非线性函数形式可能更加符合真实情况。为了简化本节内容，我们假设只有一个预测变量 `$x$`。

一般情况下，可以通过变换预测变量 `$x$` 或被预测变量 `$y$` 来构造非线性回归模型。
虽然此时为模型的函数形式是非线性的，但模型关于参数仍然是线性的。

### 对数变换

最常见的变换方式是对变量做自然对数变换。

<span style='border-bottom:1.5px dashed red;'>双对数模型</span> 形式可写为：

`$$log y = \beta_{0} +\beta_{1} log x + \varepsilon$$`

在此模型中，斜率 `$\beta_{1}$` 可以理解为弹性：当 `$x$` 变动 `$1\%$` 时，`$y$` 的平均变动程度。

此外还有其他对数模型形式：

* <span style='border-bottom:1.5px dashed red;'>对数-线性模型</span>，
  此类模型只对被预测变量做对数变换处理； 
* <span style='border-bottom:1.5px dashed red;'>线性-对数模型</span>，
  此类模型只对预测变量进行对数变换处理；

需要注意的是，进行对数变换的前提是，所有的观测值都必须大于 0。当预测值中存在 0 值时，
我们认为对所有的观测值加 1，然后进行对数变换，即取 `$log(x+1)$`。这种变换方式具备一个良好的特点：
0 的原始比例的整齐效果，即取值为 0 的观测值在对数变换之后仍为 0(`$log(0+1)=0$`)。

### 分段回归

有些情况下，简单的将数据做对数处理是不够的，需要对数据做进一步处理。模型如下所示：

`$$y=f(x)+\varepsilon$$`

其中：`$f$` 是非线性函数。在标准线性模型中，`$f(x)=\beta_{0}+\beta_{1}x$`。
在非线性回归模型中，采用比简单对数变换更为复杂的非线性变换函数 `$f$`。

较为简单的变换方式为使 `$f$` 呈现 <span style='border-bottom:1.5px dashed red;'>分段线性</span> 的特点。
也就是说，引入使得 `$f$` 斜率发生改变的点。
我们将这些点称为 <span style='border-bottom:1.5px dashed red;'>节点</span>。
令 `$x_{1,t}=x$`，引入变量 `$x_{2,t}$`：

`$$x_{2,t}=(x-c)_{+}=\begin{cases}
0, \quad x < c \\
x-c, x \geq c
\end{cases}$$`

上述表达式表明，当 `$x-c$` 大于 0 时取其真实值，当 `$x-c$` 小于 0 时取值为 0。
这就使得斜率在 `$c$` 点发生变化，进而展现出非线性形式。

令 `$x=t$`，以下通过分段拟合的方式来拟合该时间序列。以这种方式构造的分段线性关系是 <span style='border-bottom:1.5px dashed red;'>回归样条</span>的特例。通常情况下，回归样条如下所示：

`$$x_{1}=x, x_{2}=(x-c_{1})_{+}, \cdots, x_{k}=(x-c_{k-1})_{+}$$`

其中，`$c_{1}, \cdots, c_{k-1}$` 为节点（回归线的斜率在此处会改变）。

该方法的难点是选择节点(`$k=1$`)的数量及其放置位置，某些软件中可以自动选择数量及放置位置，
但由于一些其他原因，这些算法并未被广泛使用。

使用 <span style='border-bottom:1.5px dashed red;'>三次回归样条</span> 可以得到相较于分段线性回归更为平滑的回归结果。三次回归样条中的约束是连续且平滑的（没有分段线性样条回归中，斜率突然改变的情况）。三次回归样条通常可写为：

`$$x_{1}=x, x_{2}=x^{2}, x_{3}=x^{3}, x_{4}=(x-c_{1})_{+}, \cdots, x_{k}=(x-c_{k-3})_{+}$$`

三次回归样条通常可以更好的拟合数据。但是，当 `$x$` 超出历史数据范围时，`$y$` 的预测值变得不可靠。

## 非线性趋势预测

之前介绍了通过令 `$x=t$` 拟合存在线性趋势的时间序列数据。拟合非线性趋势的最简单方法是采用二次或更高阶趋势：

`$$x_{1,t}=t, x_{2,t}=t^{2}, \cdots$$`

但是，在预测问题中，推断二阶甚至更高阶趋势是不现实的，因此不建议采用二阶甚至更高阶趋势。

采用上述介绍的分段回归，并拟合处分段的线性趋势是较好的方法。可以将非线性趋势视为线性趋势的组合。
假如在时间 `$\tau$` 处趋势发生变化，则在模型中可以简单替换上述的 `$x=t$` 和 `$c=\tau$`：

`$$x_{1,t} =t$$`

`$$x_{2,t} = (t-\tau )_{+}=\begin{cases}
0, \quad t < \tau \\
t-\tau, t \geq \tau
\end{cases}$$`

假设 `$x_{1,t}$` 和 `$x_{2,t}$` 的系数分别为 `$\beta_{1}$` 和 `$\beta_{2}$`，
那么在时间点 `$\tau$` 之前，时间序列的趋势为 `$\beta_{1}$`，而在 `$\tau$` 之后，
时间序列的趋势则变为 `$\beta_{1}+\beta_{2}$`。通过增加节点个数，
可使得时间序列趋势变化更加平滑。

# 相关关系、因果关系和预测

## 相关性不是因果关系

切记不要将相关关系和因果关系或预测与因果关系混淆。相关关系不等同于因果关系。
变量 `$x$` 可能会对预测变量 `$y$` 的值非常有用，
但这并不意味这 `$x$` 的发生导致 `$y$` 的发生。
有可能是 `$x$` 导致 `$y$`，或者是 `$y$` 导致 `$x$`，
亦或是比因果关系更复杂的关系。

例如，我们可以使用同一时期内销售的冰淇淋数量来对每个月海滩度假村的溺水次数进行回归建模。
该模型可以给出合理的预测，但不是因为冰淇淋会导致溺水，
而是因为人们在较炎热的日子里会吃更多的冰淇淋同时也会更频繁的去游泳。
因此，两个变量（冰淇淋销售量和溺水次数）是相关关系，
但其中一个变量的变化不会导致另一个变量的变化，
它们的变化都是由第三个变量（温度）的变化导致的。

我们将未包含在预测模型中的变量描述为 <span style='border-bottom:1.5px dashed red;'>混杂因素</span>，
当它同时影响响应变量和至少一个预测变量时。混淆使得很难确定哪些变量导致其他变量的变化，但不一定使预测更加困难。

同样的，通过观察早上道路上骑自行车的人数，可以预测下午是否下雨。
当骑车人比平常少时，当天晚些时候更容易下雨。该模型可以给出合理的预测，
因为当发布的天气预报是晴天时，人们更有可能骑自行车。
在这种情况下，存在因果关系。但当我们从相反的角度进行预测时，
会得到相反的结论：由于有降雨预报，骑车人数下降。
也就是说，`$x$`（降雨）会影响 `$y$`（骑自行车的人数）。

即使两个变量之间不存在因果关系，理解相关性的意义也对模型的预测会有很大帮助。
然而，当我们清楚的知道各变量之间的因果关系，我们可以建立更好的模型。
例如，更好的溺水预测模型可能包括温度和游客数量，并剔除冰淇淋销售量；
更好的降雨预测模型将排除骑自行车人数，加入前几天的天气观测数据。

## 混淆变量

假设我们使用 2000 年至 2011 年的数据预测 2012 年公司的月度销售额。在 2008 年 1 月，
一位新的竞争者进入市场并开始占据一些市场份额，与此同时，经济开始下滑。
在我们的模型中，包括竞争者活动（使用当地电视台的广告时间衡量）和经济健康状况（使用 GDP 衡量）。
但是这两个变量之间是相关的，不可能完全将这两个变量分隔开来。
当不同变量对被预测变量的影响无法分离时会产生混淆。
事实上，任何一对相关预测变量都会有一定程度的混淆，但只有当两预测变量之间的相关性较高时，
才把它们当作混淆变量处理。

由于我们无需分隔预测变量的影响而对被预测变量进行预测，因此混淆并不会严重的影响预测结果。
但混淆变量会对情景预测产生较大影响，这是因为情景预测中需要考虑变量之间的关系。
如果需要分析各个预测变量对预测结果的贡献，混淆变量也会是一个棘手的问题。

## 多重共线性及预测

### 多重共线性问题

与预测密切相关的问题是多重共线性，当多元回归中的两个或多个预测变量高度相关时就会导致模型存在多重共线性问题。

<span style='border-bottom:1.5px dashed red;'>当两个预测变量彼此高度相关时（即，它们的相关系数接近 +1 或 -1），
就会导致模型出现多重共线性</span>。在这种情况下，假如我们知道其中一个变量的信息，
那么同时我们也知道了另一个变量的大部分信息。例如，利用脚的尺寸可用于预测身高，
但在同一模型中同时引入左脚和右脚的尺寸并不会使预测更好。
<span style='border-bottom:1.5px dashed red;'>当预测变量的线性组合与预测变量的另一个线性组合高度相关时，
也可能导致多重共线性</span>。在这种情况下，由于两组预测变量提供类似的信息，
那么当我们知道第一组预测变量值的同时也会得到很多关于第二组预测变量值的信息。
在虚拟变量陷阱的例子中，若我们对季度数据引入四个虚拟变量 `$d_{1}, d_{2}, d_{3}$` 和 `$d_{4}$`。
显然 `$d_{4}=1-d_{1}-d_{2}-d_{3}$`，因此 `$d_{4}$` 和 `$d_{1}+d_{2}+d_{3}$` 之间存在完全相关性。
对于完全线性相关的情况（即相关系数为 +1 或 -1，例如虚拟变量陷阱），不能对回归模型进行估计。

当存在多重共线性时，单个预测变量的回归系数的不确定性会很大。
因此，回归系数的统计检验（例如 t 检验）是不可靠的。
此外，无法准确说明单个预测变量对被预测变量预测值的贡献。

### 多重共线性下的预测

如果未来预测变量的值超出预测变量的历史值范围，那么模型的预测结果是不可靠的。
例如，假设我们已经拟合一个回归模型，其中预测变量 `$x_{1}$` 和 `$x_{2}$` 高度相关，
并假设拟合数据中的 `$x_{1}$` 值介于 0 和 100 之间。
那么基于 `$x_{1} >100$` 和 `$x_{1}<0$` 的预测结果将会十分不可靠。
当预测变量的未来值远远超出历史范围时，尤其当当存在多重共线性时，
这会对模型的预测结果产生巨大影响，导致预测结果不准确。

需要注意，当采用质量较高的统计软件，如果对每个预测变量的具体贡献不感兴趣，
预测变量的未来值在其历史范围内，且预测变量之间不存在完全相关性，则无需过度担心多重共线性的问题。
