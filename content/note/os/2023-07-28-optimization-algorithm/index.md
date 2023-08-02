---
title: 最优化算法
author: 王哲峰
date: '2023-07-28'
slug: optimization-algorithm
categories:
  - algorithms
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

- [规划论](#规划论)
- [库存论](#库存论)
- [图论](#图论)
- [排队论](#排队论)
- [可靠性理论](#可靠性理论)
- [对策论](#对策论)
- [决策论](#决策论)
- [搜索论](#搜索论)
</p></details><p></p>

最优化算法，即最优计算方法，也是运筹学。最优化同运筹学一样，是利用现代数学、系统科学、计算机科学及其他学科的最新成果，
来研究人类从事的各种活动中处理事务的数量化规律，使有限的人、物、财、时空、信息等资源得到充分和合理的利用，
以期获得尽可能满意的经济和社会效果。

最优化算法的内容包括：

> * 规划论
>     - 线性规划
>     - 非线性规划
>     - 整数规划
>     - 动态规划
> * 库存论
> * 图论
> * 排队论
> * 可靠性理论
> * 对策论
> * 决策论
> * 搜索论

# 规划论

<span style='border-bottom:1.5px dashed red;'>规划论（数学规划）</span> 是运筹学的一个重要分支，
早在 1939 年苏联的康托罗维奇和美国的希奇柯克等人就在生产组织管理和编制交通运输方案时研究和应用了线性规划的方法。
1947 年美国的旦茨格等人提出了求解线性规划问题的<span style='border-bottom:1.5px dashed red;'>单纯形法</span>，
为线性规划的理论与计算奠定了基础。
<span style='border-bottom:1.5px dashed red;'>非线性规划</span>的基础性工作是在 1951 年由库恩和塔克等人完成的，
到了 20 世纪 70 年代，数学规划无论是在理论上还是方法上，还是在应用的深度和广度上都得到了进一步的发展。

数学规划的研究对象是计划管理工作中有关<span style='border-bottom:1.5px dashed red;'>安排</span>和<span style='border-bottom:1.5px dashed red;'>估值</span>的问题，即在给定条件下，
按某个衡量指标来寻找安排的最佳方案。它可以表示为求函数在满足约束条件下的极大值或极小值问题。

数学规划中最简单的一类问题就是<span style='border-bottom:1.5px dashed red;'>线性规划</span>。
如果约束条件和目标函数都属于线性关系就叫线性规划。要解决线性规划问题，从理论上讲要解线性方程组。

<span style='border-bottom:1.5px dashed red;'>非线性规划</span>是线性规划的进一步发展和延伸。
非线性规划扩大了数学规划的应用范围，同时也给数学工作者提出了许多基本的理论问题，是数学中的如凸分析、数值分析等也得到了发展。

还有一种规划问题和时间有关，即<span style='border-bottom:1.5px dashed red;'>动态规划</span>，
它已经成为在工程控制、技术物理和通信中最佳控制问题的重要工具。

# 库存论

<span style='border-bottom:1.5px dashed red;'>库存论（存贮论）</span>是运筹学中发展较早的分支。
早在 1915 年，哈里斯就针对银行货币的储备问题进行了详细的研究，建立了一个确定性的存贮费用模型，并求得了 <span style='border-bottom:1.5px dashed red;'>最佳批量公式</span>。
1934 年，威尔逊重新得出<span style='border-bottom:1.5px dashed red;'>经济订购批量公式（EOQ 公式）</span>。

物资的存贮按其目的的不同可分为以下三种：

1. 生产存贮
    - 它是企业为了维持正常生产而储备的原材料或半成品
2. 产品存贮
    - 它是企业为了满足其他部门的需要而存贮的半成品或成品
3. 供销存贮
    - 它是指存贮在供销部门的各种物资，可直接满足顾客的需要

库存论中研究的主要问题可以概括为<span style='border-bottom:1.5px dashed red;'>何时订货（补充存贮）</span>和<span style='border-bottom:1.5px dashed red;'>每次订多少货（补充多少库存）</span>这两个问题。

# 图论

图论既是拓扑学的一个分支，也是运筹学的重要分支，它是建立和处理离散数学模型的有用工具。

# 排队论

排队论（随机服务系统理论）是在 20 世纪由丹麦工程师爱尔朗对电话交换机的效率研究开始的，
在第二次世界大战中为了对飞机跑道的容纳量进行估算，该理论得到了进一步的发展，其相应的学科更新论、可靠性理论等也发展了起来。

排队论主要研究各种系统的排队长度、排队的等待时间及所提供的服务等各种参数，以便求得更好的服务，它是研究系统随机聚散现象的理论。
排队论的研究目的是要回答如何改进服务机构或组织所服务的对象，使某种指标达到最优的问题。

因为排队现象是一个随机现象，因此在研究排队现象时，主要采用将研究随机现象的概率论作为主要工具。此外，还涉及微分和微分方程的相关内容。

# 可靠性理论

可靠性理论是研究系统故障，以提高系统可靠性问题的理论。可靠性理论研究的系统一般分为以下两类：

1. 不可修复系统：这种系统的参数是寿命、可靠度等，如导弹。
2. 可修复系统：这种系统的重要参数是有效度，其值为系统的正常工作时间与正常工作时间加上事故修理时间之比，如一般的机电设备等。

# 对策论

对策论（博弈论）是指研究多个个体或团队之间在特定条件制约下的对局中，利用相关方的策略而实施对应策略的学科，
如田忌赛马、智猪博弈就是经典的博弈论问题。它是应用数学的一个分支，既是现代数学的一个新分支，也是运筹学的一个重要学科。

# 决策论

决策论是研究决策问题的，所谓<span style='border-bottom:1.5px dashed red;'>决策</span>就是<span style='border-bottom:1.5px dashed red;'>根据客观可能性，
借助一定的理论、方法和工具，科学地选择最优方案的过程</span>。决策问题由 <span style='border-bottom:1.5px dashed red;'>决策者</span>和 <span style='border-bottom:1.5px dashed red;'>决策域</span>构成，而决策域则由<span style='border-bottom:1.5px dashed red;'>决策空间、状态空间和结果函数</span>构成。研究决策理论与方法的科学就是决策科学。

决策所要解决的问题是多种多样的，不同角度有不同的分类方法。按决策者所面临的自然状态的确定与否可分为<span style='border-bottom:1.5px dashed red;'>确定型决策</span>、<span style='border-bottom:1.5px dashed red;'>不确定型决策</span>和<span style='border-bottom:1.5px dashed red;'>风险型决策</span>，
按决策所依据的目标个数可分为 <span style='border-bottom:1.5px dashed red;'>单目标决策</span>与 <span style='border-bottom:1.5px dashed red;'>多目标决策</span>，按决策问题的性质可分为 <span style='border-bottom:1.5px dashed red;'>战略决策</span>与 <span style='border-bottom:1.5px dashed red;'>策略决策</span>，以及按不同准则划分成的种种决策问题类型。
不同类型的决策问题应采用不同的决策方法。

决策的基本步骤如下：

1. 确定问题，提出决策的目标；
2. 发现、探索和拟定各种可行方案；
3. 从多种可行方案中，选出最佳方案；
4. 决策的执行与反馈，以寻求决策的动态最优。

如果对方决策者也是人（一个人或一群人），双方都希望取胜，这类具有竞争性的决策称为 <span style='border-bottom:1.5px dashed red;'>对策或博弈型决策</span>。
构成决策问题的三个根本问题是：<span style='border-bottom:1.5px dashed red;'>局中人</span>、<span style='border-bottom:1.5px dashed red;'>策略</span>和 <span style='border-bottom:1.5px dashed red;'>一局对策的得失</span>。对策问题按照局中人数分类可分成两人对策和多人对策，
按局中人赢得函数的代数和是否为零可分成零和对策和非零和对策，按解的表达形式可分成纯策略对策和混合策略对策，按问题是否静态形式可分成动态对策和静态对策。

# 搜索论

搜索论主要研究在资源和探测手段受到限制的情况下，如何设计寻找某种目标的最优方案，并加以实施的理论和方法。

在第二次世界大战中，同盟国的空军和海军在研究如何针对轴心国的潜艇活动、舰队运输和兵力部署等进行甄别的过程中产生的。
搜索论在实际应用中也取得了不少成效，如 20 世纪 60 年代，美国寻找在大西洋失踪的核潜艇“蝎子号”，以及在地中海寻找丢失的氢弹，
都是依据搜索论获得成功的。
