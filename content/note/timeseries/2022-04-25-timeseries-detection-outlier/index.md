---
title: 时间序列异常值检测与处理
author: 王哲峰
date: '2022-04-25'
slug: timeseries-outlier-detection
categories:
  - timeseries
tags:
  - ml
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

- [基于滑动窗口预测的水文时间序列异常检测](#基于滑动窗口预测的水文时间序列异常检测)
  - [水文时间序列(Hydrological Time Series)](#水文时间序列hydrological-time-series)
  - [异常值检测算法](#异常值检测算法)
</p></details><p></p>

# 基于滑动窗口预测的水文时间序列异常检测

首先基于滑动窗口对时间序列进行子序列分割, 再以子序列为基础建立预测模型对未来值进行预测, 
并将预测值和真实值之间的差异范围大于预设阈值的序列点判定为异常.
探讨了算法中的滑动窗口和参数设置, 并以实例数据对算法进行了验证. 
实验结果表明, 所提算法不仅能够有效挖掘出水文时间序列中的异常点,
 而且将异常检测的灵敏度和特异度分别提高到80%和98以上. 

异常检测(Outlier Detection)
也称为异常挖掘或异常检测, 是从大量数据中提取隐含在其中的人们事先不知道的但又是潜在的有用的信息和知识的过程. 

- 异常检测需要解决两个主要问题: 
- 在给定的数据集合中定义什么样的数据是异常的
- 找到一个有效的方法来检测这样的异常数据
- 按异常在时间序列中的不同表现形式, 时间序列异常可以分为3种: 
- 序列异常
- 点异常
- 模式异常
- 时间序列异常检测方法主要包括: 
  1. 基于窗口的方法
  2. 基于距离的方法
  3. 基于密度的方法
  4. 基于支持向量机的方法
  5. 基于聚类的方法

## 水文时间序列(Hydrological Time Series)

**数据:**

$$D^{n} = < d_1 = (v_1, t_1), d_2 = (v_2, t_2), \cdots, d_n = (v_n, t_n) >$$

| value          | timestamp      |
|----------------|----------------|
| $v_1`    | $t_1`    |
| $v_2`    | $t_2`    |
| $\cdots` | $\cdots` |
| $v_n`    | $t_n`    |

**异常值定义: **

点 $d_{i}` 的 k-最近邻窗口: 

$$\eta_{t_i}^{<k\>}=\\{ d_{i-2k}, d_{i-2k+1}, \cdots, d_{i-1} \\}$$

k-最近邻窗口观测值集合: 

$\\{v_{i-2k}, v_{i-2k+1}, \cdots, v_{i-1}\\}$

| value        | timeseries   |
|--------------|--------------|
| $v_{i-2k}`   | $t_{i-2k}`   |
| $v_{i-2k+1}` | $t_{i-2k+1}` |
| $\cdots`     | $\cdots`     |
| $v_{i-1}`    | $t_{i-1}`    |
| $v_i`        | $t_i`        |

若点 $d_{i}$ 的实际观测值和依据其
k-最近邻窗口模型预测值之间的差值超过某一特定阈值
$\tau$ , 则判断该点为异常点. 


## 异常值检测算法

