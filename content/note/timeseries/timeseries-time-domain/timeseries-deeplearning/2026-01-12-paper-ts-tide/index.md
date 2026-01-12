---
title: 【Paper】Time-series Dense Encoder（TiDE）：长期时间序列预测模型研究总结
author: wangzf
date: '2026-01-12'
slug: paper-ts-tide
categories:
  - timeseries
  - 论文阅读
tags:
  - paper
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
img {
    pointer-events: none;
}
</style>

<details><summary>目录</summary><p>

- [一、研究背景与动机](#一研究背景与动机)
- [二、核心模型设计（TiDE）](#二核心模型设计tide)
    - [1. 核心组件：残差块（Residual Block）](#1-核心组件残差块residual-block)
    - [2. 编码阶段（Encoding）](#2-编码阶段encoding)
    - [3. 解码阶段（Decoding）](#3-解码阶段decoding)
    - [4. 全局线性残差连接](#4-全局线性残差连接)
    - [5. 训练与评估](#5-训练与评估)
- [三、理论分析](#三理论分析)
- [四、实验结果](#四实验结果)
    - [1. 基准数据集与对比模型](#1-基准数据集与对比模型)
    - [2. 核心性能表现（MSE 指标）](#2-核心性能表现mse-指标)
    - [3. 协变量处理能力（M5 竞赛数据集）](#3-协变量处理能力m5-竞赛数据集)
    - [4. 效率优势（训练/推理速度）](#4-效率优势训练推理速度)
- [五、消融实验](#五消融实验)
- [六、结论与未来方向](#六结论与未来方向)
    - [1. 核心结论](#1-核心结论)
    - [2. 未来方向](#2-未来方向)
- [七、关键附录信息](#七关键附录信息)
- [参考](#参考)
</p></details><p></p>


# 一、研究背景与动机

1. **长期时间序列预测的重要性**：长期时间序列预测（基于长历史窗口预测未来多步）是时间序列分析的核心问题，在能源、金融、交通等领域应用广泛。
2. **现有模型的局限**：
    - 深度学习模型（如 Transformer 类模型，包括 Informer、Autoformer 等）虽曾被认为是长期预测的先进方案，但近年研究（Zeng et al., 2023）发现，简单线性模型（如 DLinear）在部分长期预测基准上可超越 Transformer，且速度更快。
    - 线性模型存在明显缺陷：无法建模时间序列中的非线性依赖关系，也难以有效处理静态/动态协变量（如节假日、商品折扣等）。
    - 部分 Transformer 改进模型（如 PatchTST）虽能提升性能，但计算复杂度高、内存消耗大，且对协变量的支持不足。
3. **研究目标**：提出一种基于多层感知机（MLP）的编码器-解码器模型 TiDE，兼顾线性模型的简洁性与速度，同时具备处理非线性依赖和协变量的能力。


# 二、核心模型设计（TiDE）

![img](images/model_arch.png)

TiDE 采用“编码-解码”架构，以 MLP 为核心组件，无自注意力、循环或卷积机制，实现对历史数据和协变量的高效处理，具体结构如下：

## 1. 核心组件：残差块（Residual Block）

- **结构**：单隐藏层 MLP（ReLU 激活）+ 全线性跳跃连接，输出端含 Dropout（防止过拟合）和 Layer Norm（稳定训练）。
- **作用**：作为编码器、解码器的基础单元，平衡模型复杂度与训练稳定性。

## 2. 编码阶段（Encoding）

目标：将历史时间序列数据与协变量映射为密集特征表示，分两步进行：
- **特征投影（Feature Projection）**：通过残差块将每个时间步的动态协变量（维度为\(r\)）降维至低维度\(\tilde{r}\)（\(\tilde{r} \ll r\)），公式为\(\tilde{x}_{t}^{(i)} = \text{ResidualBlock}(x_{t}^{(i)})\)，避免因协变量维度过高导致的计算负担。
- **密集编码器（Dense Encoder）**：将“历史时间序列（\(y_{1:L}^{(i)}\)）+ 降维后的历史/未来协变量（\(\tilde{x}_{1:L+H}^{(i)}\)）+ 静态属性（\(a^{(i)}\)）”堆叠展平后，通过多层残差块（数量为\(n_e\)，隐藏层维度为\(\text{hiddenSize}\)）生成编码向量\(e^{(i)}\)，公式为\(e^{(i)} = \text{Encoder}(y_{1:L}^{(i)} ; \tilde{x}_{1:L+H}^{(i)} ; a^{(i)})\)。

## 3. 解码阶段（Decoding）

目标：将编码向量映射为未来时间序列预测值，分两步进行：

- **密集解码器（Dense Decoder）**：通过多层残差块（数量为\(n_d\)）将编码向量\(e^{(i)}\)映射为维度为\(H \times p\)的向量\(g^{(i)}\)，再重塑为矩阵\(D^{(i)} \in \mathbb{R}^{p \times H}\)（\(H\)为预测 horizon 长度，\(p\)为解码器输出维度），每一列\(d_t^{(i)}\)对应第\(t\)个预测时间步的解码向量。
- **时间解码器（Temporal Decoder）**：通过残差块（输出维度为1）将“第\(t\)步解码向量\(d_t^{(i)}\) + 第\(t\)步降维未来协变量\(\tilde{x}_{L+t}^{(i)}\)”映射为最终预测值\(\hat{y}_{L+t}^{(i)}\)，公式为\(\hat{y}_{L+t}^{(i)} = \text{TemporalDecoder}(d_t^{(i)} ; \tilde{x}_{L+t}^{(i)})\)。该步骤为协变量搭建“直接通道”，强化关键协变量（如节假日）对预测的影响。

## 4. 全局线性残差连接

在最终预测结果中加入“历史时间序列到预测 horizon 的线性映射”，确保 TiDE 兼容线性模型（如 DLinear）的能力，即线性模型是 TiDE 的子集。

## 5. 训练与评估

- **训练方式**： mini-batch 梯度下降，损失函数为均方误差（MSE），训练集构造采用“滚动窗口”方式（所有可能的历史-未来窗口对）。
- **评估方式**：测试集采用“滚动验证”（Rolling Validation），计算 MSE、平均绝对误差（MAE）或竞赛专用指标（如 M5 的 WRMSSE）。


# 三、理论分析

针对 TiDE 的线性简化版本（所有残差连接激活、编码维度≥预测 horizon 长度），在**线性动态系统（LDS）** 假设下进行理论证明：

1. **LDS 定义**：时间序列由隐藏状态转移生成，公式为\(h_{t+1}=Ah_t+Bx_t+\eta_t\)、\(y_t=Ch_t+Dx_t+\xi_t\)（\(A\)为状态转移矩阵，\(x_t\)为协变量，\(\eta_t/\xi_t\)为噪声）。
2. **核心结论**：当 LDS 的状态转移矩阵\(A\)的最大奇异值远离1（即\(A \preccurlyeq \gamma I\)，\(\gamma < 1\)）时，TiDE 的线性简化版本可实现**近最优误差率**，且仅需较短的历史窗口（\(k = \Theta(\log(1/\varepsilon))\)，\(\varepsilon\)为误差容忍度）即可逼近最优 LDS 预测器。
3. **实验验证**：在 LDS 生成的合成数据集上，线性模型（TiDE 简化版）的 MSE（0.510±0.001）显著优于 LSTM（1.455±0.455）和 Transformer（0.731±0.041），验证了理论结论。


# 四、实验结果

## 1. 基准数据集与对比模型

- **数据集**：7个长期预测基准数据集，包括 Weather、Traffic、Electricity 及4个 ETT 系列（ETTh1/ETTh2/ETTm1/ETTm2），涵盖不同时间粒度（15分钟-1小时）和序列长度。
- **对比模型**：Transformer 类（Informer、Autoformer、FEDformer、Pyraformer、PatchTST）、线性模型（DLinear）、MLP 类（N-HiTS）、结构化状态空间模型（S4）。

## 2. 核心性能表现（MSE 指标）

TiDE 在多数数据集上实现“性能最优或与最优模型统计等效”，关键结果如下：
| 数据集       | 核心发现                                                                 |
|--------------|--------------------------------------------------------------------------|
| Traffic（最大数据集） | 所有预测 horizon（96/192/336/720）均显著优于 PatchTST，如 horizon=720 时 MSE 低10.6% |
| Weather      | horizon=96-336 时 PatchTST 最优，horizon=720 时 TiDE 最优（MSE=0.313 vs 0.314）      |
| Electricity  | 与 PatchTST 性能相当（如 horizon=720 时 MSE 均为0.196-0.197），优于 DLinear（0.203）  |
| ETT 系列      | 多数 horizon 下性能与 PatchTST 持平或更优，如 ETTh1 的 horizon=720 时 MSE=0.454（PatchTST=0.446，统计等效） |

## 3. 协变量处理能力（M5 竞赛数据集）

M5 数据集含3万+时间序列及静态属性（商品类别）、动态协变量（促销信息），TiDE 表现如下：

| 模型       | 协变量类型       | 测试集 WRMSSE（竞赛指标） |
|------------|------------------|--------------------------|
| TiDE       | 静态+动态        | 0.611±0.009              |
| TiDE       | 仅日期特征       | 0.637±0.005              |
| DeepAR     | 静态+动态        | 0.789±0.025              |
| PatchTST   | 无（不支持协变量）| 0.976±0.014              |
TiDE 利用协变量后较 DeepAR 提升20%，证明其处理复杂协变量的能力。

## 4. 效率优势（训练/推理速度）

在 Electricity 数据集上，TiDE 与 PatchTST 的效率对比（单 NVIDIA T4 GPU）：
- **推理速度**：随历史窗口（L）增大，TiDE 推理时间呈线性增长，而 PatchTST 因自注意力的二次复杂度增长更快，L=2880 时 TiDE 比 PatchTST 快5-10倍。
- **训练速度**：TiDE 单 epoch 训练时间远低于 PatchTST，且 PatchTST 在 L≥1440 时因内存不足无法运行。


# 五、消融实验

验证 TiDE 关键组件的必要性：

1. **时间解码器（Temporal Decoder）**：在含事件协变量的修改版 Electricity 数据集上，TiDE（带时间解码器）在事件后时间步的预测误差显著低于无时间解码器版本，证明其能快速捕捉协变量与预测的直接关联。
2. **残差连接**：在 Electricity 数据集上，移除所有残差连接后，TiDE（no res）的 MSE 在 horizon=96-336 时显著上升（如 horizon=96 时从0.132升至0.136），说明残差连接对稳定性能的重要性。
3. **历史窗口长度**：在 Traffic 数据集上，TiDE 的性能随历史窗口长度增加而提升（符合直觉），而部分 Transformer 模型（如 FEDformer）随窗口增大性能下降。


# 六、结论与未来方向

## 1. 核心结论

- TiDE 基于 MLP 的简单架构，在长期时间序列预测基准上实现与 Transformer 类模型相当或更优的性能，同时训练/推理速度快5-10倍，且能有效处理协变量和非线性依赖。
- 理论与实验证明，线性模型在 LDS 场景下的近最优性，为 TiDE 的有效性提供了理论支撑；自注意力机制并非长期时间序列预测的必需组件。

## 2. 未来方向

- 对 MLP 与 Transformer 在时间序列数据（如不同季节性、趋势强度）下的优缺点进行量化分析。
- 探索 TiDE 在超大规模预训练时间序列模型中的应用（需平衡参数效率与计算成本）。


# 七、关键附录信息

1. **超参数设置**：TiDE 的核心超参数及调优范围，如 hiddenSize（256-1024）、numEncoderLayers/numDecoderLayers（1-3）、dropoutLevel（0-0.5）等，不同数据集的最优超参数已明确（如 Traffic 的 hiddenSize=256，Electricity 的 hiddenSize=1024）。
2. **数据预处理**：时间协变量（如小时、星期）归一化到[-0.5, 0.5]，分类静态属性采用可学习嵌入，部分数据集使用可逆实例归一化（RevIn）稳定训练。
3. **补充对比**：TiDE 在 Weather、ETT 等数据集上显著优于 S4 模型（如 Weather 的 horizon=336 时，TiDE 的 MSE=0.254 vs S4 的 0.531）。

# 参考

1. [Long-term Forecasting with TiDE: Time-series Dense Encoder](https://arxiv.org/pdf/2304.08424)
