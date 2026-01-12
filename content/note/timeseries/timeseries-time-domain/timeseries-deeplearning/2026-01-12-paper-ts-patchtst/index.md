---
title: 【Paper】PatchTST：A Time Series is Worth 64 Words：Long-Term Forecasting with Transformers
author: wangzf
date: '2026-01-12'
slug: paper-ts-patchtst
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

- [一、论文基本信息](#一论文基本信息)
- [二、研究背景与动机](#二研究背景与动机)
- [三、核心方法：PatchTST 模型设计](#三核心方法patchtst-模型设计)
    - [3.1 核心组件设计](#31-核心组件设计)
        - [1. 时间序列分块（Patching）](#1-时间序列分块patching)
        - [2. 通道独立（Channel-independence）](#2-通道独立channel-independence)
    - [3.2 模型整体结构](#32-模型整体结构)
- [四、实验设计与结果](#四实验设计与结果)
    - [4.1 实验设置](#41-实验设置)
    - [4.2 核心实验结果](#42-核心实验结果)
        - [1. 多变量长期预测（监督学习）](#1-多变量长期预测监督学习)
        - [2. 自监督表示学习](#2-自监督表示学习)
        - [3. 迁移学习](#3-迁移学习)
        - [4. 消融实验](#4-消融实验)
- [五、结论与未来工作](#五结论与未来工作)
    - [5.1 核心结论](#51-核心结论)
    - [5.2 未来方向](#52-未来方向)
- [六、附录关键补充](#六附录关键补充)
- [参考](#参考)
</p></details><p></p>

# 一、论文基本信息

- **发表会议**：ICLR 2023
- **作者团队**：Yuqi Nie（普林斯顿大学）、Nam H. Nguyen、Phanwadee Sinthong、Jayant Kalagnanam（IBM 研究院）
- **核心主题**：提出基于 Transformer 的高效模型 PatchTST，用于多变量时间序列预测和自监督表示学习，
  解决传统 Transformer 在时间序列预测中的计算复杂、内存消耗大及长历史依赖捕捉不足等问题。

# 二、研究背景与动机

1. **时间序列预测的重要性**：预测是时间序列分析的核心任务，深度学习模型（如 Transformer）在该领域应用广泛，但面临挑战。
2. **传统 Transformer 的局限性**：
    - 计算与内存复杂度高：原始 Transformer 注意力机制复杂度为 `$O(N^2)$`（`$N$` 为输入 token 数），
      当输入序列长度 `$L$` 大时，`$N=L$`，导致计算瓶颈。
    - 局部语义信息缺失：多数模型使用“逐点输入 token”，无法像自然语言中“单词”那样捕捉局部语义，难以分析时间步间的关联。
    - 通道混合设计缺陷：传统多变量时间序列模型常采用“通道混合”（输入 token 包含多通道信息），需更多数据学习跨通道关联，易过拟合且泛化性差。
3. **关键挑战回应**：针对 Zeng 等人（2022）提出“简单线性模型优于 Transformer”的观点，
   本文通过 PatchTST 验证 Transformer 在时间序列预测中的有效性。

# 三、核心方法：PatchTST 模型设计

PatchTST 的核心是 **时间序列分块（Patching）** 与 **通道独立（Channel-independence）** 两大组件，
结合 Transformer 编码器实现高效预测。

![img](images/model_arch.png)

## 3.1 核心组件设计

### 1. 时间序列分块（Patching）

- **原理**：将单变量时间序列按“块长度 `$P$`”和“步长 `$S$`”划分为重叠或非重叠的子序列块（Patch），
  每个 Patch 作为 Transformer 的输入 token。
  - 块数量计算：`$N=\left\lfloor\frac{(L-P)}{S}\right\rfloor+2$`（`$L$` 为回溯窗口长度，末尾补 `$S$` 个最后值以保证完整性）。
- **三大优势**：
  - 保留局部语义：聚合多个时间步信息，捕捉逐点输入无法获得的局部关联（如短期趋势、周期性片段）。
  - 降低复杂度：输入 token 数从 `$L$` 降至 `$L/S$`，注意力计算与内存消耗呈二次方降低（如 `$L=336$`、`$S=8$` 时，训练时间减少 22 倍）。
  - 支持长历史窗口：在有限计算资源下，可处理更长的回溯窗口（如 `$L=336$` vs `$L=96$`），提升预测精度。

### 2. 通道独立（Channel-independence）

- **原理**：多变量时间序列的每个通道（单变量序列）独立输入 Transformer，共享嵌入层和 Transformer 权重，不混合跨通道信息。
- **优势**：
  - 适应性强：每个通道学习专属注意力模式（如相似序列注意力图相似，差异序列模式不同），适配多变量序列的异质性。
  - 快速收敛与抗过拟合：无需联合学习跨通道-时间信息，训练数据需求低，不易过拟合（对比“通道混合”模型，测试损失持续优化无过拟合）。
  - 鲁棒性高：单个通道的噪声不会扩散到其他通道，可通过调整噪声通道的损失权重进一步优化。

## 3.2 模型整体结构

1. **输入处理**：
    - 多变量序列拆分为 `$M$` 个单变量通道（`$x^{(i)} \in \mathbb{R}^{1 \times L}$`，`$i=1,...,M$`）。
    - 每个通道经**实例归一化**（Instance Norm）后分块，生成 Patch 序列（`$x_p^{(i)} \in \mathbb{R}^{P \times N}$`）。
2. **Transformer 编码器**：
    - Patch 线性投影：将 Patch 映射到 latent 空间（`$x_d^{(i)}=W_p x_p^{(i)}+W_{pos}$`，`$W_{pos}$` 为可学习位置嵌入）。
    - 多头注意力：计算注意力输出 `$O_h^{(i)}=Softmax\left(\frac{Q_h^{(i)} K_h^{(i) T}}{\sqrt{d_k}}\right) V_h^{(i)}$`，
      含 BatchNorm 和残差连接的前馈网络。
3. **预测头**：监督学习中，通过 Flatten 层+线性头输出单通道预测结果（`$\hat{x}^{(i)} \in \mathbb{R}^{1 \times T}$`，
   `$T$` 为预测 horizon）；自监督学习中，用 `$D \times P$` 线性层重构掩码 Patch。
4. **损失函数**：
    - 监督学习用 MSE 损失（平均所有通道的预测与真实值差异）；
    - 自监督学习用 MSE 损失重构掩码 Patch。

# 四、实验设计与结果

## 4.1 实验设置

- **数据集**：8 个主流多变量时间序列数据集，涵盖气象（Weather）、交通（Traffic）、
  电力（Electricity）、流感（ILI）及 4 个电力变压器温度（ETT）数据集，统计如下：

| 数据集   | 特征数（通道数） | 时间步数  |
|----------|------------------|-----------|
| Weather  | 21               | 52696     |
| Traffic  | 862              | 17544     |
| Electricity | 321            | 26304     |
| ILI      | 7                | 966       |
| ETTh1/ETTh2 | 7             | 17420     |
| ETTm1/ETTm2 | 7             | 69680     |

- **基线模型**：Transformer 类（FEDformer、Autoformer、Informer、Pyraformer、LogTrans）、
  非 Transformer 类（DLinear），均优化回溯窗口 `$L$` 取最优结果。
- **模型变体**：
  - PatchTST/42：默认回溯窗口 `$L=336$`，输入 Patch 数 42（`$P=16$`，`$S=8$`）。
  - PatchTST/64：`$L=512$`，输入 Patch 数 64，用于大数据集优化。

## 4.2 核心实验结果

### 1. 多变量长期预测（监督学习）

- **整体性能**：PatchTST 在所有数据集上优于基线，对比最优 Transformer 基线，PatchTST/64 的 MSE 降低 21.0%、
  MAE 降低 16.7%；对比 DLinear，在大数据集（Weather、Traffic、Electricity）优势更显著。
- **关键案例（Traffic 数据集，预测 horizon=96）**：

| 模型                  | 回溯窗口 $L$ | 输入 token 数 $N$ | 是否分块 | 方法         | MSE   |
|-----------------------|-------------|----------------|----------|--------------|-------|
| 通道独立 PatchTST      | 96          | 96             | 否       | 监督         | 0.518 |
| 通道独立 PatchTST      | 380         | 96             | 否       | 下采样（步 4） | 0.447 |
| 通道独立 PatchTST      | 336         | 336            | 否       | 监督         | 0.397 |
| 通道独立 PatchTST      | 336         | 42             | 是       | 监督         | 0.367 |
| 通道独立 PatchTST      | 336         | 42             | 是       | 自监督       | 0.349 |
| 通道混合 FEDformer     | 336         | 336            | 否       | 监督         | 0.597 |
| 通道混合 DLinear       | 336         | 336            | 否       | 监督         | 0.410 |

- **效率提升**：分块设计大幅减少训练时间（`$L=336$` 时，Traffic 数据集从 10040s 降至 464s，提升 22 倍）。

### 2. 自监督表示学习

- **预训练与微调**：
  - 预训练：非重叠 Patch，40% 掩码率，预训练 100 轮。
  - 微调：两种方式——线性探测（仅训练头 20 轮）、端到端微调（先线性探测 10 轮，再全模型微调 20 轮）。
- **结果**：
  - 自监督 PatchTST 微调后性能优于监督训练（如 Weather 数据集 `$T=96$`，自监督 MSE=0.144 vs 监督 MSE=0.152）。
  - 对比其他自监督方法（BTSF、TS2Vec、TNC、TS-TCC），PatchTST 在 ETTh1 数据集上 MSE 降低 34.5%-48.8%。

### 3. 迁移学习

- **任务**：Electricity 数据集预训练，迁移到其他数据集微调。
- **结果**：虽略差于“同数据集预训练-微调”，但仍优于其他基线，且微调仅需更新线性头或少量轮次，计算成本低。

### 4. 消融实验

- **分块与通道独立的必要性**：两者结合（P+CI）性能最优，单独使用任一组件均有损失（如 Weather 数据集 `$T=96$`，
  P+CI 的 MSE=0.152，仅 CI 为 0.164，仅 P 为 0.177）。
- **回溯窗口影响**：PatchTST 随 `$L$` 增大 MSE 持续降低（如 Electricity 数据集 `$T=96$`，
  `$L=720$` 时 MSE=0.202 vs `$L=24$` 时 0.316），而传统 Transformer 对 `$L$` 不敏感。
- **通道独立的通用性**：将通道独立应用于 FEDformer、Autoformer、Informer，
  均提升预测精度（如 Informer 在 Weather 数据集 `$T=96$` 的 MSE 从 0.300 降至 0.174）。

# 五、结论与未来工作

## 5.1 核心结论

1. PatchTST 通过“分块+通道独立”设计，解决了传统 Transformer 在时间序列预测中的复杂度、局部语义缺失及过拟合问题，
   在监督/自监督/迁移学习中均达 SOTA。
2. 分块设计是通用高效的操作，可迁移到其他模型；通道独立可增强模型适应性与抗过拟合能力，且适用于多种 Transformer 变体。

## 5.2 未来方向

1. 结合图神经网络（GNN）扩展通道独立设计，显式建模跨通道关联。
2. 将通道独立与更先进的注意力机制（如稀疏注意力）结合，进一步提升效率与精度。
3. 探索 PatchTST 作为时间序列基础模型（Foundation Model）的潜力，适配更多下游任务（如分类、异常检测）。

# 六、附录关键补充

1. **超参数鲁棒性**：Patch 长度 `$P$`（4-40）、Transformer 层数（3-5）、latent 维度 `$D$`（128-256）对性能影响小，模型稳定性高。
2. **单变量预测结果**：在 ETT 数据集的“油温”单变量预测中，PatchTST 仍优于所有基线（如 ETTm1 数据集 `$T=720$`，
   PatchTST/64 的 MSE=0.073 vs DLinear=0.102）。
3. **可视化**：PatchTST 预测曲线更贴合真实值（如 Weather 数据集 192 步预测，红色预测曲线与蓝色真实曲线几乎重合）。

# 参考

1. [A Time Series is Worth 64 Words：Long-Term Forecasting with Transformers](https://openreview.net/pdf?id=Jbdc0vTOcol)
