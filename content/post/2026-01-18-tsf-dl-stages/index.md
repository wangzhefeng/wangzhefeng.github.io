---
title: 深度时序模型研究的三个阶段
author: wangzf
date: '2026-01-18'
slug: tsf-dl-stages
categories:
  - timeseries
tags:
  - note
---

**知乎问题：时间序列预测还能再进步吗？**

> 在实现了市面上几个主要的模型例如 PatchTST，FITS，TimesNet，iTransformer 之后，
> 感觉 23,24 年的时间序列模型进步都不大。我个人看法是，模型的输入并不能完全代表所有影响结果的因素，
> 因此一分不差地预测未来发生的值是不可能的。会不会已经到极限了

**回答：**

深度时序模型的研究主要经历了下面三个阶段，每个阶段都有大家重点关注的，想要解决的“热点问题”。

1. **如何将序列模型应用到时序数据中（2021-2022 年）**
  
在这个初期阶段，大家都在试图解决如何将经典的序列模型结构（比如 Transformer、RNN）结合进时序建模这一特定任务中。
这一阶段的代表性工作有：Informer、Autoformer、FEDformer、Non-stationary Transformer 等。

经过这一阶段之后，大家总结到的实用技巧有：

* 分解建模（Autoformer）
* 窗口归一化（Revin，Non-stationary Transformer）
  
2. **如何完成时序的令牌化 Tokenization（2023 年）**

在第一阶段的时候，大家输入时序模型的还是离散的点，但是其实从深度学习 Token 构建角度来说，
单个时刻的信息量还是太少了，所以需要将表征增强到成 Patch（一段序列），这样后续的建模会更加可靠，
这一阶段的代表性工作有：PatchTST、Crossformer、iTransformer等。

经过这一阶段之后，大家总结到的实用技巧有：

* PatchEmbedding（PatchTST、Crossformer）
* VariateEmbedding（iTransformer）

3. **多任务、多模态建模（2023-2024 年）**

当第一第二阶段积累的训练技巧逐步完备之后（注意，并不是说大家贡献仅仅是一个训练技巧，
如何让一个深度模型在时序数据上训练得很好是非常重要的进展），大家开始想做一些更加多样的任务，
这一阶段的代表性工作有 TimesNet、Timer、Time-LLM 等。

在这一阶段，大家讨论的热点问题有“大语言模型与时序模型的关系”，“是否存在时序基础模型”。
从纯技术角度来看，具体问题还有“建模的大模型应该是单变量的，还是多变量的？”，
“是不是应该将大语言模型作为时序模型的主干？”。

4. 总结

综上，我觉得题主提出的这个问题主要是针对第 1、2 两个发展阶段。从这个角度讲，
如果问“是否有新的技巧出现”的话，确实这两年很少有了，很多在标准 benchmark 上的效果提升可能来自于调参或者实验方差。
但是如果从第 3 阶段角度来看，我觉得时序领域还处于百家争鸣阶段，很难明确哪个技术路线是对的。

这里也宣传两篇，我们最近对于第3阶段的思考

* （1）TimeXer（TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables）：协变量预测模型。
  我觉得这个“协变量预测”设置非常完美地回避了单变量预测（Channel Independent）缺少充足信息，
  多变量预测需要在不同变量间协调的问题。同时也解决混合大数据集内部，不同数据源变量数不一致的问题。
* （2）MetaTST（Metadata Matters for Time Series: Informative Forecasting with Transformers）：
  使用大语言模型编码时序数据文本形态的“元信息”，从而为大规模混合数据训练模型做准备。
  我觉得直接使用大语言模型处理文本是非常正确的思路，因为现在还没有迹象表明大语言模型对时序非常有效，
  让 LLM 干它自己擅长的事情就好了。
