---
title: 大语言模型
subtitle: Large Language Models
list_pages: true
# order_by: title
---

# LLM 发展

* word2vec(2013, Google)
    - 一种从文本数据中学习单词嵌入(word embedding)的技术，它能够捕捉到单词之间的语义关系
* Seq2Seq 模型与 attention 机制(2014~2015, Google)
    - Seq2Seq: sequence-to-seqence
    - attention 机制: 注意力机制
    - 对机器翻译和其他序列生成任务产生了重要影响，提升了模型处理长序列数据的能力
* Transformer 模型(2017, Google)
    - Transformer: Attention Is All You Need
    - 一种全新的基于注意力机制的架构，成为了后来很多大模型的基础
* BERT 模型(2018, Google)
    - BERT: Bidirectional Encoder Representations from Transformers
    - 采用了 Transformer 架构，并通过双向上下文来理解单词的意义，大幅提高了语言理解的准确性
* T5 模型(2019, Google) 
    - Text-to-Text Transfer Transformer
    - 把不同的 NLP 任务，如分类，相似度计算等，都统一到一个文本到文本的框架里进行解决，
      这样的设计使得单一模型能够处理翻译、摘要和问答等多种任务
* GPT-3 模型(2020)
    - GPT-3: Generative Pre-Train Transformer 3
    - 生成式人工智能: Generative Artificial Intelligence, Generative AI
    - 通用人工智能: Artificial General Intelligence, AGI
    - 一个拥有 1750 亿参数的巨大模型，它在很多 NLP 任务上无须进行特定训练即可达到很好的效果，
      显示出令人惊叹的零样本(zero-shot)和小样本(few-shot)学习能力
* ChatGPT(2022.11.30, OpenAI)
    - 基于 GPT-3.5 模型调优的新一代对话式 AI 模型。该模型能够自然地进行多轮对话，精确地回答问题，并能生成编码代码、电子邮件、学术论文和小说等多种文本
* LLaMA(2023.2.24, Meta)
    - Meta 开源的大模型，其性能超越了 OpenAI 的 GPT-3
* GPT-4(2023.3.14, OpenAI)
    - 多模态模型，其回答准确度较 GPT-3.5 提升了 40%，在众多领域的测试中超越了大部分人类的水平，
      展示了 AI 在理解复杂任务方面的巨大潜力 
* Vicuna-13B(2023.3.31, 加州大学伯克利分校、CMU、斯坦福、UCSD、MBZUAI)
    - 拥有 130 亿参数的模型仅需要 300 美元的训练成本，为 AI 领域带来了成本效益上的重大突破
* PaLM 2 AI(2023.5.10, Google)
    - 支持对话导出、编码生成以及新增视觉搜索和图像生成功能
* Claude 2(2023.7.12, Anthropic)
    - 支持多达 100k token(4 万至 5 万个汉字)的上下文处理，在安全性和编码、数学及推理方面表现出色，
      提升了 AI 在处理长文本和复杂问题方面的能力
* LLaMA 2(2023.7.19, Meta)
    - 包含 70 亿、130 亿、700 亿参数版本的模型，其性能赶上了 GPT-3.5，显示了 AI 模型在不同规模下的多样性和适应性
* 国内
    - GhatGLM-6B(清华大学)
    - 文心一言(百度)
    - 通义千问(阿里)
    - 星火认知大模型(科大讯飞)
    - MaaS: Model as a Service(腾讯)
    - 盘古大模型 3.0(华为)



# TODO

* [Google Gemma 2B 微调实战（IT 科技新闻标题生成）](https://mp.weixin.qq.com/s/MX_7kiwhWzPd3REOc_KJaQ)

# 文档
