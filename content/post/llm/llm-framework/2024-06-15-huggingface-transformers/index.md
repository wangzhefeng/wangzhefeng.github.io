---
title: LLM 框架--Huggingface
author: 王哲峰
date: '2024-06-15'
slug: huggingface-transformers
categories:
  - llm
tags:
  - tool
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

- [Huggging Face NLP ecosystem](#huggging-face-nlp-ecosystem)
- [NLP 介绍](#nlp-介绍)
- [Transformers](#transformers)
  - [简介](#简介)
  - [安装](#安装)
  - [使用](#使用)
    - [快速上手](#快速上手)
  - [工具 pipeline](#工具-pipeline)
  - [Fine-tuning Pretrained Model](#fine-tuning-pretrained-model)
    - [处理数据](#处理数据)
    - [Fine-tuning model](#fine-tuning-model)
  - [分享 Models 和 Tokenizers](#分享-models-和-tokenizers)
  - [模型](#模型)
- [Datasets](#datasets)
- [Tokenizers](#tokenizers)
- [Accelerate](#accelerate)
- [Hugging Face Hub](#hugging-face-hub)
- [参考](#参考)
</p></details><p></p>

# Huggging Face NLP ecosystem

![img](images/overview.png)

* [Transformers](https://github.com/huggingface/transformers)
* [Datasets](https://github.com/huggingface/datasets)
* [Tokenizers](https://github.com/huggingface/tokenizers)
* [Accelerate](https://github.com/huggingface/accelerate)
* [Hugging Face Hub](https://huggingface.co/models)

# NLP 介绍

NLP 任务：

* **Classifying whole sentences**: Getting the sentiment of a review, detecting if an email is spam,
  determining if a sentence is grammatically correct or whether two sentences are logically related or not
* **Classifying each word in a sentence**: Identifying the grammatical components of a sentence (noun, 
  verb, adjective), or the named entities (person, location, organization)
* **Generating text content**: Completing a prompt with auto-generated text, 
  filling in the blanks in a text with masked words
* **Extracting an answer from a text**: Given a question and a context, 
  extracting the answer to the question based on the information provided in the context
* **Generating a new sentence from an input text**: Translating a text into another language, 
  summarizing a text

# Transformers

## 简介

* 🤗 Transformers 提供了数以千计的预训练模型，支持 100 多种语言的文本分类、
  信息抽取、问答、摘要、翻译、文本生成。它的宗旨是让最先进的 NLP 技术人人易用。
* 🤗 Transformers 提供了便于快速下载和使用的 API，让你可以把预训练模型用在给定文本、
  在你的数据集上微调然后通过 model hub 与社区共享。同时，每个定义的 Python 模块均完全独立，
  方便修改和快速研究实验。
* 🤗 Transformers 支持三个最热门的深度学习库：Jax, PyTorch 以及 TensorFlow — 并与之无缝整合。
  你可以直接使用一个框架训练你的模型然后用另一个加载和推理。

为什么要用 transformers？

* 便于使用的先进模型：
    - NLU 和 NLG 上表现优越
    - 对教学和实践友好且低门槛
    - 高级抽象，只需了解三个类
    - 对所有模型统一的 API
* 更低计算开销，更少的碳排放：
    - 研究人员可以分享已训练的模型而非每次从头开始训练
    - 工程师可以减少计算用时和生产环境开销
    - 数十种模型架构、2000 多个预训练模型、100 多种语言支持
* 对于模型生命周期的每一个部分都面面俱到：
    - 训练先进的模型，只需 3 行代码
    - 模型在不同深度学习框架间任意转移，随你心意
    - 为训练、评估和生产选择最适合的框架，衔接无缝
* 为你的需求轻松定制专属模型和用例：
    - 我们为每种模型架构提供了多个用例来复现原论文结果
    - 模型内部结构保持透明一致
    - 模型文件可单独使用，方便魔改和快速实验

## 安装

* Python 3.8+
* PyTorch 1.11+

pip:

```bash
$ pip install transformers
```

conda:

```bash
$ conda install conda-forge::transformers
```

## 使用

### 快速上手

1. 使用 `pipeline` 判断正负面情绪

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
res1 = classifier("We are very happy to introduce pipeline to the transformers repository.")
res2 = classifier(
    ["I've been waiting for a HuggingFace course my whole life.", "I hate this so much!"]
)
print(res)
```

```
[{'label': 'POSITIVE', 'score': 0.9996980428695679}]
```

2. 从给定文本中抽取问题答案

```python
from transformers import pipeline

question_answerer = pipeline("question-answering")
res = question_answerer({
    "question": "What is the name of the repository ?",
    "context": "Pipeline has been included in the huggingface/transformers repository",
})
print(res)
```

3. 可以在任务中下载、上传、使用任意预训练模型。

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = AutoModel.from_pretrained("google-bert/bert-base-uncase")
inputs = tokenizer("Hello world!", return_tensors = "pt")
outputs = model(**inputs)
```

## 工具 pipeline

> https://huggingface.co/docs/transformers/main_classes/pipelines#pipelines

可用的 pipeline：

* `feature-extraction` (get the vector representation of a text)
* `fill-mask`
* `ner` (named entity recognition)
* `question-answering`
* `sentiment-analysis`
* `summarization`
* `text-generation`
* `translation`
* `zero-shot-classification`


## Fine-tuning Pretrained Model

### 处理数据



### Fine-tuning model



## 分享 Models 和 Tokenizers


## 模型

* [模型架构](https://huggingface.co/docs/transformers/model_summary)
* [支持的模型(包括 tokenizer 模型)](https://huggingface.co/docs/transformers/index#supported-frameworks)

# Datasets


# Tokenizers

# Accelerate

# Hugging Face Hub


https://huggingface.co/models

# 参考

* [Hugging Face Transformers Github](https://github.com/huggingface/transformers/blob/main/README_zh-hans.md)
* [DeepLearning.AI’s Natural Language Processing Specialization](https://www.coursera.org/specializations/natural-language-processing?utm_source=deeplearning-ai&utm_medium=institutions&utm_campaign=20211011-nlp-2-hugging_face-page-nlp-refresh)
* [fast.ai’s Practical Deep Learning for Coders](https://course.fast.ai/)