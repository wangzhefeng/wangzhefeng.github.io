---
title: BERT
author: 王哲峰
date: '2022-04-05'
slug: dl-bert
categories:
  - nlp
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
</style>

<details><summary>目录</summary><p>

- [TODO](#todo)
- [BERT 介绍](#bert-介绍)
  - [BERT 简介](#bert-简介)
  - [BERT](#bert)
- [BERT 预训练模型库](#bert-预训练模型库)
- [bert-serving-server 搭建 BERT 词向量服务](#bert-serving-server-搭建-bert-词向量服务)
  - [bert-serving-server 简介](#bert-serving-server-简介)
  - [安装 bert-as-service](#安装-bert-as-service)
    - [环境要求](#环境要求)
    - [安装服务端和客户端](#安装服务端和客户端)
  - [启动 BERT 服务](#启动-bert-服务)
    - [下载预训练模型](#下载预训练模型)
    - [启动服务](#启动服务)
    - [调用 BERT 进行应用](#调用-bert-进行应用)
- [参考](#参考)
</p></details><p></p>

# TODO

* [万字长文带你纵览 BERT 家族](https://mp.weixin.qq.com/s/ejWRhjYDFSNkAVMyMzwpOQ)
* [Bert系列之模型参数计算](https://mp.weixin.qq.com/s/D7T0Pdqr01viicJfAWdIMQ)
- [BERT代码从零解读](https://mp.weixin.qq.com/s/vUcryYyedxlbe0WhAOZhIA)
- https://mp.weixin.qq.com/s/HHVMYv66nJ_vHZgit81gdQ
* [BERT meet Knowledge Graph：预训练模型与知识图谱相结合的研究进展](https://mp.weixin.qq.com/s?__biz=MzI3MTA0MTk1MA==&mid=2652088025&idx=4&sn=909d52eada0f70bf4826bb218d3291ad&chksm=f120d228c6575b3e1e4c28dfbba41cac4e3db300efe546e05977b7b90742d00f4afe963c0ae6&mpshare=1&scene=1&srcid=12010etKB71ivFrNo8lrZknY&sharer_sharetime=1606810826557&sharer_shareid=d02c7b7d24f901d5f03e661b5ee5c1e5&version=3.0.36.6180&platform=mac#rd)
* [技术详解：BERT的分词预处理、输入Embedding、中间编码与输出向量解析](https://mp.weixin.qq.com/s/LglV7qE5vP9hmDL2azvnfg)
* [BERT 详解](https://zhuanlan.zhihu.com/p/103226488)

# BERT 介绍

## BERT 简介

## BERT

Pre-training of Deep Bidirectional Transformers for Language Understanding

# BERT 预训练模型库

- 哈工大讯飞联合实验室
    - 基于TensorFlow/PyTorch + Transformers
    - https://github.com/ymcui/Chinese-BERT-wwm
- CLUE预训练模型
    - 基于TensorFlow/PyTorch + Transformers
    - https://github.com/CLUEbenchmark/CLUEPretrainedModels
- 百度预训练模型ERNIE/BERT
    - 基于PaddlePaddle
    - https://www.paddlepaddle.org.cn/modelbasedetail/ERNIE
- 阿里云预训练模型BERT等
    - 基于PAI-TensorFlow
    - https://www.yuque.com/easytransfer/cn/xfe19v
- 百度ernie
    - pytorch实现
    - https://huggingface.co/nghuyong/ernie-1.0


# bert-serving-server 搭建 BERT 词向量服务

## bert-serving-server 简介

BERT 工具能够快速得到词向量表示, 叫做: bert-as-service, 
只要调用该服务就能够得到想要的向量表示

## 安装 bert-as-service

### 环境要求

- Python>=3.5
- TensorFlow>=1.10

### 安装服务端和客户端

```bash
$ $ pip install -i https://pypi.tuna.tsinghua.edu.cn/simple bert-serving-server bert-serving-client
```

## 启动 BERT 服务

### 下载预训练模型

- [GitHub 地址](https://github.com/google-research/bert/) 
    - `BERT-Large, Uncased (Whole Word Masking)$`: 
        - 24-layer, 1024-hidden, 16-heads, 340M parameters
    - `BERT-Large, Cased (Whole Word Masking)$`: 
        - 24-layer, 1024-hidden, 16-heads, 340M parameters
    - `BERT-Base, Uncased`: 
        - 12-layer, 768-hidden, 12-heads, 110M parameters
    - `BERT-Large, Uncased`: 
        - 24-layer, 1024-hidden, 16-heads, 340M parameters
    - `BERT-Base, Cased`: 
        - 12-layer, 768-hidden, 12-heads , 110M parameters
    - `BERT-Large, Cased`: 
        - 24-layer, 1024-hidden, 16-heads, 340M parameters
    - `BERT-Base, Multilingual Cased (New, recommended)$`: 
        - 104 languages, 12-layer, 768-hidden, 12-heads, 110M parameters
    - `BERT-Base, Multilingual Uncased (Orig, not recommended)$` (Not recommended, use Multilingual Casedinstead): 
        - 102 languages, 12-layer, 768-hidden, 12-heads, 110M parameters
    - `BERT-Base, Chinese`: Chinese Simplified and Traditional
        - 12-layer, 768-hidden, 12-heads, 110M parameters
- 下载 `BERT-Base, Chinese`, 放在项目根目录下

### 启动服务

解压缩后, 运行如下命令进行启动, 目录换成解压后的路径

```bash
$ $ bert-serving-start -model_dir /path/to/model -num_worker=4
$ bert-serving-start -model_dir /Users/zfwang/project/machinelearning/deeplearning/datasets/NLP_data/chinese_L-12_H-768_A-12 -num_worker=4
```
- `-num_worker`: 指定使用多少个 CPU

### 调用 BERT 进行应用

```python
from bert_serving.client import BertClient
bc = BertClient(ip = "localhost", check_version = False, check_length = False)
vec = bc.encode(["学习"])
print(vec)
```

# 参考



