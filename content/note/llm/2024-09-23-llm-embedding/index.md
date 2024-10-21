---
title: LLM Embedding 调用
subtitle: LLM Embedding API
author: wangzf
date: '2024-09-23'
slug: llm-embedding
categories:
  - llm
tags:
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

- [Embedding 模型使用](#embedding-模型使用)
    - [OpenAI API](#openai-api)
        - [模型简介](#模型简介)
        - [使用示例](#使用示例)
    - [文心千帆 API](#文心千帆-api)
        - [模型简介](#模型简介-1)
        - [使用示例](#使用示例-1)
    - [智谱 API](#智谱-api)
        - [模型简介](#模型简介-2)
        - [使用示例](#使用示例-2)
    - [M3E](#m3e)
        - [安装](#安装)
        - [模型 Embedding](#模型-embedding)
        - [模型微调](#模型微调)
    - [讯飞星火 API](#讯飞星火-api)
- [Embedding 模型微调](#embedding-模型微调)
    - [uniem 微调示例](#uniem-微调示例)
    - [FineTuner 支持的数据类型](#finetuner-支持的数据类型)
        - [PairRecord](#pairrecord)
        - [TripletRecord](#tripletrecord)
        - [ScoredPairRecord](#scoredpairrecord)
    - [微调示例](#微调示例)
        - [微调 M3E](#微调-m3e)
        - [微调 text2vec](#微调-text2vec)
        - [微调 sentences\_transformers](#微调-sentences_transformers)
        - [从头训练](#从头训练)
        - [SGPT](#sgpt)
- [参考](#参考)
</p></details><p></p>

# Embedding 模型使用

为了方便 Embedding API 调用，首先，应将 API key 填入 `.env` 文件，
然后使用代码读取并加载环境变量。

## OpenAI API

### 模型简介

GPT 有封装好的接口，使用时简单封装即可。目前 GPT Embedding model 有三种，性能如下

| 模型                   | 每美元页数 | MTEB得分 | MIRACL得分  |
|------------------------|-----------|---------|-------------|
| text-embedding-3-large | 9,615     | 64.6    | 54.9        |
| text-embedding-3-small | 62,500    | 62.3    | 44.0        |
| text-embedding-ada-002 | 12,500    | 61.0    | 31.4        |

其中：

* MTEB 得分为 Embedding model 分类、聚类、配对等八个任务的平均得分
* MIRACL 得分为 Embedding model 在检索任务上的平均得分

从以上三个 Embedding model 可以看出：

* `text-embedding-3-large` 有最好的性能和最贵的价格，
  当搭建的应用需要更好的表现且成本充足的情况下可以使用；
* `text-embedding-3-small` 有较好的性价比，当预算有限时可以选择该模型；
* `text-embedding-ada-002` 是 OpenAI 上一代的模型，
  无论在性能还是价格都不及前两者，因此不推荐使用。

### 使用示例

```python
import os

from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"


def openai_embedding(text: str, model: str = None):
    # 获取环境变量 OPENAI_API_KEY
    api_key = os.environ["OPENAI_API_KEY"]
    client = OpenAI(api_key = api_key)
    # embedding model
    if model == None:
        model = "text-embedding-3-small"
    # 模型调用    
    response = client.embeddings.create(
        input = text,
        model = model,
    )

    return response


response = openai_embedding(text="要生成 embedding 的输入文本，字符串形式。")
print(f"返回的 embedding 类型为：{response.object}")
print(f"embedding 长度为：{len(response.data[0].embedding)}")
print(f"embedding (前 10) 为：{response.data[0].embedding[:10]}")
print(f"本次 embedding model 为：{response.model}")
print(f"本次 token 使用情况为：{response.usage}")
```

API 返回的数据为 JSON 格式，除 `object` 向量类型外还有存放数据的 `data`、
embedding model 型号 `model` 以及本次 token 使用情况 `usage` 等数据，
具体如下所示：

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [
        -0.006929283495992422,
        ... (省略)
        -4.547132266452536e-05,
      ],
    }
  ],
  "model": "text-embedding-3-small",
  "usage": {
    "prompt_tokens": 5,
    "total_tokens": 5
  }
}
```

```
返回的 embedding 类型为：list

embedding 长度为：1536

embedding（前 10）为：[0.03884002938866615, 0.013516489416360855, -0.0024250170681625605, -0.01655769906938076, 0.024130908772349358, -0.017382603138685226, 0.04206013306975365, 0.011498954147100449, -0.028245486319065094, -0.00674333656206727]

本次 embedding model 为：text-embedding-3-small

本次 token 使用情况为：Usage(prompt_tokens=12, total_tokens=12)
```

## 文心千帆 API

### 模型简介

Embedding-V1 是基于百度文心大模型技术的文本表示模型，`Access token` 为调用接口的凭证，
使用 Embedding-V1 时应先凭 `API Key`、`Secret Key` 获取 `Access token`，
再通过 `Access token` 调用接口来 Embedding text。
同时千帆大模型平台还支持 `bge-large-zh` 等 Embedding 模型。

### 使用示例

```python
# -*- coding: utf-8 -*-

# ***************************************************
# * File        : wenxin_embedding_api.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-08-03
# * Version     : 0.1.080316
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import json
import requests
from dotenv import load_dotenv, find_dotenv

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
# 读取本地/项目的环境变量
_ = load_dotenv(find_dotenv())
# 如果需要通过代理端口访问，还需要做如下配置
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"


def wenxin_embedding(text: str):
    # 获取环境变量 wenxin_api_key, wenxin_secret_key
    api_key = os.environ["QIANFN_AK"]
    secret_key = os.environ["QIANFAN_SK"]
    
    # 使用 API Key、Secret Key 向 https://aip.baidubce.com/oauth/2.0/token 获取 Access token
    url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={api_key}&client_secret={secret_key}"
    payload = json.dumps("")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    response = requests.request("POST", url, headers = headers, data = payload)
    
    # 通过获取的 Access token 来 embedding text
    url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/embeddings/embedding-v1?access_token={str(response.json().get('access_token'))}"
    input = []
    input.append(text)
    payload = json.dumps({"input": input})
    headers = {
        "Content-Type": "application/json"
    }
    response = requests.request("POST", url, headers = headers, data = payload)

    return json.loads(response.text)


# 测试代码 main 函数
def main():
    # text 应为 List(str)
    text = "要生成 embedding 的输入文本，字符串形式。"
    response = wenxin_embedding(text = text)

    print(f"本次 embedding id 为：{response["id"]}")
    print(f"本次 embedding 产生的时间戳为：{response["created"]}")
    print(f"返回的 embedding 类型为：{response["object"]}")
    print(f"embedding 长度为：{response["data"][0]["embedding"]}")
    print(f"embedding (前 10) 为：{response["data"][0]["embedding"][:10]}")

if __name__ == "__main__":
    main()
```

```python
本次 embedding id 为：as-hvbgfuk29u

本次 embedding 产生时间戳为：1711435238

返回的 embedding 类型为:embedding_list

embedding 长度为：384

embedding（前 10）为：[0.060567744076251984, 0.020958080887794495, 0.053234219551086426, 0.02243831567466259, -0.024505289271473885, -0.09820500761270523, 0.04375714063644409, -0.009092536754906178, -0.020122773945331573, 0.015808865427970886]
```

## 智谱 API

### 模型简介

智谱有封装好的 SDK，直接调用即可。

### 使用示例

```python
# -*- coding: utf-8 -*-

# ***************************************************
# * File        : zhipu_embedding_api.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-08-03
# * Version     : 0.1.080317
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from dotenv import load_dotenv, find_dotenv

from zhipuai import ZhipuAI

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
# 读取本地/项目的环境变量
_ = load_dotenv(find_dotenv())
# 如果需要通过代理端口访问，还需要做如下配置
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"


def zhipu_embedding(text: str):
    api_key = os.environ["ZHIPUAI_API_KEY"]
    client = ZhipuAI(api_key = api_key)
    response = client.embeddings.create(
        model = "embedding-2",
        input = text,
    )
    
    return response


# 测试代码 main 函数
def main():
    text = "要生成 embedding 的输入文本，字符串形式。"
    response = zhipu_embedding(text = text)

    print(f"response 类型为：{type(response)}")
    print(f"embedding 类型为：{response.object}")
    print(f"生成 embedding 的 model 为：{response.model}")
    print(f"生成的 embedding 长度为：{len(response.data[0].embedding)}")
    print(f"embedding(前 10)为: {response.data[0].embedding[:10]}")

if __name__ == "__main__":
    main()
```

`response` 为 `zhipuai.types.embeddings.EmbeddingsResponsed` 类型，
可以调用以下属性

* `object`: 来查看 `response` 的 embedding 类型
* `data`: 查看 embedding
* `model`: 查看 embedding model
* `usage`: 查看 embedding model 使用情况

```
response类型为：<class 'zhipuai.types.embeddings.EmbeddingsResponded'>

embedding类型为：list

生成embedding的model为：embedding-2

生成的embedding长度为：1024

embedding（前10）为: [0.017892399802803993, 0.0644201710820198, -0.009342825971543789, 0.02707476168870926, 0.004067837726324797, -0.05597858875989914, -0.04223804175853729, -0.03003198653459549, -0.016357755288481712, 0.06777040660381317]
```

## M3E

> HuggingFace M3E: https://huggingface.co/moka-ai/m3e-base

M3E 是 Moka Massive Mixed Embedding 的缩写。

* `Moka`，此模型由 MokaAI 训练，开源和评测，训练脚本使用 `uniem` ，评测 BenchMark 使用 `MTEB-zh`
* `Massive`，此模型通过千万级 (2200w+) 的中文句对数据集进行训练
* `Mixed`，此模型支持中英双语的同质文本相似度计算，异质文本检索等功能，未来还会支持代码检索
* `Embedding`，此模型是文本嵌入模型，可以将自然语言转换成稠密的向量

### 安装

```bash
$ pip install sentence-transformers
```

### 模型 Embedding

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('moka-ai/m3e-base')

# Our sentences we like to encode
sentences = [
    '* Moka 此文本嵌入模型由 MokaAI 训练并开源，训练脚本使用 uniem',
    '* Massive 此文本嵌入模型通过**千万级**的中文句对数据集进行训练',
    '* Mixed 此文本嵌入模型支持中英双语的同质文本相似度计算，异质文本检索等功能，未来还会支持代码检索，ALL in one'
]

# Sentences are encoded by calling model.encode()
embeddings = model.encode(sentences)

#Print the embeddings
for sentence, embedding in zip(sentences, embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("")
```

### 模型微调

```python
from datasets import load_dataset
from uniem.finetuner import FineTuner

dataset = load_dataset('shibing624/nli_zh', 'STS-B')

# 指定训练的模型为 m3e-small
finetuner = FineTuner.from_pretrained('moka-ai/m3e-small', dataset = dataset)
finetuner.run(epochs = 1)
```

## 讯飞星火 API

未开放

# Embedding 模型微调

> * [uniem 中文通用文本嵌入模型](https://github.com/wangyuxinwhy/uniem)
> * [uniem 微调示例](https://github.com/wangyuxinwhy/uniem/blob/main/examples/finetune.ipynb)

## uniem 微调示例

`uniem` 安装：

```bash
$ pip install uniem
```

使用一个简单示例感受一下微调的过程：

```python
from datasets import load_dataset
from uniem.finetuner import FineTuner

# data
dataset = load_dataset("shibing624/nli_zh", "STS-B", cache_dir = "cache")

# finetune
finetuner = FineTuner.from_pretrained(
    "moka-ai/m3e-small", 
    dataset = dataset
)
finetuned_model = finetuner.run(epochs = 3, batch_size = 64, lr = 3e-5)
```

```
Batch size: 64
Start with seed: 42
Output dir: finetuned-model
Learning rate: 3e-05
Start training for 3 epochs

  0%|          | 0/82 [00:00<?, ?it/s]
  1%|          | 1/82 [00:00<00:13,  5.86it/s]
Epoch 1/3 - loss: 6.9539:   1%|          | 1/82 [00:00<00:13,  5.86it/s]
Epoch 1/3 - loss: 6.9539:   6%|▌         | 5/82 [00:00<00:04, 18.80it/s]
Epoch 1/3 - loss: 6.9539:  10%|▉         | 8/82 [00:00<00:03, 22.45it/s]
Epoch 1/3 - loss: 6.6007:  13%|█▎        | 11/82 [00:00<00:03, 22.45it/s]
Epoch 1/3 - loss: 6.6007:  15%|█▍        | 12/82 [00:00<00:02, 25.96it/s]
Epoch 1/3 - loss: 6.6007:  18%|█▊        | 15/82 [00:00<00:02, 25.51it/s]
Epoch 1/3 - loss: 6.6007:  22%|██▏       | 18/82 [00:00<00:02, 25.51it/s]
Epoch 1/3 - loss: 6.6007:  26%|██▌       | 21/82 [00:00<00:02, 25.54it/s]
Epoch 1/3 - loss: 6.5924:  26%|██▌       | 21/82 [00:00<00:02, 25.54it/s]
Epoch 1/3 - loss: 6.5924:  29%|██▉       | 24/82 [00:01<00:02, 25.02it/s]
Epoch 1/3 - loss: 6.5924:  33%|███▎      | 27/82 [00:01<00:02, 25.10it/s]
Epoch 1/3 - loss: 6.5924:  37%|███▋      | 30/82 [00:01<00:02, 25.41it/s]
Epoch 1/3 - loss: 6.7482:  38%|███▊      | 31/82 [00:01<00:02, 25.41it/s]
Epoch 1/3 - loss: 6.7482:  40%|████      | 33/82 [00:01<00:02, 22.46it/s]
Epoch 1/3 - loss: 6.7482:  44%|████▍     | 36/82 [00:01<00:02, 19.65it/s]
Epoch 1/3 - loss: 6.7482:  48%|████▊     | 39/82 [00:01<00:02, 18.01it/s]
Epoch 1/3 - loss: 6.7482:  50%|█████     | 41/82 [00:01<00:02, 16.81it/s]
Epoch 1/3 - loss: 6.8519:  50%|█████     | 41/82 [00:01<00:02, 16.81it/s]
Epoch 1/3 - loss: 6.8519:  52%|█████▏    | 43/82 [00:02<00:02, 16.59it/s]
Epoch 1/3 - loss: 6.8519:  55%|█████▍    | 45/82 [00:02<00:02, 16.13it/s]
Epoch 1/3 - loss: 6.8519:  57%|█████▋    | 47/82 [00:02<00:02, 15.03it/s]
Epoch 1/3 - loss: 6.8519:  60%|█████▉    | 49/82 [00:02<00:02, 13.58it/s]
Epoch 1/3 - loss: 6.8694:  62%|██████▏   | 51/82 [00:02<00:02, 13.58it/s]
Epoch 1/3 - loss: 6.8694:  63%|██████▎   | 52/82 [00:02<00:01, 16.52it/s]
Epoch 1/3 - loss: 6.8694:  67%|██████▋   | 55/82 [00:02<00:01, 19.07it/s]
Epoch 1/3 - loss: 6.8694:  71%|███████   | 58/82 [00:02<00:01, 21.05it/s]
Epoch 1/3 - loss: 6.8694:  74%|███████▍  | 61/82 [00:03<00:00, 21.75it/s]
Epoch 1/3 - loss: 6.8882:  74%|███████▍  | 61/82 [00:03<00:00, 21.75it/s]
Epoch 1/3 - loss: 6.8882:  78%|███████▊  | 64/82 [00:03<00:00, 22.81it/s]
Epoch 1/3 - loss: 6.8882:  82%|████████▏ | 67/82 [00:03<00:00, 22.65it/s]
Epoch 1/3 - loss: 6.8882:  85%|████████▌ | 70/82 [00:03<00:00, 23.23it/s]
Epoch 1/3 - loss: 6.9010:  87%|████████▋ | 71/82 [00:03<00:00, 23.23it/s]
Epoch 1/3 - loss: 6.9010:  89%|████████▉ | 73/82 [00:03<00:00, 23.42it/s]
Epoch 1/3 - loss: 6.9010:  93%|█████████▎| 76/82 [00:03<00:00, 22.86it/s]
Epoch 1/3 - loss: 6.9010:  96%|█████████▋| 79/82 [00:03<00:00, 21.19it/s]
Epoch 1/3 - loss: 6.8902:  99%|█████████▉| 81/82 [00:03<00:00, 21.19it/s]
Epoch 1/3 - loss: 6.8902: 100%|██████████| 82/82 [00:03<00:00, 22.95it/s]
Epoch 1/3 - loss: 6.8902: 100%|██████████| 82/82 [00:03<00:00, 20.82it/s]
Epoch 1 Validation loss: 6.7225

  0%|          | 0/82 [00:00<?, ?it/s]
Epoch 2/3 - loss: 6.7364:   1%|          | 1/82 [00:00<00:03, 26.94it/s]
Epoch 2/3 - loss: 6.7364:   5%|▍         | 4/82 [00:00<00:02, 30.55it/s]
Epoch 2/3 - loss: 6.7364:  10%|▉         | 8/82 [00:00<00:02, 30.92it/s]
Epoch 2/3 - loss: 6.3974:  13%|█▎        | 11/82 [00:00<00:02, 30.92it/s]
Epoch 2/3 - loss: 6.3974:  15%|█▍        | 12/82 [00:00<00:02, 31.25it/s]
Epoch 2/3 - loss: 6.3974:  20%|█▉        | 16/82 [00:00<00:02, 28.92it/s]
Epoch 2/3 - loss: 6.3974:  23%|██▎       | 19/82 [00:00<00:02, 27.65it/s]
Epoch 2/3 - loss: 6.3703:  26%|██▌       | 21/82 [00:00<00:02, 27.65it/s]
Epoch 2/3 - loss: 6.3703:  27%|██▋       | 22/82 [00:00<00:02, 27.12it/s]
Epoch 2/3 - loss: 6.3703:  30%|███       | 25/82 [00:00<00:02, 26.00it/s]
Epoch 2/3 - loss: 6.3703:  34%|███▍      | 28/82 [00:01<00:02, 26.15it/s]
Epoch 2/3 - loss: 6.3703:  38%|███▊      | 31/82 [00:01<00:01, 25.91it/s]
Epoch 2/3 - loss: 6.4807:  38%|███▊      | 31/82 [00:01<00:01, 25.91it/s]
Epoch 2/3 - loss: 6.4807:  41%|████▏     | 34/82 [00:01<00:02, 22.19it/s]
Epoch 2/3 - loss: 6.4807:  45%|████▌     | 37/82 [00:01<00:02, 19.52it/s]
Epoch 2/3 - loss: 6.4807:  49%|████▉     | 40/82 [00:01<00:02, 17.45it/s]
Epoch 2/3 - loss: 6.5948:  50%|█████     | 41/82 [00:01<00:02, 17.45it/s]
Epoch 2/3 - loss: 6.5948:  51%|█████     | 42/82 [00:01<00:02, 16.89it/s]
Epoch 2/3 - loss: 6.5948:  54%|█████▎    | 44/82 [00:01<00:02, 16.55it/s]
Epoch 2/3 - loss: 6.5948:  56%|█████▌    | 46/82 [00:02<00:02, 15.73it/s]
Epoch 2/3 - loss: 6.5948:  59%|█████▊    | 48/82 [00:02<00:02, 14.41it/s]
Epoch 2/3 - loss: 6.5948:  61%|██████    | 50/82 [00:02<00:02, 14.92it/s]
Epoch 2/3 - loss: 6.6302:  62%|██████▏   | 51/82 [00:02<00:02, 14.92it/s]
Epoch 2/3 - loss: 6.6302:  65%|██████▍   | 53/82 [00:02<00:01, 17.80it/s]
Epoch 2/3 - loss: 6.6302:  68%|██████▊   | 56/82 [00:02<00:01, 19.87it/s]
Epoch 2/3 - loss: 6.6302:  72%|███████▏  | 59/82 [00:02<00:01, 20.99it/s]
Epoch 2/3 - loss: 6.6536:  74%|███████▍  | 61/82 [00:02<00:01, 20.99it/s]
Epoch 2/3 - loss: 6.6536:  76%|███████▌  | 62/82 [00:02<00:00, 22.64it/s]
Epoch 2/3 - loss: 6.6536:  79%|███████▉  | 65/82 [00:03<00:00, 23.38it/s]
Epoch 2/3 - loss: 6.6536:  83%|████████▎ | 68/82 [00:03<00:00, 23.56it/s]
Epoch 2/3 - loss: 6.6536:  87%|████████▋ | 71/82 [00:03<00:00, 24.44it/s]
Epoch 2/3 - loss: 6.6723:  87%|████████▋ | 71/82 [00:03<00:00, 24.44it/s]
Epoch 2/3 - loss: 6.6723:  90%|█████████ | 74/82 [00:03<00:00, 24.78it/s]
Epoch 2/3 - loss: 6.6723:  94%|█████████▍| 77/82 [00:03<00:00, 25.04it/s]
Epoch 2/3 - loss: 6.6723:  98%|█████████▊| 80/82 [00:03<00:00, 23.08it/s]
Epoch 2/3 - loss: 6.6704:  99%|█████████▉| 81/82 [00:03<00:00, 23.08it/s]
Epoch 2/3 - loss: 6.6704: 100%|██████████| 82/82 [00:03<00:00, 22.11it/s]
Epoch 2 Validation loss: 6.7612

  0%|          | 0/82 [00:00<?, ?it/s]
Epoch 3/3 - loss: 6.6210:   1%|          | 1/82 [00:00<00:02, 27.12it/s]
Epoch 3/3 - loss: 6.6210:   5%|▍         | 4/82 [00:00<00:02, 31.04it/s]
Epoch 3/3 - loss: 6.6210:  10%|▉         | 8/82 [00:00<00:02, 30.94it/s]
Epoch 3/3 - loss: 6.1564:  13%|█▎        | 11/82 [00:00<00:02, 30.94it/s]
Epoch 3/3 - loss: 6.1564:  15%|█▍        | 12/82 [00:00<00:02, 31.24it/s]
Epoch 3/3 - loss: 6.1564:  20%|█▉        | 16/82 [00:00<00:02, 28.81it/s]
Epoch 3/3 - loss: 6.1564:  23%|██▎       | 19/82 [00:00<00:02, 27.55it/s]
Epoch 3/3 - loss: 6.1758:  26%|██▌       | 21/82 [00:00<00:02, 27.55it/s]
Epoch 3/3 - loss: 6.1758:  27%|██▋       | 22/82 [00:00<00:02, 26.98it/s]
Epoch 3/3 - loss: 6.1758:  30%|███       | 25/82 [00:00<00:02, 25.98it/s]
Epoch 3/3 - loss: 6.1758:  34%|███▍      | 28/82 [00:01<00:02, 26.23it/s]
Epoch 3/3 - loss: 6.1758:  38%|███▊      | 31/82 [00:01<00:01, 26.05it/s]
Epoch 3/3 - loss: 6.3027:  38%|███▊      | 31/82 [00:01<00:01, 26.05it/s]
Epoch 3/3 - loss: 6.3027:  41%|████▏     | 34/82 [00:01<00:02, 22.29it/s]
Epoch 3/3 - loss: 6.3027:  45%|████▌     | 37/82 [00:01<00:02, 19.85it/s]
Epoch 3/3 - loss: 6.3027:  49%|████▉     | 40/82 [00:01<00:02, 17.75it/s]
Epoch 3/3 - loss: 6.4225:  50%|█████     | 41/82 [00:01<00:02, 17.75it/s]
Epoch 3/3 - loss: 6.4225:  51%|█████     | 42/82 [00:01<00:02, 17.14it/s]
Epoch 3/3 - loss: 6.4225:  54%|█████▎    | 44/82 [00:01<00:02, 16.68it/s]
Epoch 3/3 - loss: 6.4225:  56%|█████▌    | 46/82 [00:02<00:02, 15.87it/s]
Epoch 3/3 - loss: 6.4225:  59%|█████▊    | 48/82 [00:02<00:02, 14.49it/s]
Epoch 3/3 - loss: 6.4225:  61%|██████    | 50/82 [00:02<00:02, 14.92it/s]
Epoch 3/3 - loss: 6.4587:  62%|██████▏   | 51/82 [00:02<00:02, 14.92it/s]
Epoch 3/3 - loss: 6.4587:  65%|██████▍   | 53/82 [00:02<00:01, 17.88it/s]
Epoch 3/3 - loss: 6.4587:  68%|██████▊   | 56/82 [00:02<00:01, 19.81it/s]
Epoch 3/3 - loss: 6.4587:  72%|███████▏  | 59/82 [00:02<00:01, 21.00it/s]
Epoch 3/3 - loss: 6.4870:  74%|███████▍  | 61/82 [00:02<00:00, 21.00it/s]
Epoch 3/3 - loss: 6.4870:  76%|███████▌  | 62/82 [00:02<00:00, 22.64it/s]
Epoch 3/3 - loss: 6.4870:  79%|███████▉  | 65/82 [00:02<00:00, 23.38it/s]
Epoch 3/3 - loss: 6.4870:  83%|████████▎ | 68/82 [00:03<00:00, 23.65it/s]
Epoch 3/3 - loss: 6.4870:  87%|████████▋ | 71/82 [00:03<00:00, 24.25it/s]
Epoch 3/3 - loss: 6.5054:  87%|████████▋ | 71/82 [00:03<00:00, 24.25it/s]
Epoch 3/3 - loss: 6.5054:  90%|█████████ | 74/82 [00:03<00:00, 24.68it/s]
Epoch 3/3 - loss: 6.5054:  94%|█████████▍| 77/82 [00:03<00:00, 25.01it/s]
Epoch 3/3 - loss: 6.5054:  98%|█████████▊| 80/82 [00:03<00:00, 23.01it/s]
Epoch 3/3 - loss: 6.4995:  99%|█████████▉| 81/82 [00:03<00:00, 23.01it/s]
Epoch 3/3 - loss: 6.4995: 100%|██████████| 82/82 [00:03<00:00, 22.19it/s]
Epoch 3 Validation loss: 6.8698
Training finished
Saving model to finetuned-model\model
```

微调已经完成了，通过 `FineTuner` 只需要几行代码就可以完成微调，就像魔法一样！
让我们看看这背后发生了什么，为什么可以这么简单？

1. `FineTuner` 会自动根据名称识别和加载模型，您只需要声明即可，就像例子中的 `moka-ai/m3e-small`，
   这会被识别为 M3E 类模型，`FinTuner` 还支持 `sentence-transformers`, `text2vec` 等模型
2. `FineTuner` 会自动识别数据格式，只要您的数据类型在 `FineTuner` 支持的范围内，
   `FineTuner` 就会自动识别并加以使用
3. `FineTuner` 会自动选择训练方式，`FineTuner` 会根据模型和数据集自动地选择训练方式，
   即 **对比学习** 或者 **CoSent** 等
4. `FineTuner` 会自动选择训练环境和超参数，`FineTuner` 会根据您的硬件环境自动选择训练设备，
   并根据模型、数据等各种信息自动建议最佳的超参数，`lr`, `batch_size` 等，
   当然您也可以自己手动进行调整
5. `FineTuner` 会自动保存微调记录和模型，`FineTuner` 会根据您的设置自动使用您环境中的 wandb, 
   tensorboard 等来记录微调过程，同时也会自动保存微调模型

总结一下，FineTuner 会自动完成微调所需的各种工作，只要您的数据类型在 FineTuner 支持的范围内！

## FineTuner 支持的数据类型

`FineTuner` 中 `dataset` 参数是一个可供迭代 (`for` 循环) 的数据集，
每次迭代会返回一个样本，这个样本应该是以下三种格式之一：

* PairRecord，句对样本
* TripletRecord，句子三元组样本
* ScoredPairRecord，带有分数的句对样本

只要您的数据集是这三种类型之一，`FineTuner` 就可以自动识别并使用。

```python
import os
from warnings
from uniem.data_structures import (
    RecordType, 
    PairRecord, 
    TripletRecord, 
    ScoredPairRecord
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
print(f"record_type: {[record_type.value for record_type in RecordType]}")
```

```
record_types: ['pair', 'triplet', 'scored_pair']
```

### PairRecord

PairRecord 就是句对样本，每一个样本都代表一对相似的句子，
字段的名称是 `text` 和 `text_pos`。

```python
pair_record = PairRecord(
    text = '肾结石如何治疗？', 
    text_pos = '如何治愈肾结石'
)
print(f'pair_record: {pair_record}')
```

```
pair_record: PairRecord(text='肾结石如何治疗？', text_pos='如何治愈肾结石')
```

### TripletRecord

TripletRecord 就是句子三元组样本，
在 PairRecord 的基础上增加了一个不相似句子负例，
字段的名称是 `text`、`text_pos` 和 `text_neg`。

```python
triplet_record = TripletRecord(
    text = '肾结石如何治疗？', 
    text_pos = '如何治愈肾结石', 
    text_neg = '胆结石有哪些治疗方法？'
)
print(f'triplet_record: {triplet_record}')
```

```
triplet_record: TripletRecord(text='肾结石如何治疗？', text_pos='如何治愈肾结石', text_neg='胆结石有哪些治疗方法？')
```

### ScoredPairRecord

ScoredPairRecord 就是带有分数的句对样本，
在 PairRecord 的基础上添加了句对的相似分数(程度)。
字段的名称是 `sentence1` 和 `sentence2`，以及 `label`。

```python
# 1.0 代表相似，0.0 代表不相似
scored_pair_record1 = ScoredPairRecord(
    sentence1='肾结石如何治疗？', 
    sentence2='如何治愈肾结石', 
    label=1.0
)
scored_pair_record2 = ScoredPairRecord(
    sentence1='肾结石如何治疗？', 
    sentence2='胆结石有哪些治疗方法？', 
    label=0.0
)
print(f'scored_pair_record: {scored_pair_record1}')
print(f'scored_pair_record: {scored_pair_record2}')


# 2.0 代表相似，1.0 代表部分相似，0.0 代表不相似
scored_pair_record1 = ScoredPairRecord(
    sentence1 = '肾结石如何治疗？', 
    sentence2 = '如何治愈肾结石', 
    label = 2.0
)
scored_pair_record2 = ScoredPairRecord(
    sentence1 = '肾结石如何治疗？', 
    sentence2 = '胆结石有哪些治疗方法？', 
    label = 1.0
)
scored_pair_record3 = ScoredPairRecord(
    sentence1 = '肾结石如何治疗？', 
    sentence2 = '失眠如何治疗', 
    label = 0
)
print(f'scored_pair_record: {scored_pair_record1}')
print(f'scored_pair_record: {scored_pair_record2}')
print(f'scored_pair_record: {scored_pair_record3}')
```

```
scored_pair_record: ScoredPairRecord(sentence1='肾结石如何治疗？', sentence2='如何治愈肾结石', label=1.0)
scored_pair_record: ScoredPairRecord(sentence1='肾结石如何治疗？', sentence2='胆结石有哪些治疗方法？', label=0.0)
scored_pair_record: ScoredPairRecord(sentence1='肾结石如何治疗？', sentence2='如何治愈肾结石', label=2.0)
scored_pair_record: ScoredPairRecord(sentence1='肾结石如何治疗？', sentence2='胆结石有哪些治疗方法？', label=1.0)
scored_pair_record: ScoredPairRecord(sentence1='肾结石如何治疗？', sentence2='失眠如何治疗', label=0)
```

## 微调示例

### 微调 M3E

> 医疗相似问题数据集

假设想要 HuggingFace 上托管的 `vegaviazhang/Med_QQpairs` 医疗数据集上做微调，先把数据集下载好：

```python
from datasets import load_dataset

# 下载数据
med_dataset_dict = load_dataset('vegaviazhang/Med_QQpairs')
# 查看一下 Med_QQpairs 的数据格式是不是在 FineTuner 支持的范围内
print(med_dataset_dict["train"][0])
print(med_dataset_dict["train"][1])
```

```

```

发现 `Med_QQpairs 数据集正好符合 `ScoredPairRecord` 的数据格式，
只是字段名称是 `question1` 和 `question2`，
只需要修改成 `sentence1` 和 `sentence2` 就可以直接进行微调了：

```python
from datasets import load_dataset
from uniem.finetuner import FineTuner

# data load
dataset = load_dataset('vegaviazhang/Med_QQpairs')["train"]
dataset = dataset.rename_columns({
    "question1": "sentence1",
    "question2": "sentence2",
})
#  Med_QQpairs 只有训练集，需要手动划分训练集和验证集
dataset = dataset.train_test_split(test_size = 0.1, seed = 42)
dataset['validation'] = dataset.pop('test')

# model finetune
finetuner = FineTuner.from_pretrained(
    'moka-ai/m3e-small', 
    dataset = dataset
)
fintuned_model = finetuner.run(epochs = 3)
```

### 微调 text2vec

> 猜谜数据集

要对一个猜谜的数据集进行微调，这个数据集是通过 json line 的形式存储的，先看看数据格式：

```python
import pandas as pd

df = pd.read_json(
    'https://raw.githubusercontent.com/wangyuxinwhy/uniem/main/examples/example_data/riddle.jsonl', 
    lines = True
)
records = df.to_dict('records')
print(records[0])
print(records[1])
```

这个数据集中有 `instruction` 和 `output`，可以把这两者看成一个相似句对。
这是一个典型的 PairRecord 数据集。PairRecord 需要 `text` 和 `text_pos` 两个字段，
因此我们需要对数据集的字段进行重新命名，以符合 PairRecord 的格式。

Fintuner 会根据模型名称自动识别模型类型，不需要额外处理。
这里选择微调 `text2vec-base-chinese-sentence`。

```python
import pandas as pd
from uniem.finetuner import FineTuner

# 读取 jsonl 文件
df = pd.read_json(
    'https://raw.githubusercontent.com/wangyuxinwhy/uniem/main/examples/example_data/riddle.jsonl', 
    lines = True
)
df = df.rename(columns = {
    'instruction': 'text', 
    'output': 'text_pos'
})

# 指定训练的模型为 m3e-small
finetuner = FineTuner.from_pretrained(
    'shibing624/text2vec-base-chinese-sentence', 
    dataset = df.to_dict('records')
)
fintuned_model = finetuner.run(
    epochs = 3, 
    output_dir = 'finetuned-model-riddle'
)
```

上面的两个示例分别展示了对 jsonl 本地 PairRecord 类型数据集，
以及 huggingface 远程 ScoredPair 类型数据集的读取和训练过程。
TripletRecord 类型的数据集的读取和训练过程与 PairRecord 类型的数据集的读取和训练过程类似，
这里就不再赘述了。也就是说，只要构造了符合 uniem 支持的数据格式的数据集，
就可以使用 `FineTuner` 对你的模型进行微调了。

`FineTuner` 接受的 `dataset` 参数，只要是可以迭代的产生有指定格式的字典 `dict` 就行了，
所以上述示例分别使用 `datasets.DatasetDict` 和 `list[dict]` 两种数据格式。

### 微调 sentences_transformers

`FineTuner` 在设计实现的时候也同时兼容了其他框架的模型，而不仅仅是 uniem！
比如，sentece_transformers 的 `all-MiniLM-L6-v2` 是一个广受欢迎的模型。
现在将使用前文提到过的 `Med_QQpairs` 数据对其进行微调。

```python
from datasets import load_dataset
from uniem.finetuner import FineTuner

dataset = load_dataset('vegaviazhang/Med_QQpairs')
dataset = dataset.rename_columns({'question1': 'sentence1', 'question2': 'sentence2'})

finetuner = FineTuner.from_pretrained(
    'sentence-transformers/all-MiniLM-L6-v2', 
    dataset = dataset
)
fintuned_model = finetuner.run(epochs = 3, batch_size = 32)
```

### 从头训练

除了可以在训练好的 Embedding 模型基础上进行微调外，还可以选择从一个预训练模型开始训练，
这个预训练模型可以是 BERT，RoBERTa，T5 等。

这里，将通过 `datasets` 的 `streaming` 的方式来使用一个数据规模较大的数据集，
并对一个只有两层的 BERT `uer/chinese_roberta_L-2_H-128` 进行微调。

```python
from datasets import load_dataset
from transformers import AutoTokenizer

from uniem.finetuner import FineTuner
from uniem.model import create_uniem_embedder

dataset = load_dataset('shibing624/nli-zh-all', streaming = True)
dataset = dataset.rename_columns({
    'text1': 'sentence1', 
    'text2': 'sentence2'
})

# 由于是从头训练，需要初始化 embedder 和 tokenizer。
# 当然，也可以选择新的 pooling 策略。
embedder = create_uniem_embedder(
    'uer/chinese_roberta_L-2_H-128', 
    pooling_strategy = 'cls'
)
tokenizer = AutoTokenizer.from_pretrained('uer/chinese_roberta_L-2_H-128')

finetuner = FineTuner(embedder, tokenizer = tokenizer, dataset=dataset)
fintuned_model = finetuner.run(
    epochs = 3, 
    batch_size = 32, 
    lr = 1e-3
)
```

### SGPT

`FineTuner` 在设计实现的时候还提供了更多的灵活性，以 `SGPT` 为例，`SGPT` 和前面介绍的模型主要有以下三点不同：

1. `SGPT` 使用 `GPT` 系列模型(transformer decoder)作为 Embedding 模型的基础模型
2. Embedding 向量的提取策略不再是 `LastMeanPooling`，而是根据 token position 来加权平均
3. 使用 bitfit 的微调策略，在微调时只对模型的 bias 进行更新

现在将效仿 `SGPT` 的训练策略，使用 `Med_QQpairs` 对 `GPT2` 进行微调。

```python
from datasets import load_dataset
from transformers import AutoTokenizer

from uniem.finetuner import FineTuner
from uniem.training_strategy import BitFitTrainging
from uniem.model import PoolingStrategy, create_uniem_embedder

dataset = load_dataset('vegaviazhang/Med_QQpairs')
dataset = dataset.rename_columns({
    'question1': 'sentence1', 
    'question2': 'sentence2'
})

embedder = create_uniem_embedder(
    'gpt2', 
    pooling_strategy = PoolingStrategy.last_weighted
)
tokenizer = AutoTokenizer.from_pretrained('gpt2')

finetuner = FineTuner(embedder, tokenizer, dataset = dataset)
finetuner.tokenizer.pad_token = finetuner.tokenizer.eos_token
finetuner.run(
    epochs = 3, 
    lr = 1e-3, 
    batch_size = 32, 
    training_strategy = BitFitTrainging()
)
```

# 参考

* [Embedding 中文模型-uniem](https://github.com/wangyuxinwhy/uniem)
* [uniem 微调示例](https://github.com/wangyuxinwhy/uniem/blob/main/examples/finetune.ipynb)
* [GitHub text2vec](https://github.com/shibing624/text2vec)
* [SentenceTransformers](https://www.sbert.net/index.html#)
