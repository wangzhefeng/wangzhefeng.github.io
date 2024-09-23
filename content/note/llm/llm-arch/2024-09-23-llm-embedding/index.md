---
title: LLM 架构--Embedding API
author: 王哲峰
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

- [OpenAI API](#openai-api)
- [文心千帆 API](#文心千帆-api)
- [讯飞星火 API](#讯飞星火-api)
- [智谱 API](#智谱-api)
</p></details><p></p>

为了方便 Embedding API 调用，应将 API key 填入 `.env` 文件，代码将自动读取并加载环境变量。

## OpenAI API

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


response = openai_embedding(text = "要生成 embedding 的输入文本，字符串形式。")
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
返回的embedding类型为：list

embedding长度为：1536

embedding（前10）为：[0.03884002938866615, 0.013516489416360855, -0.0024250170681625605, -0.01655769906938076, 0.024130908772349358, -0.017382603138685226, 0.04206013306975365, 0.011498954147100449, -0.028245486319065094, -0.00674333656206727]

本次embedding model为：text-embedding-3-small

本次token使用情况为：Usage(prompt_tokens=12, total_tokens=12)
```

## 文心千帆 API

Embedding-V1 是基于百度文心大模型技术的文本表示模型，Access token 为调用接口的凭证，
使用 Embedding-V1 时应先凭 API Key、Secret Key 获取 Access token，
再通过 Access token 调用接口来 Embedding text。
同时千帆大模型平台还支持 `bge-large-zh` 等 Embedding model。

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
本次embedding id为：as-hvbgfuk29u

本次embedding产生时间戳为：1711435238

返回的embedding类型为:embedding_list

embedding长度为：384

embedding（前10）为：[0.060567744076251984, 0.020958080887794495, 0.053234219551086426, 0.02243831567466259, -0.024505289271473885, -0.09820500761270523, 0.04375714063644409, -0.009092536754906178, -0.020122773945331573, 0.015808865427970886]
```

## 讯飞星火 API

未开放

## 智谱 API

智谱有封装好的 SDK，直接调用即可。

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
