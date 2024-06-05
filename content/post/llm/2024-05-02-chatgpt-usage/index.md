---
title: ChatGPT 使用
author: 王哲峰
date: '2024-05-02'
slug: chatgpt-usage
categories:
  - tool
tags:
  - api
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

- [ChatGPT 基础](#chatgpt-基础)
    - [最强表示架构 Transformer 设计与演变](#最强表示架构-transformer-设计与演变)
    - [生成语言模型 GPT 进化与逆袭](#生成语言模型-gpt-进化与逆袭)
    - [利器强化学习 RLHF 流程与思想](#利器强化学习-rlhf-流程与思想)
- [ChatGPT 应用开发](#chatgpt-应用开发)
    - [ChatGPT Embedding 接口](#chatgpt-embedding-接口)
    - [ChatGPT 接口 + 提示词](#chatgpt-接口--提示词)
- [Jupyter 中使用 ChatGPT](#jupyter-中使用-chatgpt)
    - [安装](#安装)
    - [使用](#使用)
- [参考](#参考)
</p></details><p></p>

# ChatGPT 基础

## 最强表示架构 Transformer 设计与演变


## 生成语言模型 GPT 进化与逆袭



## 利器强化学习 RLHF 流程与思想



# ChatGPT 应用开发

## ChatGPT Embedding 接口

> 获取给定文本的向量表示

1. 设置 `OPENAI_API_KEY`

首先要做一些准备工作，主要是设置 `OPENAI_API_KEY`，这里建议读者用环境变量来获取，
而不要将自己的密钥明文写在任何代码文件里。当然，更不要上传到开源代码仓库。

```python
import os
import openai

# 用环境变量来获取
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
# 或直接填入自己专属的 API key(接口密钥)，不建议在正式场景下使用
OPENAI_API_KEY = "填入专属的 API key"

openai.api_key = OPENAI_API_KEY
```

2. 输入文本，指定相应模型，获取文本对应的 Embedding

```python
import openai

# 文本
text = "我喜欢你"

# 模型
model = "text-embedding-ada-002"

# 获取文本对应的 Embedding
emb_req = openai.Embedding.create(input = [text] model = model)

# 接口会返回所输入文本的向量表示
emb = emb_req.data[0].embedding
len(emb) == 1536
type(emb) == list
```

OpenAI 官方还提供了一个集成接口，既包括获取 Embedding，也包括计算相似度，
使用起来更加简单：

```python
from openai.embedding_utils import get_embedding,, cosine_similarity

text1 = "我喜欢你"
text2 = "我中意你"
text3 = "我不喜欢你"

# 注意默认的模型是 text-similarity-davinci-001，我们也可以换成 text-embedding-ada-002
emb1 = get_embedding(text1)
emb2 = get_embedding(text2)
emb3 = get_embedding(text3)

# 接口直接返回向量表示
len(emb1) == 12288
type(emb1) == list
```




## ChatGPT 接口 + 提示词

> 完成语义匹配任务




# Jupyter 中使用 ChatGPT

通过自定义魔法命令，可以在 Jupyter Notebook/Lab 中直接调用 ChatGPT。

## 安装

1. 首先，通过注册一个 ChatGPT 账号获取一个 ChatGPT 的 `api_key`。
   并将 `api_key` 导入脚本:

```python
import os
import openai
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local `.env` file

openai.api_key = os.getenv("OPEN_API_KEY")
print(openai.api_key)
```

2. 有了 `api_key` 后，可以通过如下代码自定义 Jupyter Magic Command。

```python
%%writefile chatgpt.py

import openai
from IPython.core.magic import (
    Magics, 
    magics_class, 
    line_magic, 
    cell_magic, 
    line_cell_magic,
)

openai.api_key = ""

def ask(promot):
    model = "gpt-3.5-turbo"
    messages = [{
        "role": "user",
        "content": prompt,
    }]
    response = openai.ChatCompletion.create(
        model = model,
        messages = messages,
        # this is the degree of randomness of the model's output
        temperature = 0,
    )
    result = response.choices[0].message["content"]
    
    return result


@magics_class
class ChatGPTMagics(Magics):

    @line_magic
    def chat(self, line):
        pass
        
    @cell_magic
    def gpt(self, line, cell):
        pass

    @line_cell_magic
    def chatgpt(self, line, cell = None):
        pass


def load_ipython_extension(ipython):
    """
    In order to actually use these magics, 
    you must register them with a running IPython.

    Any module file that define a function named `load_ipython_extension`
    can be loaded via `%load_ext_module.path` or be configured to be
    autoloaded by IPython at startup time.
    """
    ipython.register_magics(ChatGPTMagics)
```

## 使用

1. 导入魔法命令：

```python
%load_ext_chatgpt
```

2. 聊天：

```python
%%chatgpt

# 开始问问题：
问题 1.
问题 2.
```

# 参考

* [算法工程师如何优雅地使用 ChatGPT](https://mp.weixin.qq.com/s?__biz=MzU3OTQzNTU2OA==&mid=2247491771&idx=1&sn=f6a4780106f72c47c28f51cf3a303a46&chksm=fd648de4ca1304f29d2ad7e5282e71215c039dcd717fa350453dcfafaca53e6735b32ebbfd99&cur_album_id=2917869728717750275&scene=190#rd)
* 《ChatGPT 原理及应用开发》
