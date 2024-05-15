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
h1 {
    background-color: #2B90B6;
    background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
    background-size: 100%;
    -webkit-background-clip: text;
    -moz-background-clip: text;
    -webkit-text-fill-color: transparent;
    -moz-text-fill-color: transparent;
}
h2 {
    background-color: #2B90B6;
    background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
    background-size: 100%;
    -webkit-background-clip: text;
    -moz-background-clip: text;
    -webkit-text-fill-color: transparent;
    -moz-text-fill-color: transparent;
}
h3 {
    background-color: #2B90B6;
    background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
    background-size: 100%;
    -webkit-background-clip: text;
    -moz-background-clip: text;
    -webkit-text-fill-color: transparent;
    -moz-text-fill-color: transparent;
}
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

- [ChatGPT 原理](#chatgpt-原理)
- [ChatGPT 应用开发](#chatgpt-应用开发)
- [Jupyter 中使用 ChatGPT](#jupyter-中使用-chatgpt)
    - [安装](#安装)
    - [使用](#使用)
- [参考](#参考)
</p></details><p></p>

# ChatGPT 原理



# ChatGPT 应用开发





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
