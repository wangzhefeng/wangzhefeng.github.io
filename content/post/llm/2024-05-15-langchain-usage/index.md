---
title: LangChain 使用
author: 王哲峰
date: '2024-05-15'
slug: langchain-usage
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

- [LangChain 框架及类似框架](#langchain-框架及类似框架)
- [LangChain 简介](#langchain-简介)
    - [LangChain 模块](#langchain-模块)
- [LangChain 快速使用](#langchain-快速使用)
    - [开发环境](#开发环境)
    - [LangChain 及管理工具安装](#langchain-及管理工具安装)
        - [langchain 库安装](#langchain-库安装)
        - [其他库安装](#其他库安装)
    - [构建一个简单的 LLM 应用](#构建一个简单的-llm-应用)
        - [语言模型](#语言模型)
        - [提示模版](#提示模版)
        - [输出解析器](#输出解析器)
        - [使用 LCEL 进行组合](#使用-lcel-进行组合)
        - [使用 LangSmith 进行观测](#使用-langsmith-进行观测)
        - [使用 LangServe 提供服务](#使用-langserve-提供服务)
    - [最佳安全实践](#最佳安全实践)
- [模型输入与输出](#模型输入与输出)
    - [大模型输入与输出](#大模型输入与输出)
    - [提示模板组件](#提示模板组件)
        - [基础提示模板](#基础提示模板)
        - [自定义提示模板](#自定义提示模板)
        - [使用 FewShotPromptTemplate](#使用-fewshotprompttemplate)
    - [大模型接口](#大模型接口)
- [链的构建](#链的构建)
- [RAG](#rag)
- [智能代理设计](#智能代理设计)
- [记忆组件](#记忆组件)
- [回调机制](#回调机制)
- [构建多模态机器人](#构建多模态机器人)
- [资源](#资源)
- [参考](#参考)
</p></details><p></p>

# LangChain 框架及类似框架

# LangChain 简介

LangChain 框架与大模型时代的开发范式紧密相关，它简化了大模型的集成过程，提供了一种新的 AI 应用构建方式，
允许开发者快速集成 GPT-3.5 等模型，增强了应用程序功能。

LangChain 作为一种大模型应用开发框架，针对当前 AI 应用开发中的一些关键挑战提供了有效的解决方案，概述如下：

* 数据时效性
    - GPT-3.5 等模型
* token 数量限制
* 网络连接限制
* 数据源整合限制


## LangChain 模块

* 模型 I/O 模块
* 检索模块
* 链模块
* 记忆模块
* 代理模块
* 回调模块

# LangChain 快速使用

## 开发环境

* Python 3.9
* LangChain

## LangChain 及管理工具安装

### langchain 库安装

pip:

```bash
$ pip install langchain
$ pip install langchain-experimental
```

conda:

```bash
$ conda install langchain -c conda-forge
```

源代码:

```bash
$ git clone https://github.com/langchain-ai/langchain.git
$ cd langchain
$ pip install -e .
```

### 其他库安装

LLM 应用托管服务 LangServe：用于一键部署 LangChain 应用

```bash
$ pip install langchain-cli
```

LLM 应用监控服务 LangSmith：用于调试和监控，默认包含在 LangChain 安装包中(如需单独使用，使用下面的命令)

```bash
$ pip install langsmith
```

OpenAI 的 GPT-3.5 模型需要安装 OpenAI SDK：

```bash
$ pip intall openai
```

python-dotenv：为了支持与多种外部资源的集成，安装 `python-dotenv` 来管理访问密钥：

```bash
$ pip install python-dotenv
```

[Python 后端框架 FastAPI](https://wangzhefeng.com/tool/python/fastapi/):

```bash
$ pip install fastapi
```

## 构建一个简单的 LLM 应用

### 语言模型


### 提示模版


### 输出解析器


### 使用 LCEL 进行组合


### 使用 LangSmith 进行观测


### 使用 LangServe 提供服务

## 最佳安全实践

* 限制权限
* 防范滥用
* 层层防护

# 模型输入与输出

## 大模型输入与输出

在传统的软件开发实践中，API 的调用者和提供者通常遵循详细的文档规定，
已确保输出的一致性和可预测性。然而，大模型的运作方式有所不同。
它们更像是带有不确定性的“黑盒”，其输出不仅难以精确控制，而且很大程度上依赖输入的质量。

输入的质量直接影响模型的输出效果。模糊、错误或不相关的输入可能导致输出偏离预期；
相反，清晰、准确的输入有助于模型更好地理解请求，提供更相关的输出。

CRISPE 框架由开源社区的 Matt Nigh 提出，它可以帮助我们为模型提供详细的背景，
任务目标和输出格式要求，这样的输入使得模型输出更加符合预期，内容更加清晰和详细。

| 概念 | 含义 | 示例 |  |  |
|----|----|----|----|----|
| CR(capacity and role, 能力与角色) | 希望模型扮演怎样的角色以及角色具备的能力 | 你是一个专门指导初学者编程的经验丰富的老师 |
| I(insight, 洞察力) | 完成任务依赖的背景信息 | 根据基础编程概念和最佳实践 |
| S(statement, 指令) | 希望模型做什么，任务的核心关键词和目标 | 解释 Python 中变量的作用，并给出实例 |
| P(personality, 个性) | 希望模型以什么风格或方式输出 | 使用简洁明了的语言，避免使用复杂的术语 |
| E(experiment, 尝试) | 要求模型提供多个答案，任务输出结果数量 | 提供两个不同的例子来展示变量的使用 |

上面描述的输入其实就是 **提示词(prompt)**，提示词在于大模型的交互中扮演着关键角色。
它们是提供给模型的输入文本，可以引导模型生成特定主题或类型的文本，
在自然语言处理任务中，提示词通常作为问题或任务的输入，而模型的输出则是对这些输入的回答或完成任务的结果。

## 提示模板组件

LangChain 的提示模板组件是一个强大的工具，用于简化和高效地构建提示词。
其优势在于能够让我们 **复用大部分静态内容，同时只需动态修改部分变量**。

### 基础提示模板

在程序中引入 `PromptTemplate` 类，构建一个基础的提示模板。
这个类允许我们定义一个包含变量的模板字符串，从而在需要时替换这些变量。


```python
from langchain.prompts import PromptTemplate

# 创建一个提示模板
template = PromptTemplate.from_template("翻译这段文字：{text}，风格：{style}")

# 使用具体的值格式化模板
formatted_prompt = template.format(text = "我爱编程", style = "诙谐有趣")
print(formatted_prompt)
```

### 自定义提示模板

### 使用 FewShotPromptTemplate

## 大模型接口




# 链的构建

# RAG


# 智能代理设计

# 记忆组件


# 回调机制


# 构建多模态机器人


# 资源





# 参考

* 《LangChain 编程-从入门到实践》
