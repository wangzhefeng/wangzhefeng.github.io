---
title: LLM 架构--RAG
author: 王哲峰
date: '2024-03-23'
slug: llm-rag
categories:
  - nlp
  - deeplearning
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

- [RAG 介绍](#rag-介绍)
     - [LLM 问题](#llm-问题)
     - [RAG 原理](#rag-原理)
     - [RAG 和 Fine-tune 对比](#rag-和-fine-tune-对比)
- [RAG 流程](#rag-流程)
- [RAG 模块](#rag-模块)
     - [向量化](#向量化)
     - [文档加载和切分](#文档加载和切分)
     - [数据库和向量检索](#数据库和向量检索)
     - [大模型模块](#大模型模块)
- [RAG 组件-LangChian](#rag-组件-langchian)
     - [LangChain 中的 RAG 组件](#langchain-中的-rag-组件)
     - [LLM 接入 LangChain](#llm-接入-langchain)
          - [基于 LangChain 调用 ChatGPT](#基于-langchain-调用-chatgpt)
               - [Model](#model)
               - [Prompt](#prompt)
               - [Output parser](#output-parser)
               - [完整的流程](#完整的流程)
          - [使用 LangChain 调用文心一言](#使用-langchain-调用文心一言)
               - [自定义 LLM 接入 langchain](#自定义-llm-接入-langchain)
               - [在 langchain 直接调用文心一言](#在-langchain-直接调用文心一言)
          - [使用 LangChain 调用讯飞星火](#使用-langchain-调用讯飞星火)
          - [使用 LangChain 调用智谱 GLM](#使用-langchain-调用智谱-glm)
               - [自定义 chatglm](#自定义-chatglm)
               - [自定义 chatglm 接入 LangChain](#自定义-chatglm-接入-langchain)
     - [基于 LangChain 构建检索问答链](#基于-langchain-构建检索问答链)
          - [加载数据库向量](#加载数据库向量)
          - [创建一个 LLM](#创建一个-llm)
          - [构建检索问答链](#构建检索问答链)
          - [检索问答链效果测试](#检索问答链效果测试)
          - [添加历史对话的记忆功能](#添加历史对话的记忆功能)
     - [基于 Streamlit 部署知识库助手](#基于-streamlit-部署知识库助手)
          - [构建应用程序](#构建应用程序)
          - [添加检索回答](#添加检索回答)
          - [部署应用程序](#部署应用程序)
- [RAG 组件-LlamaIndex](#rag-组件-llamaindex)
- [RAG 组件-dify](#rag-组件-dify)
- [参考](#参考)
</p></details><p></p>

# RAG 介绍

## LLM 问题

大型语言模型（LLM）相较于传统的语言模型具有更强大的能力，然而在某些情况下，
它们仍可能无法提供准确的答案。由于训练这些模型需要耗费大量时间，
因此它们所依赖的数据可能已经过时。此外，大模型虽然能够理解互联网上的通用事实，
但往往缺乏对特定领域或企业专有数据的了解，而这些数据对于构建基于 AI 的应用至关重要。

在大模型出现之前，**微调(fine-tuning)** 是一种常用的扩展模型能力的方法。
然而，随着模型规模的扩大和训练数据数据量的增加，微调变得越来越不适用于大多数情况，
除非需要模型以指定风格进行交流或充当领域专家的角色，
一个显著的例子是 OpenAI 将补全模型 GPT-3.5 改进为新的聊天模型 ChatGPT，
微调效果出色。微调不仅需要大量的高质量数据，还消耗巨大的计算资源和时间，
这对于许多个人和企业用户来说是昂贵且稀缺的资源。
因此，研究如何有效地利用专有数据来辅助大模型生成内容，成为了学术界和工业界的一个重要领域。
这不仅能够提高模型的实用性，还能够减轻对微调的依赖，使得 AI 应用更加高效和经济。

目前 LLM 面临的主要问题以及 RAG 的作用：

* 信息偏差/幻觉：LLM 有时会产生与客观事实不符的信息，导致用户接收到的信息不准确。
  RAG 通过检索数据源，辅助模型生成过程，确保输出内容的精确性和可信度，减少信息偏差。
* 知识更新滞后性：LLM 基于静态的数据集训练，这可能导致模型的知识更新滞后，
  无法及时反映最新的信息动态。RAG 通过实时检索最新数据，保持内容的时效性，确保信息的持续更新和准确性。
* 内容不可追溯：LLM 生成的内容往往缺乏明确的信息来源，影响内容的可信度。
  RAG 将生成内容与检索到的原始资料建立链接，增强了内容的可追溯性，从而提升了用户对生成内容的信任度。
* 领域专业知识能力欠缺：LLM 在处理特定领域的专业知识时，效果可能不太理想，
  这可能会影响到其在相关领域的回答质量。RAG 通过检索特定领域的相关文档，
  为模型提供丰富的上下文信息，从而提升了在专业领域内的问题回答质量和深度。
* 推理能力限制：面对复杂问题时，LLM 可能缺乏必要的推理能力，这影响了其对问题的理解和回答。
  RAG 结合检索到的信息和模型的生成能力，通过提供额外的背景知识和数据支持，增强了模型的推理和理解能力。
* 应用场景适应性受限：LLM 需在多样化的应用场景中保持高效和准确，
  但单一模型可能难以全面适应所有场景。RAG 使得 LLM 能够通过检索对应应用场景数据的方式，
  灵活适应问答系统、推荐系统等多种应用场景。
* 长文本处理能力较弱： LLM 在理解和生成长篇内容时受限于有限的上下文窗口，
  且必须按顺序处理内容，输入越长，速度越慢。RAG 通过检索和整合长文本信息，
  强化了模型对长上下文的理解和生成，有效突破了输入长度的限制，同时降低了调用成本，并提升了整体的处理效率。

## RAG 原理

为了解决大型语言模型在生成文本时面临的一系列挑战，提高模型的性能和输出质量，
研究人员提出了一种新的模型架构：**检索增强生成(RAG, Retrieval-Augmented Generation)**。
该架构巧妙地 **整合了从庞大知识库中检索到的相关信息，并以此为基础，指导大型语言模型生成更为精准的答案**，
即：RAG 的作用是 **帮助模型查找外部信息以改善其响应**，从而显著提升了回答的准确性与深度。

RAG 技术基于 **提示词(prompt)**，最早由 Facebook AI 研究机构(FAIR)与其合作者于 2021 年发布的论文 “Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks” 中提出。
RAG 有效地缓解了幻觉问题，提高了知识更新的速度，并增强了内容生成的可追溯性，
使得大型语言模型在实际应用中变得更加实用和可信。

RAG 技术十分强大，它已经被必应搜索、百度搜索以及其他大公司的产品所采用，旨在将最新的数据融入其模型。
在没有大量新数据、预算有限或时间紧张的情况下，这种方法也能取得不错的效果，而且它的原理足够简单。

## RAG 和 Fine-tune 对比

在提升大语言模型效果中，RAG 和 微调(Fine-tune)是两种主流的方法：

* RAG 结合了 **检索（从大型文档系统中获取相关文档片段）** 和 **生成（模型使用这些片段中的信息生成答案）** 两部分。
  RAG 通过在语言模型生成答案之前，先从广泛的文档数据库中检索相关信息，然后利用这些信息来引导生成过程，
  极大地提升了内容的准确性。
* 微调通过在特定数据集上进一步训练大语言模型，来提升模型在特定任务上的表现。

RAG 和 微调的对比可以参考下表：

| 特征比较 | RAG                                                                    | 微调                                                                       |
| -------- | ---------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| 知识更新 | 直接更新检索知识库，无需重新训练。信息更新成本低，适合动态变化的数据。 | 通常需要重新训练来保持知识和数据的更新。更新成本高，适合静态数据。         |
| 外部知识 | 擅长利用外部资源，特别适合处理文档或其他结构化/非结构化数据库。        | 将外部知识学习到 LLM 内部。                                                |
| 数据处理 | 对数据的处理和操作要求极低。                                           | 依赖于构建高质量的数据集，有限的数据集可能无法显著提高性能。               |
| 模型定制 | 侧重于信息检索和融合外部知识，但可能无法充分定制模型行为或写作风格。   | 可以根据特定风格或术语调整 LLM 行为、写作风格或特定领域知识。              |
| 可解释性 | 可以追溯到具体的数据来源，有较好的可解释性和可追踪性。                 | 黑盒子，可解释性相对较低。                                                 |
| 计算资源 | 需要额外的资源来支持检索机制和数据库的维护。                           | 依赖高质量的训练数据集和微调目标，对计算资源的要求较高。                   |
| 推理延迟 | 增加了检索步骤的耗时                                                   | 单纯 LLM 生成的耗时                                                        |
| 降低幻觉 | 通过检索到的真实信息生成回答，降低了产生幻觉的概率。                   | 模型学习特定领域的数据有助于减少幻觉，但面对未见过的输入时仍可能出现幻觉。 |
| 伦理隐私 | 检索和使用外部数据可能引发伦理和隐私方面的问题。                       | 训练数据中的敏感信息需要妥善处理，以防泄露。                               |

# RAG 流程

RAG 技术在具体实现方式上可能有所变化，但在概念层面，将其融入应用通常包括以下几个步骤（见下图）：

![img](images/RAG.png)

![img](images/RAG-APP.png)

1. 用户提交一个问题
2. RAG 系统搜索可能回答这个问题的相关文档。这些文档通常包含了专有数据，
   并被存储在某种形式的文档索引里
3. RAG 系统构建一个提示词，它结合了用户输入、相关文档以及对大模型的提示词，
   引导其使用相关文档来回答用户的问题
4. RAG 系统将这个提示词发送给大模型
5. 大模型基于提供的上下文返回对用户问题的回答，这就是系统的输出结果

RAG 是一个完整的系统，其具体实现方式上基本流程是：

![img](images/C1-2-RAG.png)

1. 数据处理
    - 对原始数据进行清洗和处理
    - 将处理后的数据转化为检索模型可以使用的格式 
    - 将处理后的数据存储在对应的数据库中
2. 检索/索引：
    - 将用户的问题输入到检索系统中，从数据库中检索相关信息
3. 增强
    - 对检索到的信息进行处理和增强，以便生成模型可以更好地理解和使用
4. 生成
    - 将增强后的信息输入到生成模型中，生成模型根据这些信息生成答案



# RAG 模块

* 一个向量化模块，用来将文档片段向量化
* 一个文档加载和切分的模块，用来加载文档并切分成文档片段
* 一个数据库来存放文档片段和对应的向量表示
* 一个检索模块，用来根据 Query(问题) 检索相关的文档片段
* 一个大模型模块，用来根据检索出来的文档回答用户的问题

## 向量化

> Embedding

手动实现一个向量化的类，这是 RAG 架构的基础。向量化的类主要用来将文档片段向量化，
将一段文本映射为一个向量。这里设置一个 `Embedding` 基类，
这样我们在用其他的 Embedding 模型的时候，只需要继承这个基类，
然后在此基础上进行修改即可，方便代码扩展。

## 文档加载和切分

接下来实现一个文档加载、切分的类，这个类主要是用来加载文档并切分成文档片段。

那么需要切分什么文档呢？这个文档可以是一篇文章、一本书、一段对话、一段代码等等。
这个文档的内容可以是任何的，只要是文本就行，比如：PDF 文件、MD 文件、TXT 文件等。

把文件内容都读取之后，还需要切分。按 Token 的长度来切分文档。
可以设置一个最大的 Token 长度，然后根据这个最大的 Token 长度来切分文档。
这样切分出来的文档片段就是一个一个的差不多相同长度的文档片段了。
不过在切分的时候要注意，片段与片段之间最好要有一些重叠的内容，
这样才能保证检索的时候能够检索到相关的文档片段。
还有就是切分文档的时候最好以句子为单位，也就是按 `\n` 进行粗切分，
这样可以基本保证句子内容是完整的。

## 数据库和向量检索

做好了文档切分后，也做好了 Embedding 模型的加载。
接下来就得设计一个向量数据库用来存放文档片段和对应的向量表示了。

并且需要设计一个检索模块，用来根据 Query （问题）检索相关的文档片段。

一个数据库对于最小 RAG 架构来说，需要实现几个功能:

* persist：数据库持久化，本地保存
* load_vector：从本地加载数据库
* get_vector：获得文档的向量表示
* query：根据问题检索相关的文档片段

以上四个模块就是一个最小的 RAG 结构数据库需要实现的功能

query 方法具体实现：

1. 首先，先把用户提出的问题向量化
2. 然后, 在数据库中检索相关的文档片段
3. 最后返回检索到的文档片段

在向量检索的时候仅使用 Numpy 进行加速。

## 大模型模块

这个模块主要是用来根据检索出来的文档回答用户的问题。

# RAG 组件-LangChian

在实际的生产环境中，通常会面对来自多种渠道的数据，其中很大一部分是复杂的非机构化数据，
处理这些数据，特别是提取和预处理，往往是耗费精力的任务之一。
因此 LangChain 提供了专门的文档加载和分割模块。RAG 技术的每个阶段都在 LangChain 中得到完整的实现。

## LangChain 中的 RAG 组件

TODO

## LLM 接入 LangChain

LangChain 为基于 LLM 开发自定义应用提供了高效的开发框架，便于开发者迅速地激发 LLM 的强大能力，
搭建 LLM 应用。LangChain 也同样支持多种大模型，内置了 OpenAI、LLAMA 等大模型的调用接口。
但是，LangChain 并没有内置所有大模型，它通过允许用户自定义 LLM 类型，来提供强大的可扩展性。

### 基于 LangChain 调用 ChatGPT

LangChain 提供了对于多数大模型的封装，
基于 LangChain 的接口可以便捷地调用 ChatGPT 并将其集合在以 LangChain 为基础框架搭建的个人应用中。

注：基于 LangChain 接口调用 ChatGPT 同样需要配置个人密钥。

#### Model

从 `langchain.chat_models` 导入 OpenAI 的对话模型 `ChatOpenAI`。除了 OpenAI 以外，
`langchain.chat_models` 还集成了其他对话模型。

```python
import os
import openai
from dotenv import load_dotenv, find_dotenv
from langchain.openai import ChatOpenAI

# 读取本地的环境变量
_ = load_dotenv(find_dotenv())

# 获取环境变量 OPENAI_API_KEY
openai_api_key = os.environ("OPENAI_API_KEY")

# OpenAI API 密钥在环境变量中设置
llm = ChatOpenAI(temperature = 0.0)
# 手动指定 API 密钥
llm = ChatOpenAI(temperature = 0.0, openai_api_key = "YOUR_API_KEY")

output = llm.invoke("请你自我介绍以下自己！")
output
```

可以看到，默认调用的是 ChatGPT-3.5 模型。另外，几种常用的超参数设置包括：

* `model_name`：所要使用的模型，默认为 `'gpt-3.5-turbo'`，参数设置与 OpenAI 原生接口参数设置一致。
* `temperature`：温度系数，取值同原生接口。
* `openai_api_key`：OpenAI API key，如果不使用环境变量设置 API Key，也可以在实例化时设置。
* `openai_proxy`：设置代理，如果不使用环境变量设置代理，也可以在实例化时设置。
* `streaming`：是否使用流式传输，即逐字输出模型回答，默认为 `False`，此处不赘述。
* `max_tokens`：模型输出的最大 token 数，意义及取值同上。

#### Prompt

在开发大模型应用时，大多数情况下不会直接将用户的输入直接传递给 LLM。
通常，他们会将用户输入添加到一个较大的文本中，称为提示模板(Prompt Template)，
该文本提供有关当前特定任务的附加上下文。

`PromptTemplates` 正是帮助解决这个问题，它们捆绑了从用户输入到完全格式化的提示的所有逻辑。
这可以非常简单地开始。例如，生成上述字符串的提示就是。

聊天模型的接口是基于消息（message），而不是原始的文本。
`PromptTemplates` 也可以用于产生消息列表，在这种样例中，
prompt 不仅包含了输入内容信息，也包含了每条 message 的信息(角色、在列表中的位置等)。
通常情况下，一个 `ChatPromptTemplate` 是一个 `ChatMessageTemplate` 的列表。
每个 `ChatMessageTemplate` 包含格式化该聊天消息的说明（其角色以及内容）。

```python
from langchain.prompts.chat import ChatPromptTemplate

template = "你是一个翻译助手，可以帮助我将 {input_language} 翻译成 {output_language}"
human_template = "{text}"
text = "我带着比身体重的行李，\
游入尼罗河底，\
经过几道闪电 看到一堆光圈，\
不确定是不是这里。\"

chat_prompt = ChatPromptTemplate([
     ("system", template),
     ("human", human_template),
])

message = chat_prompt.format_messages(
     input_language = "中文", 
     output_language = "英文", 
     text = text
)
print(message)

output = llm.invoke(message)
print(output)
```

#### Output parser

`OutputParsers` 将语言模型的原始输出转换为可以在下游使用的格式。
`OutputParser` 有几种主要类型，包括：

* 将 LLM 文本转换为结构化信息(例如 JSON)
* 将 `ChatMessage` 转换为字符串
* 将除消息之外的调用返回的额外信息（如 OpenAI 函数调用）转换为字符串

最后，我们将模型输出传递给 `output_parser`，它是一个 `BaseOutputParser`，
这意味着它接受字符串或 `BaseMessage` 作为输入。
`StrOutputParser` 特别简单地将任何输入转换为字符串。

```python
from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()
output_parser.invoke(output)
```

从上面结果可以看到，我们通过输出解析器成功将 ChatMessage 类型的输出解析为了字符串。

#### 完整的流程

现在可以将所有这些组合成一条链，该链将获取输入变量，将这些变量传递给提示模板以创建提示，
将提示传递给语言模型，然后通过（可选）输出解析器传递输出。下面使用 LCEL 这种语法去快速实现一条链（chain）。

```python
chain = chat_prompt | llm | output_parser
chain.invoke({
     "input_language": "中文",
     "output_language": "英文",
     "text": text,
})

text = "I carried luggage heavier than my body and dived into the bottom of the Nile River. After passing through several flashes of lightning, I saw a pile of halos, not sure if this is the place."
chain.invoke({
     "input_language": "英文", 
     "output_language": "中文",
     "text": text
})
```

### 使用 LangChain 调用文心一言

通过 LangChain 框架来调用百度文心大模型，以将文心模型接入到应用框架中。

#### 自定义 LLM 接入 langchain



#### 在 langchain 直接调用文心一言

### 使用 LangChain 调用讯飞星火



### 使用 LangChain 调用智谱 GLM

#### 自定义 chatglm

由于 LangChain 中提供的 ChatGLM 已不可用，因此需要自定义一个 LLM。

```python
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
from typing import Any, List, Mapping, Optional, Dict
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from zhipuai import ZhipuAI

# 继承自 langchain.llms.base.LLM
class ZhipuAILLM(LLM):
    # 默认选用 glm-4
    model: str = "glm-4"
    # 温度系数
    temperature: float = 0.1
    # API_Key
    api_key: str = None
    
    def _call(self, prompt : str, stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any):
        client = ZhipuAI(
            api_key = self.api_key
        )

        def gen_glm_params(prompt):
            '''
            构造 GLM 模型请求参数 messages

            请求参数：
                prompt: 对应的用户提示词
            '''
            messages = [{"role": "user", "content": prompt}]
            return messages
        
        messages = gen_glm_params(prompt)
        response = client.chat.completions.create(
            model = self.model,
            messages = messages,
            temperature = self.temperature
        )

        if len(response.choices) > 0:
            return response.choices[0].message.content
        return "generate answer error"


    # 首先定义一个返回默认参数的方法
    @property
    def _default_params(self) -> Dict[str, Any]:
        """获取调用API的默认参数。"""
        normal_params = {
            "temperature": self.temperature,
            }
        # print(type(self.model_kwargs))
        return {**normal_params}

    @property
    def _llm_type(self) -> str:
        return "Zhipu"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {**{"model": self.model}, **self._default_params}
```

#### 自定义 chatglm 接入 LangChain

```python
import os
from zhipuai_llm import ZhipuAILLM
from dotenv import find_dotenv, load_dotenv

# 读取本地/项目的环境变量
_ = load_dotenv(find_dotenv())

# 获取环境变量 API_KEY
api_key = os.environ["ZHIPUAI_API_KEY"]

zhipuai_model = ZhipuAILLM(model = "glm-4", temperature = 0.1, api_key = api_key)  # model="glm-4-0520"
zhipuai_model("你好，请自我介绍以下！")
```

## 基于 LangChain 构建检索问答链

在[这里]()介绍了如何根据自己的本地知识文档，搭建一个向量知识库。
使用搭建好的向量数据库，对 query 查询问题进行召回，
并将召回结果和 query 结合起来构建 prompt，输入到大模型中进行问答。

### 加载数据库向量



### 创建一个 LLM


### 构建检索问答链



### 检索问答链效果测试


### 添加历史对话的记忆功能


## 基于 Streamlit 部署知识库助手

当对知识库和 LLM 已经有了基本的理解，现在是将它们巧妙地融合并打造成一个富有视觉效果的界面的时候了。
这样的界面不仅对操作更加便捷，还能便于与他人分享。

> Streamlit 是一种快速便捷的方法，可以直接在 Python 中通过友好的 Web 界面演示机器学习模型。
> 在构建了机器学习模型后，如果想构建一个 demo 给其他人看，也许是为了获得反馈并推动系统的改进，
> 或者只是因为觉得这个系统很酷，所以想演示一下：Streamlit 可以通过 Python 接口程序快速实现这一目标，
> 而无需编写任何前端、网页或 JavaScript 代码。

### 构建应用程序

```python
# streamlit_app.py
import streamlit as st
from langchain_openai import ChatOpenAI


def generate_response(input_text, openai_api_key):
     """
     定义一个函数，使用用户密钥对 OpenAI API 进行身份验证、发送提示并获取 AI 生成的响应。
     该函数接受用户的提示作为参数，并使用 st.info 来在蓝色框中显示 AI 生成的响应。
     """
     llm = ChatOpenAI(temperature = 0.7, openai_api_key = openai_api_key)
     output = llm.invoke(input_text)
     output_parser = StrOutputParser()
     output = output_parser.invoke(output)
     # st.info(output)
     return output


# Streamlit 应用程序界面
def main():
     # 创建应用程序的标题
     st.title("🦜🔗 动手学大模型应用开发")

     # 添加一个文本输入框，供用户输入其 OpenAI API 密钥
     openai_api_key = st.sidebar.text_input("OpenAI API Key", type = "password")

     # 使用 st.form() 创建一个文本框 st.text_area() 供用户输入。
     # 当用户点击 Submit 时，generate_response 将使用用户的输入作为参数来调用该函数
     with st.form("my_form"):
          text = st.text_area(
               "Enter text:", 
               "What are the three key pieces of advice for learning how to code?"
          )
          
          submitted = st.form_submit_button("Submit")

          if not openai_api_key.startswith("sk-"):
               st.warning("Please enter your OpenAI API key!", icon = "")
          if submitted and openai_api_key.startswith("sk-"):
               generate_response(text, openai_api_key)
  
     # 通过使用 st.session_state 来存储对话历史，
     # 可以在用户与应用程序交互时保留整个对话的上下文，
     # 用于跟踪对话历史
     if "messages" not in st.session_state:
          st.session_state.messages = []
     
     messages = st.container(height = 300)
     if prompt := st.chat_input("Say something"):
          # 将用户输入添加到对话历史中
          st.session_state.messages.appen({
               "role": "user",
               "text": prompt,
          })
          
          # 调用 respond 函数获取回答
          answer = generate_response(prompt, openai_api_key)
          # 检查回答是否为 None
          if answer is not None:
               # 将 LLM 的回答添加到对话历史中
               st.session_state.messages.append({
                    "role": "assistant",
                    "text": answer,
               })
          
          # 显示整个对话历史
          for message in st.session_state.messages:
               if message["role"] == "user":
                    messages.chat_message("user").write(message["text"])
               else:
                    messages.chat_message("assistant").write(message["text"])
```

```bash
$ streamlit run streamlit_app.py
```

### 添加检索回答

构建检索问答链代码：

* `get_vectordb` 函数返回持久化后的向量知识库
* `get_chat_qa_chain` 函数返回调用带有历史记录的检索问答链后的结果
* `get_qa_chain` 函数返回调用不带有历史记录的检索问答链后的结果

```python
def get_vectordb():
     """
     函数返回持久化后的向量知识库
     """
     # 定义 Embeddings
     embedding = ZhipuAIEmbeddings()
     # 向量数据库持久化路径
     persist_directory = "data_base/vector_db/chroma"
     # 加载数据库
     vectordb = Chroma(
          persist_directory = persist_directory,
          embedding_function = embedding,
     )
     return vectordb

def get_chat_qa_chain(question: str, openai_api_key: str):
     """
     带有历史记录的问答链
     """
     vectordb = get_vectordb()
     llm = ChatOpenAI(
          model_name = "gpt-3.5-turbo", 
          temperature = 0, 
          openai_api_key = openai_api_key
     )
     memory = ConversationBufferMemory(
          memory_key = "chat_history",  # 与 prompt 的输入变量保持一致
          return_messages = True,  # 将消息列表的形式返回聊天记录，而不是单个字符串
     )
     retriever = vectordb.as_retriever()
     qa = ConversationBufferMemory.from_llm(
          llm, 
          retriever = retriever,
          memory = memory,
     )
     result = qa({
          "question": question
     })
     
     return result["answer"]


def get_qa_chain(question: str, openai_api_key: str):
     """
     不带历史记录的问答链
     """
     vectordb = get_vectordb()
     llm = ChatOpenAI(
          model = "gpt-3.5-turbo",
          temperature = 0,
          opanai_api_key = oepnai_api_key,
     )
     template = """"使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
        案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
        {context}
        问题: {question}
        """
     QA_CHAIN_PROMPT = PromptTemplate(
          input_variables = ["context", "question"],
          template = template,
     )
     qa_chain = RetrievalQA.from_chain_type(
          llm,
          retriever = vectordb.as_retriever(),
          return_source_documents = True,
          chain_type_kwargs = {"prompt": QA_CHAIN_PROMPT}
     )
     result = qa_chain({"query": question})

     return result["result"]
```

然后，添加一个单选按钮部件 `st.radio`，选择进行回答模式：

* `None` 不使用检索问答的普通模式
* `qa_chain` 不带历史记录的检索问答模式
* `chat_qa_chain` 带历史记录的检索问答模式

```python
selected_method = st.radio(
     "你想选择哪种模式进行对话？",
     [
          "None", 
          "qa_chain", 
          "chat_qa_chain"
     ],
     caption = [
          "不使用检索回答的普通模式", 
          "不带历史记录的检索问答模式", 
          "带历史记录的检索问答模式"
     ]
)
```

最后，进入页面，首先先输入 `OPEN_API_KEY`（默认），
然后点击单选按钮选择进行问答的模式，最后在输入框输入你的问题，按下回车即可。

完整代码：

```python
# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores.chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser

from embedding_api.zhipuai_embedding import ZhipuAIEmbeddings
from dotenv import load_dotenv, find_dotenv

# 将父目录放入系统路径中
sys.path.append("../knowledge_lib") 
# 读取本地 .env 文件
_ = load_dotenv(find_dotenv())
# 载入 ***_API_KEY
os.environ["OPENAI_API_BASE"] = "https://api.chatgptid.net/v1"
zhipuai_api_key = os.environ["ZHIPUAI_API_KEY"]
# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def generate_response(input_text, openai_api_key):
     """
     定义一个函数，使用用户密钥对 OpenAI API 进行身份验证、发送提示并获取 AI 生成的响应。
     该函数接受用户的提示作为参数，并使用 st.info 来在蓝色框中显示 AI 生成的响应。
     """
     llm = ChatOpenAI(temperature = 0.7, openai_api_key = openai_api_key)
     output = llm.invoke(input_text)
     output_parser = StrOutputParser()
     output = output_parser.invoke(output)
     # st.info(output)
     return output


def get_vectordb():
     """
     函数返回持久化后的向量知识库
     """
     # 定义 Embeddings
     embedding = ZhipuAIEmbeddings()
     # 向量数据库持久化路径
     persist_directory = "data_base/vector_db/chroma"
     # 加载数据库
     vectordb = Chroma(
          persist_directory = persist_directory,
          embedding_function = embedding,
     )

     return vectordb


def get_chat_qa_chain(question: str, openai_api_key: str):
     """
     带有历史记录的问答链
     """
     vectordb = get_vectordb()
     llm = ChatOpenAI(
          model_name = "gpt-3.5-turbo", 
          temperature = 0, 
          openai_api_key = openai_api_key
     )
     memory = ConversationBufferMemory(
          memory_key = "chat_history",  # 与 prompt 的输入变量保持一致
          return_messages = True,  # 将消息列表的形式返回聊天记录，而不是单个字符串
     )
     retriever = vectordb.as_retriever()
     qa = ConversationBufferMemory.from_llm(
          llm, 
          retriever = retriever,
          memory = memory,
     )
     result = qa({
          "question": question
     })
     
     return result["answer"]


def get_qa_chain(question: str, openai_api_key: str):
     """
     不带历史记录的问答链
     """
     vectordb = get_vectordb()
     llm = ChatOpenAI(
          model = "gpt-3.5-turbo",
          temperature = 0,
          opanai_api_key = openai_api_key,
     )
     template = """"使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
        案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
        {context}
        问题: {question}
        """
     QA_CHAIN_PROMPT = PromptTemplate(
          input_variables = ["context", "question"],
          template = template,
     )
     qa_chain = RetrievalQA.from_chain_type(
          llm,
          retriever = vectordb.as_retriever(),
          return_source_documents = True,
          chain_type_kwargs = {"prompt": QA_CHAIN_PROMPT}
     )
     result = qa_chain({"query": question})

     return result["result"]




# Streamlit 应用程序界面
def main():
     # 创建应用程序的标题
     st.title("🦜🔗 动手学大模型应用开发")
     # 添加一个文本输入框，供用户输入其 OpenAI API 密钥
     openai_api_key = st.sidebar.text_input("OpenAI API Key", type = "password")
     # 添加一个选择按钮来选择不同的模型
     # selected_method = st.sidebar.selectbox(
     #      "选择模式", 
     #      [
     #           "None", 
     #           "qa_chain", 
     #           "chat_qa_chain"
     #      ]
     # )
     selected_method = st.radio(
          "你想选择哪种模式进行对话？",
          [
               "None", 
               "qa_chain", 
               "chat_qa_chain"
          ],
          caption = [
               "不使用检索回答的普通模式", 
               "不带历史记录的检索问答模式", 
               "带历史记录的检索问答模式"
          ]
     )

     # 用于跟踪对话历史
     # 通过使用 st.session_state 来存储对话历史，
     # 可以在用户与应用程序交互时保留整个对话的上下文
     if "messages" not in st.session_state:
          st.session_state.messages = []
     
     # 当用户点击 Submit 时，generate_response 将使用用户的输入作为参数来调用该函数
     # with st.form("my_form"):
     #      text = st.text_area("Enter text:", "What are the three key pieces of advice for learning how to code?")
     #      submitted = st.form_submit_button("Submit")
     #      if not openai_api_key.startswith("sk-"):
     #           st.warning("Please enter your OpenAI API key!", icon = "")
     #      if submitted and openai_api_key.startswith("sk-"):
     #           generate_response(text, openai_api_key)
   
     messages = st.container(height = 300)
     if prompt := st.chat_input("Say something"):
          # 将用户输入添加到对话历史中
          st.session_state.messages.appen({
               "role": "user",
               "text": prompt,
          })
          # 调用 respond 函数获取回答
          if selected_method == "None":
               answer = generate_response(prompt, openai_api_key)
          if selected_method == "qa_chain":
               answer = get_qa_chain(prompt, openai_api_key)
          elif selected_method == "chat_qa_chain":
               answer = get_chat_qa_chain(prompt, openai_api_key)
          
          # 检查回答是否为 None
          if answer is not None:
               # 将 LLM 的回答添加到对话历史中
               st.session_state.messages.append({
                    "role": "assistant",
                    "text": answer,
               })
          
          # 显示整个对话历史
          for message in st.session_state.messages:
               if message["role"] == "user":
                    messages.chat_message("user").write(message["text"])
               else:
                    messages.chat_message("assistant").write(message["text"])

if __name__ == "__main__":
    main()
```

### 部署应用程序

要将应用程序部署到 Streamlit Cloud，请执行以下步骤：

1. 为应用程序创建 GitHub 存储库，存储库应包含两个文件：

```
your-repository/
     |_ streamlit_app.py
     |_ requirements.txt
```

2. 转到 [Streamlit Community Cloud](https://share.streamlit.io/)，单击工作区中的 `New app` 按钮，
   然后指定存储库、分支和主文件路径。或者，您可以通过选择自定义子域来自定义应用程序的 URL。
3. 点击 `Deploy!` 按钮。
4. 应用程序现在将部署到 Streamlit Community Cloud，并且可以访问应用。

优化方向：

* 界面中添加上传本地文档，建立向量数据库的功能
* 添加多种LLM 与 embedding方法选择的按钮
* 添加修改参数的按钮
* 更多...

# RAG 组件-LlamaIndex

# RAG 组件-dify




# 参考

* [动手做一个最小RAG——TinyRAG](https://mp.weixin.qq.com/s?__biz=MzIyNjM2MzQyNg==&mid=2247660972&idx=1&sn=0bf6fe4d0854015d18263b49cd7b81ef&chksm=e98387af128051c9cb6914cee71626e5afb79ab8439966c508999270ca5b4048ad51a386fc2f&scene=0&xtrack=1)
* [GitHub](https://github.com/KMnO4-zx/TinyRAG)
