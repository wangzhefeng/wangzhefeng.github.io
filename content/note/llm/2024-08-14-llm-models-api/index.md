---
title: LLM：模型调用
author: 王哲峰
date: '2024-08-14'
slug: llm-models-api
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

- [基本概念](#基本概念)
    - [Prompt](#prompt)
    - [Temperature](#temperature)
    - [System Prompt](#system-prompt)
- [LLM API](#llm-api)
    - [OpenAI ChatGPT](#openai-chatgpt)
        - [API key 申请](#api-key-申请)
        - [API key 配置](#api-key-配置)
        - [加载环境变量](#加载环境变量)
        - [调用 OpenAI API](#调用-openai-api)
    - [百度文心一言](#百度文心一言)
        - [API key 申请](#api-key-申请-1)
        - [API key 配置](#api-key-配置-1)
        - [加载环境变量](#加载环境变量-1)
        - [调用文心千帆 API](#调用文心千帆-api)
        - [ERNIE SDK](#ernie-sdk)
            - [API 申请](#api-申请)
            - [API Key 配置](#api-key-配置-2)
            - [调用 Ernie Bot API](#调用-ernie-bot-api)
    - [讯飞星火](#讯飞星火)
        - [API key 申请](#api-key-申请-2)
        - [API key 配置](#api-key-配置-3)
        - [加载环境变量](#加载环境变量-2)
        - [模型调用](#模型调用)
    - [智谱 GLM](#智谱-glm)
        - [API key 申请](#api-key-申请-3)
        - [API key 配置](#api-key-配置-4)
        - [调用智谱 GLM API](#调用智谱-glm-api)
- [LLM 接入 LangChain](#llm-接入-langchain)
    - [基于 LangChain 调用 ChatGPT](#基于-langchain-调用-chatgpt)
        - [Model](#model)
        - [Prompt](#prompt-1)
        - [Output parser](#output-parser)
        - [完整的流程](#完整的流程)
    - [使用 LangChain 调用文心一言](#使用-langchain-调用文心一言)
        - [自定义 LLM 接入 langchain](#自定义-llm-接入-langchain)
        - [在 langchain 直接调用文心一言](#在-langchain-直接调用文心一言)
    - [使用 LangChain 调用讯飞星火](#使用-langchain-调用讯飞星火)
    - [使用 LangChain 调用智谱 GLM](#使用-langchain-调用智谱-glm)
        - [自定义 chatglm](#自定义-chatglm)
        - [自定义 chatglm 接入 LangChain](#自定义-chatglm-接入-langchain)
- [参考资料](#参考资料)
</p></details><p></p>

# 基本概念

## Prompt

Prompt 最初是 NLP（自然语言处理）研究者为下游任务设计出来的一种任务专属的输入模板，
类似于一种任务（例如：分类，聚类等）会对应一种 Prompt。
在 ChatGPT 推出并获得大量应用之后，Prompt 开始被推广为给大模型的所有输入。
即，每一次访问大模型的输入为一个 Prompt，而大模型给我们的返回结果则被称为 Completion。

## Temperature

LLM 生成是具有随机性的，在模型的顶层通过选取不同预测概率的预测结果来生成最后的结果。
一般可以通过控制 `temperature` 参数来控制 LLM 生成结果的随机性与创造性。

`temperature` 一般取值在 0~1 之间，当取值较低接近 0 时，预测的随机性会较低，
产生更保守、可预测的文本，不太可能生成意想不到或不寻常的词。
当取值较高接近 1 时，预测的随机性会较高，所有词被选择的可能性更大，
会产生更有创意、多样化的文本，更有可能生成不寻常或意想不到的词。

对于不同的问题与应用场景，可能需要设置不同的 `temperature`。例如，

* 在搭建的个人知识库助手项目中，一般将 `temperature` 设置为 0，
  从而保证助手对知识库内容的稳定使用，规避错误内容、模型幻觉；
* 在产品智能客服、科研论文写作等场景中，同样更需要稳定性而不是创造性；
* 但在个性化 AI、创意营销文案生成等场景中，就更需要创意性，
  从而更倾向于将 `temperature` 设置为较高的值。

## System Prompt

System Prompt 是随着 ChatGPT API 开放并逐步得到大量使用的一个新兴概念，
事实上，它并不在大模型本身训练中得到体现，而是大模型服务方为提升用户体验所设置的一种策略。

具体来说，在使用 ChatGPT API 时，你可以设置两种 Prompt：

* 一种是 System Prompt，该种 Prompt 内容会在整个会话过程中持久地影响模型的回复，
  且相比于普通 Prompt 具有更高的重要性；
* 另一种是 User Prompt，这更偏向于我们平时提到的 Prompt，即需要模型做出回复的输入。

一般设置 System Prompt 来对模型进行一些**初始化设定**，例如，
可以在 System Prompt 中给模型设定希望它具备的人设，如一个个人知识库助手等。
System Prompt 一般在一个会话中仅有一个。在通过 System Prompt 设定好模型的**人设**或是**初始设置**后，
可以通过 User Prompt 给出模型需要遵循的指令。例如，当我们需要一个幽默风趣的个人知识库助手，
并向这个助手提问我今天有什么事时，可以构造如下的 Prompt：

```
{
    "system prompt": "你是一个幽默风趣的个人知识库助手，可以根据给定的知识库内容回答用户的提问，注意，你的回答风格应是幽默风趣的",
    "user prompt": "我今天有什么事务？"
}
```

通过如上 Prompt 的构造，我们可以让模型以幽默风趣的风格回答用户提出的问题。

# LLM API

主要介绍四种大语言模型：ChatGPT、文心一言、讯飞星火、智谱 GLM 的 API 申请指引和 Python 版本的原生 API 调用方法，
可以按照实际情况选择一种自己可以申请的 API 进行使用即可。

* ChatGPT：推荐可科学上网的读者使用；
* 文心一言：当前无赠送新用户 tokens 的活动，推荐已有文心 tokens 额度用户和付费用户使用；
* 讯飞星火：新用户赠送 tokens，推荐免费用户使用；
* 智谱 GLM：新用户赠送 tokens，推荐免费用户使用。

如果你需要在 LangChain 中使用 LLM，可以参照 LangChain 中的调用方式。

## OpenAI ChatGPT

### API key 申请

* OpenAI API 调用服务是付费的，每一个开发者都需要首先获取并配置 OpenAI API key，
  才能在自己构建的应用中访问 ChatGPT。
* 在获取 OpenAI API key 之前需要在 OpenAI 官网注册一个账号。
  选择 `API`，然后点击左侧边栏的 `API keys`。
* 点击 `Create new secret key` 按钮创建 `OpenAI API key`。

### API key 配置

* 将创建好的 API key 以 `OPENAI_API_KEY="sk-..."` 的形式保存到 `.env` 文件中，
  并将 `.env` 文件保存在项目根目录下。

### 加载环境变量

读取 `.env` 文件，将密钥加载到环境变量

```python
import os
from dotenv import load_dotenv, find_dotenv

# 读取本地/项目的环境变量
# find_dotenv(): 寻找并定位 `.env` 文件的路基那个
# load_dotenv(): 读取 `.env` 文件，并将其中的环境变量加载到当前的运行环境中，如果设置的是环境变量，代码没有任何作用
_ = load_dotenv(find_dotenv())

# 如果需要通过代理端口访问，还需要做如下配置
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
```

### 调用 OpenAI API

调用 ChatGPT 需要使用 [`ChatCompletion` API](https://platform.openai.com/docs/api-reference/chat)，
该 API 提供了 ChatGPT 系列模型的调用，包括 ChatGPT-3.5、GPT-4 等。

`ChatCompletion` API 的调用方法如下：

```python
from openai import OpenAI

client = OpenAI(api_key = os.environ.get("OPENAI_API_KEY"))

completion = client.chat.completions.create(
    # 调用模型
    model = "gpt-3.5-turbo",
    # 对话列表
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "Hello!"
        },
    ]
)
```

调用该 API 会返回一个 `ChatCompletion` 对象，其中包括了回答文本、创建时间、id 等属性。
我们一般需要的是回答文本，也就是回答对象中的 `content` 信息。

```python
completion
```

```
ChatCompletion(id='chatcmpl-9FAKG4M6HXML257axa12PUuCXbJJz', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Hello! How can I assist you today?', role='assistant', function_call=None, tool_calls=None))], created=1713401640, model='gpt-3.5-turbo-0125', object='chat.completion', system_fingerprint='fp_c2295e73ad', usage=CompletionUsage(completion_tokens=9, prompt_tokens=19, total_tokens=28))
```

```python
print(completion.choices[0].message.content)
```

```
Hello! How can I assist you today?
```

`ChatCompletion` 常用的几个参数：

* `model`：即调用的模型，一般取值包括 `"gpt-3.5-turbo"`（ChatGPT-3.5）、
    `"gpt-3.5-turbo-16k-0613"`（ChatGPT-3.5 16K 版本）、`"gpt-4"`（ChatGPT-4）。
    注意，不同模型的成本是不一样的。
* `messages`：即 prompt。`ChatCompletion` 的 `messages` 需要传入一个列表，
    列表中包括多个不同角色的 prompt。可以选择的角色一般包括：
    - `system`：即前文中提到的 system prompt；
    - `user`：用户输入的 prompt；
    - `assistant`：助手，一般是模型历史回复，作为提供给模型的参考内容。
* `temperature`：温度。即前文中提到的 Temperature 系数。
* `max_tokens`：最大 token 数，即模型输出的最大 token 数。
    OpenAI 计算 token 数是合并计算 Prompt 和 Completion 的总 token 数，
    要求总 token 数不能超过模型上限（如默认模型 token 上限为 4096）。
    因此，如果输入的 prompt 较长，需要设置较大的 max_token 值，否则会报错超出限制长度。

另外，OpenAI 提供了充分的自定义空间，支持通过自定义 prompt 来提升模型回答效果，
下面是一个简答的封装 OpenAI 接口的函数，支持直接传入 prompt 并获得模型的输出。

```python
from openai import OpenAI

client = OpenAI(api_key = os.environ.get("OPENAI_API_KEY"))

def gen_gpt_messages(prompt):
    """
    构造 GPT 模型请求参数 messages

    Params:
        prompt: 对应的用户提示词
    """
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    return messages

def get_completion(prompt, model = "gpt-3.5-turbo", temperature = 0):
    """
    获取 GPT 模型调用结果

    Params:
        prompt: 对应的提示词
        model: 调用的模型，默认为 gpt-3.5-turbo，也可以按需选择 gpt-4 等其他模型
        temperature: 模型输出的温度系数，控制输出的随机程度，取值范围是 0~2。温度系数越低，输出内容越一致。
    """
    response = client.chat.completion.create(
        model = model,
        messages = gen_gpt_messages(prompt),
        temperature = temperature,
    )
    if len(response.choices) > 0:
        return response.choices[0].message.content
    
    return "generate answer error"

get_completion("你好")
```

```
'你好！有什么可以帮助你的吗？'
```

## 百度文心一言

百度同样提供了 **文心一言的 API 接口**，其在推出大模型的同时，
也推出了 **文心千帆企业级大语言模型服务平台**，包括了百度整套大语言模型开发工作链。
对于不具备大模型实际落地能力的中小企业或传统企业，考虑文心千帆是一个可行的选择。

### API key 申请

* 百度智能云千帆大模型平台提供了[多种语言的千帆 SDK](https://cloud.baidu.com/doc/WENXINWORKSHOP/s/wlmhm7vuo)，
  开发者可使用 SDK，快捷地开发功能，提升开发效率。在使用千帆 SDK 之前，需要先获取文心一言调用密钥，
  在代码中需要配置自己的密钥才能实现对模型的调用。
* 首先需要有一个经过实名认证的百度账号，每一个账户可以创建若干个应用，
  每个应用会对应一个 API_Key 和 Secret_Key。
* 进入文心千帆服务平台，点击上述 `应用接入` 按钮，创建一个调用文心大模型的应用。
  接着点击去 `创建` 按钮，进入应用创建界面。简单输入基本信息，选择默认配置，创建应用即可。
  创建完成后，可以在控制台看到创建的应用的 `API Key`、`Secret Key`。

> 需要注意的是，千帆目前只有 [**Prompt 模板**](https://cloud.baidu.com/doc/WENXINWORKSHOP/s/Alisj3ard)、[**Yi-34B-Chat**](https://cloud.baidu.com/doc/WENXINWORKSHOP/s/vlpteyv3c) 和 [**Fuyu-8B公有云在线调用体验服务**](https://cloud.baidu.com/doc/WENXINWORKSHOP/s/Qlq4l7uw6) 这三个服务是免费调用的，如果想体验其他的模型服务，
需要在计费管理处开通相应模型的付费服务才能体验。

### API key 配置

    - 将上面获取到的 `API Key`、`Secret Key` 填写至 `.env` 文件的 `QIANFAN_AK` 和 `QIANFAN_SK` 参数。
    - 如果使用的是安全认证的参数校验，需要在 **百度智能云控制台-用户账户-安全认证** 页，
      查看 `Access Key`、`Secret Key`，并将获取到的参数相应的填写到 `.env` 文件的 `QIANFAN_ACCESS_KEY`、`QIANFAN_SECRET_KEY`。

### 加载环境变量

读取 `.env` 文件，将密钥加载到环境变量

```python
import os
from dotenv import load_dotenv, find_dotenv

# 读取本地/项目的环境变量
# find_dotenv(): 寻找并定位 `.env` 文件的路基那个
# load_dotenv(): 读取 `.env` 文件，并将其中的环境变量加载到当前的运行环境中，如果设置的是环境变量，代码没有任何作用
_ = load_dotenv(find_dotenv())

# 如果需要通过代理端口访问，还需要做如下配置
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
```

### 调用文心千帆 API

百度文心同样支持在传入参数的 `messages` 字段中配置 `user`、`assistant` 两个成员角色的 prompt，
但与 OpenAI 的 prompt 格式不同的是，模型人设是通过另一个参数 `system` 字段传入的，而不是在 `messages` 字段中。

```python
import qianfan

def gen_wenxin_messages(prompt):
    """
    构造文心模型请求参数 message

    Params:
        prompt: 对应的用户提示词
    """
    messages = [{
        "role": "user",
        "content": prompt,
    }]

    return messages


def get_completion(prompt, model = "ERNIE-Bot", temperature = 0.01):
    """
    获取文心模型调用结果

    Params:
        prompt: 对应的提示词
        model: 调用的模型，默认为 ERNIE-Bot，也可以按需选择 Yi-34B-Chat 等其他模型
        temperature: 模型输出的温度系数，控制输出的随机程度，取值范围是 0~1.0，
                        且不能设置为 0。温度系数越低，输出内容越一致。
    """
    chat_comp = qianfan.ChatCompletion()
    message = gen_wenxin_messages(prompt)
    resp = chat_comp.do(
        messages = message,
        model = model,
        temperature = temperature,
        system = "你是一名个人助理"
    )

    return resp["result"]


get_completion(prompt = "你好，介绍以下你自己", model = "Yi-34B-Chat")
get_completion(prompt = "你好，介绍以下你自己")
```

API 介绍：
- `messages`: 即调用的 prompt。文心的 `messages` 配置与 ChatGPT 有一定区别，
    其不支持 `max_token` 参数，由模型自行控制最大 token 数，`messages` 中的 content 总长度、
    `functions` 和 `system` 字段总内容不能超过 20480 个字符，且不能超过 5120 tokens，
    否则模型就会自行对前文依次遗忘。文心的 `messages` 有以下几点要求：
    - 一个成员为单轮对话，多个成员为多轮对话
    - 最后一个 `message` 为当前对话，前面的 `message` 为历史对话
    - 成员数目必须为奇数，`message` 中的 `role` 必须依次是 `user`、`assistant`
    - 注：这里介绍的是 ERNIE-Bot 模型的字符数和 tokens 限制，而参数限制因模型而异，
        请在文心千帆官网查看对应模型的参数说明。
- `stream`: 是否使用流式传输。
- `temperature`: 温度系数，默认 0.8，文心的 `temperature` 参数要求范围为 `(0, 1.0]`，不能设置为 0。

### ERNIE SDK

#### API 申请

- 这里将使用 ERNIE SDK 中的 [`ERNIE Bot`](https://ernie-bot-agent.readthedocs.io/zh-cn/latest/sdk/) 来调用文心一言。
    `ERNIE Bot` 为开发者提供了便捷易用的接口，使其能够轻松调用文心大模型的强大功能，
    涵盖了文本创作、通用对话、语义向量以及 AI 作图等多个基础功能。
    ERNIE SDK 并不像千帆 SDK 那样支持各种大语言模型，
    而是只支持百度自家的文心大模型。目前 ERNIE Bot 支持的模型有：
    - ernie-3.5               文心大模型（ernie-3.5）
    - ernie-lite              文心大模型（ernie-lite）
    - ernie-4.0               文心大模型（ernie-4.0）
    - ernie-longtext          文心大模型（ernie-longtext）
    - ernie-speed             文心大模型（ernie-speed）
    - ernie-speed-128k        文心大模型（ernie-speed-128k）
    - ernie-tiny-8k           文心大模型（ernie-tiny-8k）
    - ernie-char-8k           文心大模型（ernie-char-8k）
    - ernie-text-embedding    文心百中语义模型
    - ernie-vilg-v2           文心一格模型
- 在使用 ERNIE SDK 之前，需要先获取 AI Studio 后端的认证鉴权（access token），
    在代码中需要配置自己的密钥才能实现对模型的调用
- 首先需要在 [AI Studio 星河社区](https://aistudio.baidu.com/index) 注册并登录账号（新用户会送 100 万 token 的免费额度，为期 3 个月）

#### API Key 配置

点击 **访问令牌** 获取账户的 `access token`，
复制 `access token` 并且以此形式 `EB_ACCESS_TOKEN="..."` 保存到 `.env` 文件中

读取 `.env` 文件，将密钥加载到环境变量

```python
import os
from dotenv import load_dotenv, find_dotenv

# 读取本地/项目的环境变量
# find_dotenv(): 寻找并定位 `.env` 文件的路基那个
# load_dotenv(): 读取 `.env` 文件，并将其中的环境变量加载到当前的运行环境中，如果设置的是环境变量，代码没有任何作用
_ = load_dotenv(find_dotenv())

# 如果需要通过代理端口访问，还需要做如下配置
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
``` 

#### 调用 Ernie Bot API

```python
import os
import erniebot

erniebot.api_type = "aistudio"
erniebot.access_token = os.environ.get("EB_ACCESS_TOKEN")

def gen_wenxin_messages(prompt):
    """
    构造文心模型请求参数 messages

    Params:
        prompt: 对应的用户提示词
    """
    messages = [{
        "role": "user",
        "content": prompt
    }]

    return messages


def get_completion(prompt, model = "ernie-3.5", temperature = 0.01):
    """
    获取文心模型调用结果

    Params:
        prompt: 对应的提示词
        model: 调用的模型
        temperature: 模型输出的温度系数，控制输出的随机程度，取值范围是 0~1.0，
        且不能设置为 0。温度系数越低，输出内容越一致。
    """
    chat_comp = erniebot.ChatCompletion()
    message = gen_wenxin_messages(prompt)
    resp = chat_comp.create(
        messages = message,
        model = model,
        temperature = temperature,
        system = "你是一名个人助理",
    )

    return resp["result"]
```

## 讯飞星火

讯飞星火认知大模型，由科大讯飞于 2023 年 5 月推出的中文大模型，也是国内大模型的代表产品之一。

### API key 申请

- 申请地址：https://xinghuo.xfyun.cn/sparkapi?ch=dwKeloHY
- 如果是没有领取过免费试用包的用户，可以领取到 100000 tokens 的试用量，
    完成个人身份认证后，还可以免费领取 2000000 tokens 的试用量。
    完成领取后，点击进入控制台并创建应用，创建完成后，
    就可以看到我们获取到的 `APPID`、`APISecret` 和 `APIKey` 了。
- 星火提供了两种调用模型的方式：
    - 一种是 SDK 方式调用，上手难度小，推荐初学者使用；
    - 另一种是 WebSocket 方式调用，对企业友好，但对初学者、新手开发者来说调用难度较大。

### API key 配置

将上面获取的 `APPID`、`APISecret` 和 `APIKey` 写入 `.env`

### 加载环境变量

读取 `.env` 文件，将密钥加载到环境变量

```python
import os
from dotenv import load_dotenv, find_dotenv

# 读取本地/项目的环境变量
# find_dotenv(): 寻找并定位 `.env` 文件的路基那个
# load_dotenv(): 读取 `.env` 文件，并将其中的环境变量加载到当前的运行环境中，如果设置的是环境变量，代码没有任何作用
_ = load_dotenv(find_dotenv())

# 如果需要通过代理端口访问，还需要做如下配置
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
```

### 模型调用

使用 SDK，封装一个 `get_completion()` 函数

```python
from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler
from sparkai.core.messages import ChatMessage


def gen_spark_params(model):
    """
    构造星火模型请求参数
    """
    spark_url_tpl = "wss://spark-api.xf-yun.com/{}/chat"
    model_params_dict = {
        # v1.5 版本
        "v1.5": {
            "domain": "general", # 用于配置大模型版本
            "spark_url": spark_url_tpl.format("v1.1") # 云端环境的服务地址
        },
        # v2.0 版本
        "v2.0": {
            "domain": "generalv2", # 用于配置大模型版本
            "spark_url": spark_url_tpl.format("v2.1") # 云端环境的服务地址
        },
        # v3.0 版本
        "v3.0": {
            "domain": "generalv3", # 用于配置大模型版本
            "spark_url": spark_url_tpl.format("v3.1") # 云端环境的服务地址
        },
        # v3.5 版本
        "v3.5": {
            "domain": "generalv3.5", # 用于配置大模型版本
            "spark_url": spark_url_tpl.format("v3.5") # 云端环境的服务地址
        }
    }

    return model_params_dict[model]


def gen_spark_messages(prompt):
    """
    构造星火模型请求参数 messages

    Params:
        prompt: 对应的用户提示词
    """
    messages = [
        ChatMessage(role = "user", content = prompt)
    ]

    return messages


def get_completion(prompt, model = "v3.5", temperature = 0.1):
    """
    获取星火模型调用结果

    Params:
        prompt: 对应的提示词
        model: 调用的模型，默认为 v3.5，也可以按需选择 v3.0 等其他模型
        temperature: 模型输出的温度系数，控制输出的随机程度，
            取值范围是 0~1.0，且不能设置为 0。温度系数越低，输出内容越一致。
    """
    spark_llm = ChatSparkLLM(
        spark_api_url = gen_spark_params(model)["spark_url"],
        spark_app_id = os.environ["SPARK_APPID"],
        spark_api_key = os.environ["SPARK_API_KEY"],
        spark_api_secret = os.environ["SPARK_API_SECRET"],
        spark_llm_domain = gen_spark_params(model)["domain"],
        temperature = temperature,
        streaming = False,
    )
    messages = gen_spark_messages(prompt)
    handler = ChunkPrintHandler()
    # 当 streaming设置为 False的时候, callbacks 并不起作用
    resp = spark_llm.generate([messages], callbacks=[handler])

    return resp

get_completion("你好").generations[0][0].text
```

## 智谱 GLM

### API key 申请

- 首先进入到 智谱 AI 开放平台，点击开始使用或者开发工作台进行注册。
    新注册的用户可以免费领取有效期 1 个月的 100w token 的体验包，
    进行个人实名认证后，还可以额外领取 400w token 体验包。
    智谱 AI 提供了 GLM-4 和 GLM-3-Turbo 这两种不同模型的体验入口，
    可以点击立即体验按钮直接体验。
- 对于需要使用 API key 来搭建应用的话，需要点击右侧的查看 API key按钮，
    就会进入到个人的 API 管理列表中。在该界面，
    就可以看到获取到的 API 所对应的应用名字和 API key 了。
- 可以点击 添加新的 API key 并输入对应的名字即可生成新的 API key。

### API key 配置

- 智谱 AI 提供了 SDK 和原生 HTTP 来实现模型 API 的调用，建议使用 SDK 进行调用以获得更好的编程体验。
- 首先需要配置密钥信息，将前面获取到的 API key 设置到 `.env`  文件中的 `ZHIPUAI_API_KEY`  参数；
- 然后，运行以下代码的加载配置信息：

```python
import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
``` 

### 调用智谱 GLM API

智谱的调用传参和其他类似，也需要传入一个 `message` 列表，列表中包括 `role` 和 `prompt`。
我们封装如下的 `get_completion` 函数，供后续使用

```python
from zhipuai import ZhipuAI

client = ZhipuAI(api_key = os.environ["ZHIPUAI_API_KEY"])

def get_glm_params(prompt):
    """
    构造 GLM 模型请求参数 message

    Params:
        prompt: 对应的用户提示词
    """
    message = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    return message

def get_completion(prompt, model = "glm-4", temperature = 0.95):
    """
    
    """
```

# LLM 接入 LangChain

LangChain 为基于 LLM 开发自定义应用提供了高效的开发框架，便于开发者迅速地激发 LLM 的强大能力，
搭建 LLM 应用。LangChain 也同样支持多种大模型，内置了 OpenAI、LLAMA 等大模型的调用接口。
但是，LangChain 并没有内置所有大模型，它通过允许用户自定义 LLM 类型，来提供强大的可扩展性。

## 基于 LangChain 调用 ChatGPT

LangChain 提供了对于多数大模型的封装，
基于 LangChain 的接口可以便捷地调用 ChatGPT 并将其集合在以 LangChain 为基础框架搭建的个人应用中。

注：基于 LangChain 接口调用 ChatGPT 同样需要配置个人密钥。

### Model

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

### Prompt

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

### Output parser

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

### 完整的流程

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

## 使用 LangChain 调用文心一言

通过 LangChain 框架来调用百度文心大模型，以将文心模型接入到应用框架中。

### 自定义 LLM 接入 langchain



### 在 langchain 直接调用文心一言

## 使用 LangChain 调用讯飞星火



## 使用 LangChain 调用智谱 GLM



### 自定义 chatglm

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

### 自定义 chatglm 接入 LangChain

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

# 参考资料


