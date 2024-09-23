---
title: LLM RAG 应用--搭建知识库、检索问答链、部署知识库助手
author: 王哲峰
date: '2024-08-03'
slug: llm-rag-app-knowledge
categories:
  - llm
tags:
  - article
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

- [搭建知识库](#搭建知识库)
     - [词向量](#词向量)
          - [词向量简介](#词向量简介)
          - [通用文本向量](#通用文本向量)
          - [词向量的优势](#词向量的优势)
          - [构建词向量的方法](#构建词向量的方法)
     - [向量数据库](#向量数据库)
          - [向量数据库简介](#向量数据库简介)
- [调用 Embedding API](#调用-embedding-api)
     - [OpenAI API](#openai-api)
     - [文心千帆 API](#文心千帆-api)
     - [讯飞星火 API](#讯飞星火-api)
     - [智谱 API](#智谱-api)
- [数据处理](#数据处理)
     - [数据选取](#数据选取)
          - [PDF 文档](#pdf-文档)
          - [Markdown 文档](#markdown-文档)
     - [数据清洗](#数据清洗)
     - [文档分割](#文档分割)
          - [文档分割简介](#文档分割简介)
          - [文档分割 API](#文档分割-api)
          - [文档分割示例](#文档分割示例)
- [搭建并使用向量数据库](#搭建并使用向量数据库)
     - [配置](#配置)
     - [构建 Chroma 向量库](#构建-chroma-向量库)
     - [向量检索](#向量检索)
          - [相似度检索](#相似度检索)
          - [MMR 检索](#mmr-检索)
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
- [参考](#参考)
</p></details><p></p>

# 搭建知识库

## 词向量

### 词向量简介

在机器学习和自然语言处理(NLP)中，词向量(Word Embeddings)是一种将非结构化数据，
如单词、句子或者整个文档，转化为实数向量的技术。这些实数向量可以被计算机更好地理解和处理。

词向量(Embedding)背后的主要想法是，相似或相关的对象在嵌入空间中的距离应该很近。
举个例子，可以使用词向量来表示文本数据。在词向量中，每个单词被转换为一个向量，
这个向量捕获了这个单词的语义信息。例如，"king" 和 "queen" 这两个单词在向量空间中的位置将会非常接近，
因为它们的含义相似。而 "apple" 和 "orange" 也会很接近，
因为它们都是水果。而 "king" 和 "apple" 这两个单词在向量空间中的距离就会比较远，因为它们的含义不同。

### 通用文本向量

词向量实际上是将单词转化为固定的静态的向量，虽然可以在一定程度上捕捉并表达文本中的语义信息，
但忽略了单词在不同语境中的意思会受到影响这一现实。

因此在 RAG 应用中使用的向量技术一般为**通用文本向量(Universal Text Embedding)**，
该技术可以对一定范围内任意长度的文本进行向量化，与词向量不同的是向量化的单位不再是单词而是**输入的文本**，
输出的向量会捕捉更多的语义信息。

### 词向量的优势

在 RAG 里面词向量的优势主要有两点：

* 词向量比文字更适合检索
    - 当在数据库检索时，如果数据库存储的是文字，
      主要通过检索关键词（词法搜索）等方法找到相对匹配的数据，
      匹配的程度是取决于关键词的数量或者是否完全匹配查询句的；
    - 词向量中包含了原文本的语义信息，可以通过计算问题与数据库中数据的点积、
      余弦距离、欧几里得距离等指标，直接获取问题与数据在语义层面上的相似度。
* 词向量比其它媒介的综合信息能力更强
    - 当传统数据库存储文字、声音、图像、视频等多种媒介时，
      很难去将上述多种媒介构建起关联与跨模态的查询方法；
    - 词向量可以通过多种向量模型将多种数据映射成统一的向量形式。

### 构建词向量的方法

在搭建 RAG 系统时，可以通过使用 Embedding 模型来构建词向量，可以选择：

* 使用各个公司的 Embedding API；
* 在本地使用嵌入模型将数据构建为词向量。

## 向量数据库

### 向量数据库简介

向量数据库介绍在[这里]()。

# 调用 Embedding API

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

## 文心千帆 API

Embedding-V1 是基于百度文心大模型技术的文本表示模型，Access token 为调用接口的凭证，
使用 Embedding-V1 时应先凭 API Key、Secret Key 获取 Access token，
再通过 Access token 调用接口来 Embedding text。
同时千帆大模型平台还支持 `bge-large-zh` 等 Embedding model。

```python
import json
import requests

def wenxin_embedding(text: str):
    # 获取环境变量 wenxin_api_key, wenxin_secret_key
    api_key = os.environ["QIANFN_AK"]
    secret_key = os.environp["QIANFAN_SK"]
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


# text 应为 List(str)
text = "要生成 embedding 的输入文本，字符串形式。"
response = wenxin_embedding(text = text)

print(f"本次 embedding id 为：{response["id"]}")
print(f"本次 embedding 产生的时间戳为：{response["created"]}")
print(f"返回的 embedding 类型为：{response["object"]}")
print(f"embedding 长度为：{response["data"][0]["embedding"]}")
print(f"embedding (前 10) 为：{response["data"][0]["embedding"][:10]}")
```

## 讯飞星火 API

未开放

## 智谱 API

智谱有封装好的 SDK，直接调用即可。

```python
from zhipuai import ZhipuAI


def zhipu_embedding(text: str):
    api_key = os.environ["ZHIPUAI_API_KEY"]
    client = ZhipuAI(api_key = api_key)
    response = client.embeddings.create(
        model = "embedding-2",
        input = text,
    )
    
    return response

text = "要生成 embedding 的输入文本，字符串形式。"
response = zhipu_embedding(text = text)

print(f"response 类型为：{type(response)}")
print(f"embedding 类型为：{response.object}")
print(f"生成 embedding 的 model 为：{response.model}")
print(f"生成的 embedding 长度为：{len(response.data[0].embedding)}")
print(f"embedding(前 10)为: {response.data[0].embedding[:10]}")
```

# 数据处理

为构建本地知识库，需要对以多种类型存储的本地文档进行处理，
读取**本地文档**并通过 **Embedding 方法**将**本地文档的内容**转化为**词向量**来构建**向量数据库**。

## 数据选取

### PDF 文档

使用 LangChain 的 `PyMuPDFLoader` 来读取知识库的 PDF 文件。
`PyMuPDFLoader` 是 PDF 解析器中速度最快的一种，
结果包含 PDF 及页面的详细元数据，并且每页返回一个文档。

```python
from langchain.document_loaders.pdf import PyMuPDFLoader

# 创建一个 PyMuPDFLoader Class 实例，输入为待加载的 PDF 文档路径
loader = PyMuPDFLoader("data_base/knowledge_db/pumpkin_book/pumpkin_book.pdf")
# 调用 PyMuPDFLoader Class 的函数 load 对 PDF 文件进行加载
pdf_pages = loader.load()
```

文档加载后储存在 `pdf_pages` 变量中：

* `pdf_pages` 的变量类型为 `List`
* 打印 `pdf_pages` 的长度可以看到 PDF 一共包含多少页

```python
print(f"载入后的变量类型为：{type(pdf_pages)}, 该 PDF 一共包含 {len(pdf_pages)} 页。")
```

`pdf_pages` 中的每一个元素为一个文档，变量类型为 `langchain_core.documents.base.Document`，
文档变量类型包含两个属性：

* `meta_data` 为文档相关的描述性数据
* `page_content` 包含该文档的内容

```python
pdf_page = pdf_pages[1]
print(
    f"每一个元素的类型：{type(pdf_page)}", 
    f"该文档的描述性数据：{pdf_page.metadata}", 
    f"查看该文档的内容：\n{pdf_page.page_content}",
    sep = "\n------\n"
)
```

### Markdown 文档

可以按照读取 PDF 文档几乎一致的方式读取 Markdown 文档。

```python
from langchain.document_loaders.markdown import UnstructureMarkdownLoader

loader = UnstructureMarkdownLoader("data_base/knowledge_db/prompt_engineering/1.简介 Instroduction.md")
md_pages = loader.load()
```

读取的对象和 PDF 文档读取出来是完全一致的：

```python
print(f"载入后的变量类型为：{type(md_pages)}, 该 Markdown 一共包含 {len(md_pages)} 页。")
```

```python
md_page = md_pages[0]
print(
    f"每一个元素的类型：{type(md_page)}", 
    f"该文档的描述性数据：{md_page.metadata}", 
    f"查看该文档的内容：\n{md_page.page_content[0:][:200]}", 
    sep = "\n------\n"
)
```

## 数据清洗

期望知识库的数据尽量是有序的、优质的、精简的，因此要删除低质量的、甚至影响理解文本数据。
可以看到上下文中读取的 PDF 文件不仅将一句话按照原文的分行添加了换行符 `\n`，
也在原本两个符号中插入了 `\n`，可以使用正则表达式匹配并删除掉 `\n`。

```python
import re
pattern = re.compile(r"[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]", re.DOTALL)
pdf_page.page_content = re.sub(
    pattern, 
    lambda match: match.group(0).replace("\n", ""), 
    pdf_page.page_content
)
```

进一步分析数据，发现数据中还有不少的 `•` 和空格，简单实用的 `replace` 方法即可。

```python
pdf_page.page_content = pdf_page.page_content.replace("•", "")
pdf_page.page_content = pdf_page.page_content.replace(" ", "")
print(pdf_page.page_content)
```

上下文中读取的 Markdown 文件每一段中间隔了一个换行符，同样可以使用 `replace` 方法去除。

```python
md_page.page_content = md_page.page_content.replace("\n\n", "\n")
print(md_page.page_content)
```

## 文档分割

### 文档分割简介

由于单个文档的长度往往会超过模型支持的上下文，导致检索得到的知识太长超出模型的处理能力，
因此，在构建向量知识库的过程中，往往需要对文档进行分割，
**将单个文档按长度或者按固定的规则分割成若干个 chunk，然后将每个 chunk 转化为词向量，
存储到向量数据库中**。在检索时，会以 chunk 作为检索的元单位，
也就是每一次检索到 `k` 个 chunk 作为模型可以参考来回答用户问题的知识，
这个 `k` 是可以自由设定的。

### 文档分割 API

Langchain 中文本分割器都根据 `chunk_size`(块大小)和 `chunk_overlap`(块与块之间的重叠大小)进行分割。
`CharacterTextSpitter` API 示例如下：

```python
langchain.text_splitter.CharacterTextSpitter(
    separator: str = "\n\n",
    chunk_size = 4000,
    chunk_overlap = 200,
    length_function = <buildin function len>,
)
```

方法：

* `create_documents()`: Create documents from a list of texts.
* `split_documents()`: Split documents.

参数：

* `chunk_size` 指每个块包含的字符或 Token(如单词、句子等)的数量
* `chunk_overlap` 指两个块之间共享的字符数量，用于保持上下文的连贯性，避免分割丢失上下文信息

Langchain 提供多种文档分割方式，区别在怎么确定块与块之间的边界、
块由哪些字符/token 组成、以及如何测量块大小：

* `RecursiveCharacterTextSplitter()`: 按字符串分割文本，递归地尝试按不同的分隔符进行分割文本
* `CharacterTextSplitter()`: 按字符来分割文本
* `MarkdownHeaderTextSplitter()`: 基于指定的标题来分割 Markdown 文件
* `TokenTextSplitter()`: 按 token 来分割文本
* `SentenceTransformersTokenTextSplitter()`: 按 token 来分割文本
* `Language()`: 用于 CPP、Python、Ruby、Markdown 等
* `NLTKTextSplitter()`: 使用 `NLTK`（自然语言工具包）按句子分割文本
* `SpacyTextSplitter()`: 使用 `Spacy` 按句子的切割文本

### 文档分割示例

```python
''' 
* RecursiveCharacterTextSplitter 递归字符文本分割。
  将按不同的字符递归地分割(按照这个优先级 ["\n\n", "\n", " ", ""])，
  这样就能尽量把所有和语义相关的内容尽可能长时间地保留在同一位置
* RecursiveCharacterTextSplitter 需要关注的是4个参数：
    - separators: 分隔符字符串数组
    - chunk_size: 每个文档的字符数量限制
    - chunk_overlap: 两份文档重叠区域的长度
    - length_function: 长度计算函数
'''

# 导入文本分割器
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 知识库中单段文本长度
CHUNK_SIZE = 500
# 知识库中相邻文本重合长度
OVERLAP_SIZE = 50

# 使用递归字符文本分割器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = CHUNK_SIZE,
    chunk_overlap = OVERLAP_SIZE,
)
text_splitter.split_text(pdf_page.page_content[0:1000])

split_docs = text_splitter.split_documents(pdf_pages)
print(f"切分后的文件数量：{len(split_docs)}")
print(f"切分后的字符数（可以用来大致评估 token 数）：{sum([len(doc.page_content) for doc in split_docs])}")
```

如何对文档进行分割，其实是数据处理中最核心的一步，其往往决定了检索系统的下限。
但是，如何选择分割方式，往往具有很强的业务相关性——针对不同的业务、不同的源数据，
往往需要设定个性化的文档分割方式。因此，这里仅简单根据 `chunk_size` 对文档进行分割。

# 搭建并使用向量数据库

## 配置

```python
import os
from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders.pdf import PyMuPDFLoader
from langchain.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 读取本地/项目的环境变量
_ = load_dotenv(find_dotenv())

# 如果需要通过代理端口访问，你需要如下配置
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ["HTTP_PROXY"] = 'http://127.0.0.1:7890'


# 获取 folder_path 下所有文件路径，储存在 file_paths 里
file_paths = []
folder_path = '../../data_base/knowledge_db'
for root, dirs, files in os.walk(folder_path):
    for file in files:
        file_path = os.path.join(root, file)
        file_paths.append(file_path)
print(file_paths[:3])

# 遍历文件路径并把实例化的loader存放在loaders里
loaders = []
for file_path in file_paths:
    file_type = file_path.split('.')[-1]
    if file_type == 'pdf':
        loaders.append(PyMuPDFLoader(file_path))
    elif file_type == 'md':
        loaders.append(UnstructuredMarkdownLoader(file_path))

# 下载文件并存储到text
texts = []
for loader in loaders: 
    texts.extend(loader.load())

# 查看数据
text = texts[1]
print(
    f"每一个元素的类型：{type(text)}.", 
    f"该文档的描述性数据：{text.metadata}", 
    f"查看该文档的内容:\n{text.page_content[0:]}", 
    sep="\n------\n"
)

# 切分文档
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500, 
    chunk_overlap = 50
)
split_docs = text_splitter.split_documents(texts)
```

## 构建 Chroma 向量库

Langchain 集成了超过 30 个不同的向量存储库。选择 Chroma 是因为它轻量级且数据存储在内存中，
这使得它非常容易启动和开始使用。LangChain 可以直接使用 OpenAI 和百度千帆的 Embedding，
同时，也可以针对其不支持的 Embedding API 进行自定义。

```python
# 使用 OpenAI Embedding
# from langchain.embeddings.openai import OpenAIEmbeddings

# 使用百度千帆 Embedding
# from langchain.embeddings.baidu_qianfan_endpoint import QianfanEmbeddingsEndpoint

# 使用我们自己封装的智谱 Embedding，需要将封装代码下载到本地使用
from zhipuai_embedding import ZhipuAIEmbeddings

# 定义 Embeddings
# embedding = OpenAIEmbeddings() 
# embedding = QianfanEmbeddingsEndpoint()
embedding = ZhipuAIEmbeddings()

# 定义持久化路径
persist_directory = '../../data_base/vector_db/chroma'
```

删除旧的数据库文件（如果文件夹中有文件的话），windows 电脑请手动删除

```bash
$ !rm -rf '../../data_base/vector_db/chroma'
```

```python
from langchain.vectorstores.chroma import Chroma

vectordb = Chroma.from_documents(
    documents = split_docs,
    embedding = embedding,
    persist_directory = persist_directory  # 允许将 persist_directory 目录保存到磁盘上
)
```

在此之后，要确保通过运行 `vectordb.persist` 来持久化向量数据库，以便以后使用。

```python
vectordb.persist()
print(f"向量库中存储的数量：{vectordb._collection.count()}")
```

## 向量检索

### 相似度检索

Chroma 的相似度搜索使用的是余弦距离，即：

`$$\begin{align}
similarity
&=cos(A, B) \\
&=\frac{A \cdot B}{||A|| ||B||} \\
&=\frac{\sum_{i=1}^{n}a_{i}b_{i}}{\sqrt{\sum_{i=1}^{n}a_{i}^{2}}\sqrt{\sum_{i=1}^{n}b_{i}^{2}}}
\end{align}$$`

其中 `$a_{i}, b_{i}$` 分别是向量 `$A, B$` 的分量。

当需要数据库返回严谨的按余弦相似度排序的结果时可以使用 `similarity_search` 函数。

```python
question = "什么是大语言模型"
sim_docs = vectordb.similarity_search(question, k = 3)
print(f"检索到的内容数：{len(sim_docs)}")
for i, sim_docs in enumerate(sim_docs):
    print(f"检索到的第{i}个内容：\n{sim_doc.page_content[:200]}", end = "\n-------------\n")

```

### MMR 检索

如果只考虑检索出内容的相关性会导致内容过于单一，可能丢失重要信息。
最大边际相关性(MMR, Maximum Marginal Relevance)可以帮助在保持相关性的同时，
增加内容的丰富度。核心思想是在已经选择了一个相关性搞得文档之后，
再选择一个与已选文档相关性较低但是信息丰富的文档。
这样可以在保持相关性的同时，增加内容的多样性，避免过于单一的结果。

```python
mmr_docs = vector_db.max_marginal_relevance_search(question, k = 3)
for i, sim_doc in enumerate(mmr_docs):
    print(f"MMR 检索到的第 {i} 个内容：\n{sim_doc.page_content[:200]}", end = "\n-----------\n")
```

# 基于 LangChain 构建检索问答链

在[这里]()介绍了如何根据自己的本地知识文档，搭建一个向量知识库。
使用搭建好的向量数据库，对 query 查询问题进行召回，
并将召回结果和 query 结合起来构建 prompt，输入到大模型中进行问答。

## 加载数据库向量



## 创建一个 LLM


## 构建检索问答链



## 检索问答链效果测试


## 添加历史对话的记忆功能


# 基于 Streamlit 部署知识库助手

当对知识库和 LLM 已经有了基本的理解，现在是将它们巧妙地融合并打造成一个富有视觉效果的界面的时候了。
这样的界面不仅对操作更加便捷，还能便于与他人分享。

> Streamlit 是一种快速便捷的方法，可以直接在 Python 中通过友好的 Web 界面演示机器学习模型。
> 在构建了机器学习模型后，如果想构建一个 demo 给其他人看，也许是为了获得反馈并推动系统的改进，
> 或者只是因为觉得这个系统很酷，所以想演示一下：Streamlit 可以通过 Python 接口程序快速实现这一目标，
> 而无需编写任何前端、网页或 JavaScript 代码。

## 构建应用程序

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

## 添加检索回答

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

## 部署应用程序

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

# 参考

* [llm-universe](https://github.com/datawhalechina/llm-universe?tab=readme-ov-file)
