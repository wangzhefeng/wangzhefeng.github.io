---
title: LLM 架构--RAG 应用
subtitle: 向量知识库-检索问答链-知识库助手
author: wangzf
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

- [搭建向量知识库](#搭建向量知识库)
    - [词向量](#词向量)
        - [词向量简介](#词向量简介)
        - [通用文本向量](#通用文本向量)
        - [RAG 中词向量的优势](#rag-中词向量的优势)
        - [RAG 构建词向量的方法](#rag-构建词向量的方法)
    - [数据预处理](#数据预处理)
        - [数据读取](#数据读取)
            - [PDF 文档](#pdf-文档)
            - [Markdown 文档](#markdown-文档)
        - [数据清洗](#数据清洗)
            - [PDF 文档](#pdf-文档-1)
            - [Markdown 文档](#markdown-文档-1)
        - [文档分割](#文档分割)
            - [文档分割简介](#文档分割简介)
            - [文档分割 API](#文档分割-api)
            - [文档分割示例](#文档分割示例)
    - [构建向量数据库](#构建向量数据库)
        - [向量数据库](#向量数据库)
        - [构建向量数据库](#构建向量数据库-1)
        - [完整代码](#完整代码)
    - [向量检索](#向量检索)
        - [相似度检索](#相似度检索)
        - [MMR 检索](#mmr-检索)
- [构建检索问答链](#构建检索问答链)
    - [加载数据库向量](#加载数据库向量)
    - [创建一个 LLM](#创建一个-llm)
    - [构建检索问答链](#构建检索问答链-1)
    - [检索问答链效果测试](#检索问答链效果测试)
        - [基于召回结果和 query 结合起来构建 prompt 效果](#基于召回结果和-query-结合起来构建-prompt-效果)
        - [大模型自己回答的效果](#大模型自己回答的效果)
    - [添加历史对话的记忆功能](#添加历史对话的记忆功能)
        - [记忆](#记忆)
        - [对话检索链](#对话检索链)
- [部署知识库助手](#部署知识库助手)
    - [构建应用程序](#构建应用程序)
    - [本地运行应用](#本地运行应用)
    - [部署应用程序](#部署应用程序)
- [参考](#参考)
</p></details><p></p>

## 搭建向量知识库

### 词向量

#### 词向量简介

在机器学习和自然语言处理(NLP)中，**词向量(Word Embeddings)是一种将非结构化数据，
如单词、句子或者整个文档，转化为实数向量的技术。** 这些实数向量可以被计算机更好地理解和处理。

**词向量(Embedding)背后的主要想法是，相似或相关的对象在嵌入空间中的距离应该很近。**

> 举个例子，可以使用词向量来表示文本数据。在词向量中，每个单词被转换为一个向量，
> 这个向量捕获了这个单词的语义信息。例如，`"king"` 和 `"queen"` 这两个单词在向量空间中的位置将会非常接近，
> 因为它们的含义相似。而 `"apple"` 和 `"orange"` 也会很接近，因为它们都是水果。
> 而 `"king"` 和 `"apple"` 这两个单词在向量空间中的距离就会比较远，因为它们的含义不同。

#### 通用文本向量

词向量实际上是将单词转化为固定的静态的向量，虽然可以在一定程度上捕捉并表达文本中的语义信息，
但忽略了单词在不同语境中的意思会受到影响这一现实。

因此在 RAG 应用中使用的向量技术一般为 **通用文本向量(Universal Text Embedding)**，
该技术可以对一定范围内任意长度的文本进行向量化，
与词向量不同的是向量化的单位不再是单词而是 **输入的文本**，输出的向量会捕捉更多的语义信息。

#### RAG 中词向量的优势

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

#### RAG 构建词向量的方法

在搭建 RAG 系统时，可以通过使用 Embedding 模型来构建词向量，可以选择：

* 使用各个公司的 Embedding API；
* 在本地使用嵌入模型将数据构建为词向量。

Embedding API 调用介绍在[这里](https://wangzhefeng.com/note/2024/09/23/llm-embedding/)。

### 数据预处理

为构建本地知识库，需要对以多种类型存储的本地文档进行处理，
读取**本地文档**并通过 **Embedding 方法**将**本地文档的内容**转化为**词向量**来构建**向量数据库**。

#### 数据读取

##### PDF 文档

使用 LangChain 的 `PyMuPDFLoader` 来读取知识库的 PDF 文件。
`PyMuPDFLoader` 是 PDF 解析器中速度最快的一种，
结果包含 PDF 及页面的详细元数据，并且每页返回一个文档。

```python
# -*- coding: utf-8 -*-

# ***************************************************
# * File        : pdf_loader.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-09-23
# * Version     : 1.0.092319
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = [
    "load_pdf"
]

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import warnings

from langchain_community.document_loaders import PyMuPDFLoader

warnings.filterwarnings("ignore")
# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def load_pdf(doc_path: str):
    """
    加载 PDF 文档
    """
    # 创建一个 PyMuPDFLoader Class 实例，输入为待加载的 PDF 文档路径
    loader = PyMuPDFLoader(doc_path)
    # 调用 PyMuPDFLoader Class 的函数 load 对 PDF 文件进行加载
    pdf_pages = loader.load()
    # 打印信息
    print(f"载入后的变量类型为：{type(pdf_pages)}, 该 PDF 一共包含 {len(pdf_pages)} 页。")
    
    return pdf_pages


# 测试代码 main 函数
def main():
    if sys.platform != "win32":
        doc_path = "/Users/wangzf/llm_proj/app/qa_chain/database/pumkin_book/pumpkin_book.pdf"
    else:
        doc_path = "D:/projects/llms_proj/llm_proj/app/qa_chain/database/pumkin_book/pumpkin_book.pdf" 
    pdf_pages = load_pdf(doc_path = doc_path) 
    # 第一页
    pdf_page = pdf_pages[1]
    print(
        f"每一个元素的类型：{type(pdf_page)}", 
        f"该文档的描述性数据：{pdf_page.metadata}", 
        f"查看该文档的内容：\n{pdf_page.page_content}",
        sep = "\n------\n"
    )

if __name__ == "__main__":
    main()
```

```
载入后的变量类型为：<class 'list'>, 该 PDF 一共包含 196 页。
每一个元素的类型：<class 'langchain_core.documents.base.Document'>
------
该文档的描述性数据：{'source': 'D:/projects/llms_proj/llm_proj/app/qa_chain/data_base/knowledge_db/pumkin_book/pumpkin_book.pdf', 'file_path': 'D:/projects/llms_proj/llm_proj/app/qa_chain/data_base/knowledge_db/pumkin_book/pumpkin_book.pdf', 'page': 1, 'total_pages': 196, 'format': 'PDF 1.5', 'title': '', 'author': '', 'subject': '', 'keywords': '', 'creator': 'LaTeX with hyperref', 'producer': 'xdvipdfmx (20200315)', 'creationDate': "D:20230303170709-00'00'", 'modDate': '', 'trapped': ''}
------
查看该文档的内容：
前言
“周志华老师的《机器学习》（西瓜书）是机器学习领域的经典入门教材之一，周老师为了使尽可能多的读
者通过西瓜书对机器学习有所了解, 所以在书中对部分公式的推导细节没有详述，但是这对那些想深究公式推
导细节的读者来说可能“不太友好”，本书旨在对西瓜书里比较难理解的公式加以解析，以及对部分公式补充
具体的推导细节。”
读到这里，大家可能会疑问为啥前面这段话加了引号，因为这只是我们最初的遐想，后来我们了解到，周
老师之所以省去这些推导细节的真实原因是，他本尊认为“理工科数学基础扎实点的大二下学生应该对西瓜书
中的推导细节无困难吧，要点在书里都有了，略去的细节应能脑补或做练习”。所以...... 本南瓜书只能算是我
等数学渣渣在自学的时候记下来的笔记，希望能够帮助大家都成为一名合格的“理工科数学基础扎实点的大二
下学生”。
使用说明
• 南瓜书的所有内容都是以西瓜书的内容为前置知识进行表述的，所以南瓜书的最佳使用方法是以西瓜书
为主线，遇到自己推导不出来或者看不懂的公式时再来查阅南瓜书；
• 对于初学机器学习的小白，西瓜书第1 章和第2 章的公式强烈不建议深究，简单过一下即可，等你学得
有点飘的时候再回来啃都来得及；
• 每个公式的解析和推导我们都力(zhi) 争(neng) 以本科数学基础的视角进行讲解，所以超纲的数学知识
我们通常都会以附录和参考文献的形式给出，感兴趣的同学可以继续沿着我们给的资料进行深入学习；
• 若南瓜书里没有你想要查阅的公式，或者你发现南瓜书哪个地方有错误，请毫不犹豫地去我们GitHub 的
Issues（地址：https://github.com/datawhalechina/pumpkin-book/issues）进行反馈，在对应版块
提交你希望补充的公式编号或者勘误信息，我们通常会在24 小时以内给您回复，超过24 小时未回复的
话可以微信联系我们（微信号：at-Sm1les）；
配套视频教程：https://www.bilibili.com/video/BV1Mh411e7VU
在线阅读地址：https://datawhalechina.github.io/pumpkin-book（仅供第1 版）
最新版PDF 获取地址：https://github.com/datawhalechina/pumpkin-book/releases
编委会
主编：Sm1les、archwalker、jbb0523
编委：juxiao、Majingmin、MrBigFan、shanry、Ye980226
封面设计：构思-Sm1les、创作-林王茂盛
致谢
特别感谢awyd234、feijuan、Ggmatch、Heitao5200、huaqing89、LongJH、LilRachel、LeoLRH、Nono17、
spareribs、sunchaothu、StevenLzq 在最早期的时候对南瓜书所做的贡献。
扫描下方二维码，然后回复关键词“南瓜书”，即可加入“南瓜书读者交流群”
版权声明
本作品采用知识共享署名-非商业性使用-相同方式共享4.0 国际许可协议进行许可。****
```

##### Markdown 文档

可以按照读取 PDF 文档几乎一致的方式读取 Markdown 文档。

```python
# -*- coding: utf-8 -*-

# ***************************************************
# * File        : markdown_loader.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-09-23
# * Version     : 1.0.092319
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = [
    "load_markdown"
]

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from langchain_community.document_loaders import UnstructuredMarkdownLoader

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def load_markdown(doc_path: str):
    """
    加载 Markdown 文档
    """
    # 创建一个 UnstructuredMarkdownLoader Class 实例，输入为待加载的 Markdown 文档路径
    loader = UnstructuredMarkdownLoader(doc_path)
    # 调用 UnstructuredMarkdownLoader Class 的函数 load 对 Markdown 文件进行加载
    md_pages = loader.load()
    print(f"载入后的变量类型为：{type(md_pages)}, 该 Markdown 一共包含 {len(md_pages)} 页。")
    
    return md_pages


# 测试代码 main 函数
def main():
    if sys.platform != "win32":
        doc_path = "/Users/wangzf/llm_proj/app/qa_chain/database/prompt_engineering/1. 简介 Introduction.md"
    else:
        doc_path = "D:/projects/llms_proj/llm_proj/app/qa_chain/database/prompt_engineering/1. 简介 Introduction.md"
    md_pages = load_markdown(doc_path = doc_path)
    # 第一页
    md_page = md_pages[0]
    print(
        f"每一个元素的类型：{type(md_page)}", 
        f"该文档的描述性数据：{md_page.metadata}", 
        f"查看该文档的内容：\n{md_page.page_content[0:][:200]}", 
        sep = "\n------\n"
    )

if __name__ == "__main__":
    main()
```

```
载入后的变量类型为：<class 'list'>, 该 Markdown 一共包含 1 页。
每一个元素的类型：<class 'langchain_core.documents.base.Document'>
------
该文档的描述性数据：{'source': 'D:/projects/llms_proj/llm_proj/app/qa_chain/data_base/knowledge_db/prompt_engineering/1. 简介 Introduction.md'}
------
查看该文档的内容：
第一章 简介

欢迎来到面向开发者的提示工程部分，本部分内容基于吴恩达老师的《Prompt Engineering for Developer》课程进行编写。《Prompt Engineering for Developer》课程是由吴恩达老师与 OpenAI 技术团队成员 Isa Fulford 老师合作授课，Isa 老师曾开发过受欢迎的 ChatGPT 检索插件，并且在教授 LLM （Larg
```

#### 数据清洗

##### PDF 文档

一般期望知识库的数据尽量是有序的、优质的、精简的，因此要删除低质量的、甚至影响理解文本数据。
可以看到上下文中读取的 PDF 文件不仅将一句话按照原文的分行添加了换行符 `\n`，
也在原本两个符号中插入了 `\n`，可以使用正则表达式匹配并删除掉 `\n`。

```python
def process_pdf(pdf_page):
    """
    PDF 文档处理
    """
    pattern = re.compile(
        r"[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]", 
        re.DOTALL
    )
    # 上下文中读取的 PDF 文件不仅将一句话按照原文的分行添加了换行符 \n，
    # 也在原本两个符号中插入了 \n，可以使用正则表达式匹配并删除掉 \n。
    pdf_page.page_content = re.sub(
        pattern,
        lambda match: match.group(0).replace("\n", ""),
        pdf_page.page_content,
    )
    # 数据中还有不少的 • 和空格，简单实用的 replace 方法即可。
    pdf_page.page_content = pdf_page.page_content.replace("•", "")
    pdf_page.page_content = pdf_page.page_content.replace(" ", "")

    return pdf_page
```

##### Markdown 文档

上下文中读取的 Markdown 文件每一段中间隔了一个换行符，同样可以使用 `replace` 方法去除。

```python
def process_markdown(md_page):
    """
    Markdown 文档数据预处理
    """
    # 上下文中读取的 Markdown 文件每一段中间隔了一个换行符，
    # 同样可以使用 replace 方法去除。
    md_page.page_content = md_page.page_content.replace("\n\n", "\n")

    return md_page
```

#### 文档分割

##### 文档分割简介

由于单个文档的长度往往会超过模型支持的上下文，导致检索得到的知识太长超出模型的处理能力。
因此，在构建向量知识库的过程中，往往需要对文档进行分割，
**将单个文档按长度或者按固定的规则分割成若干个 chunk，然后将每个 chunk 转化为词向量，
存储到向量数据库中**。在检索时，会以 chunk 作为检索的元单位，
也就是每一次检索到 `k` 个 chunk 作为模型可以参考来回答用户问题的知识，
这个 `k` 是可以自由设定的。

##### 文档分割 API

Langchain 中的文本分割器都根据 `chunk_size`(块大小)和 `chunk_overlap`(块与块之间的重叠大小)进行分割。
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

* `create_documents()`: Create documents from a list of texts；
* `split_documents()`: Split documents。

参数：

* `chunk_size` 指每个块包含的字符或 Token(如单词、句子等)的数量；
* `chunk_overlap` 指两个块之间共享的字符数量，用于保持上下文的连贯性，避免分割丢失上下文信息。

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

##### 文档分割示例

如何对文档进行分割，其实是数据处理中最核心的一步，其往往决定了检索系统的下限。
但是，如何选择分割方式，往往具有很强的业务相关性——针对不同的业务、不同的源数据，
往往需要设定个性化的文档分割方式。因此，这里仅简单根据 `chunk_size` 对文档进行分割。

```python
# -*- coding: utf-8 -*-

# ***************************************************
# * File        : doc_spliter.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-09-23
# * Version     : 1.0.092319
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# 导入文本分割器
from langchain.text_splitter import RecursiveCharacterTextSplitter

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def split_doc(pages, chunk_size: int = 500, overlap_size: int = 50):
    """
    * RecursiveCharacterTextSplitter 递归字符文本分割。
        将按不同的字符递归地分割(按照这个优先级 ["\n\n", "\n", " ", ""])，
        这样就能尽量把所有和语义相关的内容尽可能长时间地保留在同一位置
    * RecursiveCharacterTextSplitter 需要关注的是四个参数：
        - separators: 分隔符字符串数组
        - chunk_size: 每个文档的字符数量限制
        - chunk_overlap: 两份文档重叠区域的长度
        - length_function: 长度计算函数

    Args:
        pages (_type_): _description_
        chunk_size (int, optional): _description_. Defaults to 500.
        overlap_size (int, optional): _description_. Defaults to 50.

    Returns:
        _type_: _description_
    """
    # 知识库中单段文本长度
    CHUNK_SIZE = chunk_size
    # 知识库中相邻文本重合长度
    OVERLAP_SIZE = overlap_size
    # 使用递归字符文本分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap = OVERLAP_SIZE,
    )
    text_splitter.split_text(pages.page_content[0:1000])
    split_docs = text_splitter.split_documents(pages)
    print(f"切分后的文件数量：{len(split_docs)}")
    print(f"切分后的字符数（可以用来大致评估 token 数）：
          {sum([len(doc.page_content) for doc in split_docs])}")
    
    return split_docs


# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
```

### 构建向量数据库

#### 向量数据库

* 向量数据库介绍在[这里](https://wangzhefeng.com/note/2024/09/23/llm-vector-database/)。

#### 构建向量数据库

Langchain 集成了超过 30 个不同的向量存储库。选择 Chroma 是因为它轻量级且数据存储在内存中，
这使得它非常容易启动和开始使用。

LangChain 可以直接使用 OpenAI 和百度千帆的 Embedding，
同时，也可以针对其不支持的 Embedding API 进行自定义。

```python
import os
from dotenv import load_dotenv, find_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 使用 OpenAI Embedding
# from langchain.embeddings.openai import OpenAIEmbeddings
# 使用百度千帆 Embedding
# from langchain.embeddings.baidu_qianfan_endpoint import QianfanEmbeddingsEndpoint
# 使用我们自己封装的智谱 Embedding，需要将封装代码下载到本地使用
from embedding_api.zhipuai_embedding import ZhipuAIEmbeddings

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

# 遍历文件路径并把实例化的 loader 存放在 loaders 里
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
from langchain_community.vectorstores import Chroma

vectordb = Chroma.from_documents(
    documents = split_docs,
    embedding = embedding,
    # 允许将 persist_directory 目录保存到磁盘上 
    persist_directory = persist_directory,
)
```

在此之后，要确保通过运行 `vectordb.persist` 来持久化向量数据库，以便以后使用。

```python
vectordb.persist()
print(f"向量库中存储的数量：{vectordb._collection.count()}")
```

#### 完整代码

```python
# -*- coding: utf-8 -*-

# ***************************************************
# * File        : build_vectordb.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-10-22
# * Version     : 0.1.102200
# * Description : 搭建向量知识库
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import re
from dotenv import load_dotenv, find_dotenv

# Vector database
from langchain_community.document_loaders import (
    PyMuPDFLoader, 
    UnstructuredMarkdownLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# OpenAI Embedding
# from langchain.embeddings.openai import OpenAIEmbeddings
# 百度千帆 Embedding
# from langchain.embeddings.baidu_qianfan_endpoint import QianfanEmbeddingsEndpoint
# 智谱 Embedding
from embedding_api.zhipuai_embedding import ZhipuAIEmbeddings

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
# 读取本地/项目的环境变量
_ = load_dotenv(find_dotenv())
# 如果需要通过代理端口访问，你需要如下配置
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ["HTTP_PROXY"] = 'http://127.0.0.1:7890'


# ------------------------------
# 文档读取、预处理、分割
# ------------------------------
# 获取 folder_path 下所有文件路径，储存在 file_paths 里
file_paths = []
folder_path = "D:/projects/llms_proj/llm_proj/app/qa_chain/database/knowledge_lib"
for root, dirs, files in os.walk(folder_path):
    for file in files:
        file_path = os.path.join(root, file)
        file_paths.append(file_path)
print(file_paths[:3])

# 遍历文件路径并把实例化的 loader 存放在 loaders 里
loaders = []
for file_path in file_paths:
    file_type = file_path.split('.')[-1]
    if file_type == 'pdf':
        loaders.append(PyMuPDFLoader(file_path))
    elif file_type == 'md':
        loaders.append(UnstructuredMarkdownLoader(file_path))

# 下载文件并存储到 text
texts = []
for loader in loaders: 
    texts.extend(loader.load())
# 查看数据
# text = texts[1]
# print(
#     f"每一个元素的类型：{type(text)}.", 
#     f"该文档的描述性数据：{text.metadata}", 
#     f"查看该文档的内容:\n{text.page_content[0:]}", 
#     sep="\n------\n"
# )


# ------------------------------
# 向量知识库
# ------------------------------
# 切分文档
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
split_docs = text_splitter.split_documents(texts)

# 定义 Embeddings
# embedding = OpenAIEmbeddings() 
# embedding = QianfanEmbeddingsEndpoint()
embedding = ZhipuAIEmbeddings()

# 定义持久化路径
persist_directory = "D:/projects/llms_proj/llm_proj/app/qa_chain/database/vector_db/chroma"

# 构建向量知识库
vectordb = Chroma.from_documents(
    documents = split_docs,
    embedding = embedding,
    # 允许将 persist_directory 目录保存到磁盘上 
    persist_directory = persist_directory,
)
# 向量知识库持久化
vectordb.persist()
print(f"向量库中存储的数量：{vectordb._collection.count()}")


# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
```

### 向量检索

#### 相似度检索

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

for i, sim_doc in enumerate(sim_docs):
    print(f"检索到的第{i}个内容：\n{sim_doc.page_content[:200]}", end = "\n-------------\n")
```

#### MMR 检索

如果只考虑检索出内容的相关性会导致内容过于单一，可能丢失重要信息。
最大边际相关性(MMR, Maximum Marginal Relevance)可以帮助在保持相关性的同时，
增加内容的丰富度。

核心思想是在已经选择了一个相关性高得文档之后，再选择一个与已选文档相关性较低但是信息丰富的文档。
这样可以在保持相关性的同时，增加内容的多样性，避免过于单一的结果。

```python
question = "什么是大语言模型"

mmr_docs = vector_db.max_marginal_relevance_search(question, k = 3)
print(f"检索到的内容数：{len(mmr_docs)}")

for i, sim_doc in enumerate(mmr_docs):
    print(f"MMR 检索到的第 {i} 个内容：\n{sim_doc.page_content[:200]}", end = "\n-----------\n")
```

## 构建检索问答链

在上面介绍了如何根据自己的本地知识文档，搭建了一个向量知识库。
这里将使用搭建好的向量数据库，对查询问题进行召回，
并将召回结果和查询结合起来构建 prompt，输入到大模型中进行问答。

### 加载数据库向量

首先，加载已经构建的向量数据库。注意，此时需要使用和构建时相同的 Embedding。

```python
import os
from dotenv import find_dotenv, load_dotenv
from langchain_community.vectorstores import Chroma
from embedding_api.zhipuai_embedding import ZhipuAIEmbeddings

# 加载环境变量中的 API_KEY
_ = load_dotenv(find_dotenv())
zhipuai_api_key = os.environ["ZHIPUAI_API_KEY"]


# 定义 Embeddings
embedding = ZhipuAIEmbeddings()
# 向量数据库持久化路径
persist_directory = "../data_base/vector_db/chroma"
# 加载数据库
vectordb = Chroma(
    persist_directory = persist_directory,
    embedding_function = embedding,
)
print(f"向量库中存储的数量：{vectordb._collection.count()}")
```

> 测试加载的向量数据库，使用一个问题 query 进行向量检索，在向量数据库中根据相似性进行检索，
> 返回前 `$k$` 个最相似的文档：
> 
> ```python
> question = "什么是 prompt engineering?"
> docs = vectordb.similarity_search(question, k = 3)
> print(f"检索到的内容数：{len(docs)}")
> 
> for i, doc in enumerate(docs):
>      print(f"检索到的第 {i} 个内容：\n{doc.page_content}", end = "\n-------")
> ```

### 创建一个 LLM

```python
import os

from langchain_openai import ChatOpenAI

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

llm = ChatOpenAI(model_name = "gpt-3.5-turbo", temperature = 0)
llm.invoke("请你自我介绍一下自己！")
```

### 构建检索问答链

```python
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

template = """
使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
{context}
问题: {question}
"""
QA_CHAIN_PROMPT = PromptTemplate(
    input_variables = ["context", "question"],
    template = template,
)
qa_chain = RetrievalQA.from_chain_type(
    # 指定使用的 LLM
    llm,
    retriever = vectordb.as_retriever(),
    # 返回源文档，通过指定该参数，可以使用 RetrievalQAWithSourceChain() 方法，
    # 返回源文档的引用(坐标或者叫主键、索引)
    return_source_documents = True,
    # 自定义 prompt
    chain_type_kwargs = {"prompt": QA_CHAIN_PROMPT},
)
```

### 检索问答链效果测试

```python
question_1 = "什么是南瓜书？"
question_2 = "Prompt Engineering for Developer 是谁写的？"
```

#### 基于召回结果和 query 结合起来构建 prompt 效果

```python
result = qa_chain({"query": question_1})
print(f"大模型+知识库后回答 question_1 的结果：\n {result["result"]}")

result = qa_chain({"query": question_2})
print(f"大模型+知识库后回答 question_2 的结果：\n {result["result"]}")
```

#### 大模型自己回答的效果

```python
prompt_template = f"请回答一下问题：{question_1}"
llm.predict(prompt_template)

prompt_template = f"请回答一下问题：{question_2}"
llm.predict(prompt_template)
```

通过以上的两个问题，发现 LLM 对于一些近几年的知识以及非常识性的专业知识，回答的并不是很好。
而加上本地知识，就可以帮助 LLM 做出更好的回答。另外，也有助于缓解大模型的“幻觉”问题。

### 添加历史对话的记忆功能

现在已经实现了通过上传本地知识文档，然后将它们保存到向量知识库，
通过将查询问题与向量知识库的召回结果进行结合输入到 LLM 中，
就得到了一个相比于直接让 LLM 回答要好的多的结果。

在与语言模型交互时，可能已经注意到一个关键的问题：**它们并不记得你之前的交流内容**。
这在构建一些应用程序（如聊天机器人）的时候，带来了很大的挑战，使得对话似乎缺乏真正的连续性。

#### 记忆

> Memory

这里将介绍 LangChain 中的存储模块，即如何将先前的对话嵌入到语言模型中，使其具有连续对话的能力。
将使用 `ConversationBufferMemory` 这个 API，它保存聊天消息历史记录的列表，
这些历史记录将在回答问题时与问题一起传递给聊天机器人，从而将它们添加到上下文中。

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
     memory_key = "chat_history",  # 与 prompt 的输入变量保持一致
     return_messages = True,  # 将以消息列表的形式返回聊天记录，而不是单个字符串
)
```

Memory 的配置还包括保留指定对话轮数、保存指定 token 数量、保存历史对话的总结摘要等内容。

#### 对话检索链

> Conversational Retrieval Chain

对话索引链在检索 QA 链的基础上，增加了处理对话历史的能力。它的工作流程是：

1. 将之前的对话与新问题合并成一个完整的查询语句；
2. 在向量数据库中搜索该查询的相关文档；
3. 获取结果后，存储所有答案到对话记忆区；
4. 用户可以在 UI 中查看完整的对话流程。

这种链式方式将 **新问题** 放在之前对话的语境中进行检索，可以处理依赖历史信息的查询。
并保留所有信息在对话记忆中，方便追踪。

接下来让我们使用上一节中的向量数据库和 LLM，测试这个对话检索链的效果。

```python
from langchain.chains import ConversationalRetrievalChain

qa = ConversationalRetrievalChain.from_llm(
     llm,
     retriever = vectordb.as_retriever(),
     memory = memory,
)

question = "我可以学习到关于提示工程的知识吗？"
result = qa({"question": question})
print(result["answer"])

question = "为什么这门课需要教这方面的知识？"
result = qa({"question": question})
print(result["answer"])
```

可以看到，LLM 它准确地判断了这方面的知识，指代内容是强化学习的知识，也就是说我们成功地传递给了它历史信息。
这种持续学习和关联前后问题的能力，可大大增强了问答系统的连续性和智能水平。

## 部署知识库助手

当对知识库和 LLM 已经有了基本的理解，现在是将它们巧妙地融合并打造成一个富有视觉效果的界面的时候了。
这样的界面不仅对操作更加便捷，还能便于与他人分享。

### 构建应用程序

```python
# -*- coding: utf-8 -*-

# ***************************************************
# * File        : steamlit_app_demo.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-08-04
# * Version     : 0.1.080417
# * Description : description
# * Link        : https://github.com/datawhalechina/llm-universe/blob/0ce94e828ce2fb63d47741098188544433c5e878/notebook/C4%20%E6%9E%84%E5%BB%BA%20RAG%20%E5%BA%94%E7%94%A8/streamlit_app.py
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from dotenv import load_dotenv, find_dotenv

# Embedding
from embedding_api.zhipuai_embedding import ZhipuAIEmbeddings
# LLM
from langchain_openai import ChatOpenAI
# Prompt
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
# RAG
from langchain_community.vectorstores import Chroma
from langchain.chains import (
    RetrievalQA,
    ConversationalRetrievalChain
)
from langchain.memory import ConversationBufferMemory
# deploy
import streamlit as st

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
# 将父目录放入系统路径中
# sys.path.append("../knowledge_lib")
# 读取本地 .env 文件
_ = load_dotenv(find_dotenv())
# 如果需要通过代理端口访问，你需要如下配置
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ["HTTP_PROXY"] = 'http://127.0.0.1:7890'
# 载入 LLM API KEY
os.environ["OPENAI_API_BASE"] = "https://api.chatgptid.net/v1"
# zhipuai embedding api key
zhipuai_api_key = os.environ["ZHIPUAI_API_KEY"]
# openai llm api key
openai_api_key = os.environ["OPENAI_API_KEY"]


def generate_response(input_text, llm_api_key):
    """
    定义一个函数，使用用户密钥对 OpenAI API 进行
        - 身份验证
        - 发送提示
        - 获取 AI 生成的响应
    该函数接受用户的提示作为参数，并使用 st.info 来在蓝色框中显示 AI 生成的响应。
    """
    llm = ChatOpenAI(temperature = 0.7, openai_api_key = llm_api_key)
    output = llm.invoke(input_text)
    output_parser = StrOutputParser()
    output = output_parser.invoke(output)
    # TODO st.info(output)
    
    return output


def get_vectordb():
    """
    函数返回持久化后的向量知识库
    """
    # 定义 Embeddings
    embedding = ZhipuAIEmbeddings()
    # 向量数据库持久化路径
    persist_directory = "D:/projects/llms_proj/llm_proj/app/qa_chain/database/vector_db/chroma"
    # 加载数据库
    vectordb = Chroma(
        persist_directory = persist_directory,
        embedding_function = embedding,
    )
    # print(f"向量库中存储的数量：{vectordb._collection.count()}")

    return vectordb


def get_qa_chain(question: str, llm_api_key: str):
    """
    函数返回调用不带有历史记录的检索问答链后的结果
    """
    # 持久化后的向量知识库
    vectordb = get_vectordb()
    # Chat LLM
    llm = ChatOpenAI(
        model = "gpt-3.5-turbo",
        temperature = 0,
        opanai_api_key = llm_api_key,
    )
    # Prompt
    template = """"使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，
    不要试图编造答案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
    {context}
    问题: {question}
    """
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables = ["context", "question"],
        template = template,
    )
    # Retrieval QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever = vectordb.as_retriever(),
        # 返回源文档，通过指定该参数，
        # 可以使用 RetrievalQAWithSourceChain() 方法，
        # 返回源文档的引用(坐标或者叫主键、索引)
        return_source_documents = True,
        chain_type_kwargs = {"prompt": QA_CHAIN_PROMPT}
    )
    result = qa_chain({"query": question})

    return result["result"]


def get_chat_qa_chain(question: str, openai_api_key: str):
    """
    函数返回调用带有历史记录的检索问答链后的结果
    """
    # 持久化后的向量知识库
    vectordb = get_vectordb()
    # Chat LLM
    llm = ChatOpenAI(
        model_name = "gpt-3.5-turbo", 
        temperature = 0, 
        openai_api_key = openai_api_key
    )
    # memory history
    memory = ConversationBufferMemory(
        memory_key = "chat_history",  # 与 prompt 的输入变量保持一致
        return_messages = True,  # 将消息列表的形式返回聊天记录，而不是单个字符串
    ) 
    qa = ConversationBufferMemory.from_llm(
        llm, 
        retriever = vectordb.as_retriever(),
        memory = memory,
    )
    result = qa({"question": question})
    
    return result["answer"]


# Streamlit 应用程序界面
def main():
    # 创建应用程序的标题
    st.title("🦜🔗 向量知识库-检索问答链-知识库助手")
    
    # 添加一个文本输入框，供用户输入其 LLM API 密钥
    llm_api_key = st.sidebar.text_input("OpenAI API Key", type = "password")
    
    # ------------------------------
    # 添加一个选择按钮来选择不同的模型 
    # ------------------------------
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

    # ------------------------------
    # 用于跟踪对话历史 
    # ------------------------------
    # 通过使用 st.session_state 来存储对话历史，可以在用户与应用程序交互时保留整个对话的上下文, 用于跟踪对话历史
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # ------------------------------
    # 使用 st.form() 创建一个文本框 st.text_area() 供用户输入
    # ------------------------------
    # 当用户点击 Submit 时，generate_response 将使用用户的输入作为参数来调用该函数
    with st.form("my_form"):
        text = st.text_area(
            "Enter text:", 
            "What are the three key pieces of advice for learning how to code?"
        )
        submitted = st.form_submit_button("Submit")
        
        if not llm_api_key.startswith("sk-"):
            st.warning("Please enter your LLM API key!", icon = "")
        
        if submitted and llm_api_key.startswith("sk-"):
            generate_response(text, llm_api_key)
    
    # ------------------------------
    # 
    # ------------------------------
    messages = st.container(height = 300)
    if prompt := st.chat_input("Say something"):
        # 将用户输入添加到对话历史中
        st.session_state.messages.append({
            "role": "user",
            "text": prompt,
        })
        # 调用 respond 函数获取回答
        if selected_method == "None":
            answer = generate_response(prompt, llm_api_key)
        if selected_method == "qa_chain":
            answer = get_qa_chain(prompt, llm_api_key)
        elif selected_method == "chat_qa_chain":
            answer = get_chat_qa_chain(prompt, llm_api_key)
        
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

### 本地运行应用

```bash
$ streamlit run streamlit_app.py
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

**优化方向：**

* 界面中添加上传本地文档，建立向量数据库的功能；
* 添加多种 LLM 与 Embedding 方法选择的按钮；
* 添加修改参数的按钮；
* 更多...

## 参考

* [llm-universe](https://github.com/datawhalechina/llm-universe?tab=readme-ov-file)
