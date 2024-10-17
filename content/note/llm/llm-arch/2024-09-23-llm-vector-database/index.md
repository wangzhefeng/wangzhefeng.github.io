---
title: LLM 架构--Vector Database
author: wangzf
date: '2024-09-23'
slug: llm-vector-database
categories:
  - llm
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
img {
    pointer-events: none;
}
</style>

<details><summary>目录</summary><p>

- [向量数据库简介](#向量数据库简介)
- [向量数据库原理及优势](#向量数据库原理及优势)
- [主流向量数据库](#主流向量数据库)
- [参考](#参考)
</p></details><p></p>

# 向量数据库简介

向量数据库是用于高效计算和管理大量 **向量数据** 的解决方案。
向量数据库是一种专门用于 **存储和检索向量数据(embedding)** 的数据库系统。
它与传统的基于关系模型的数据库不同，它主要关注的是 **向量数据的特性和相似性**。

在向量数据库中，数据被表示为向量形式，每个向量代表一个数据项。
这些向量可以是数字、文本、图像或其他类型的数据。
向量数据库使用高效的 **索引** 和 **查询** 算法来加速向量数据的存储和检索过程。

# 向量数据库原理及优势

向量数据库中的数据以向量作为基本单位，对向量进行存储、处理及检索。
向量数据库通过计算与目标向量的余弦距离、点积等获取与目标向量的相似度。
当处理大量甚至海量的向量数据时，向量数据库索引和查询算法的效率明显高于传统数据库。

# 主流向量数据库

* [Chroma](https://www.trychroma.com/)：是一个轻量级向量数据库，拥有丰富的功能和简单的 API，
  具有简单、易用、轻量的优点，但功能相对简单且不支持 GPU 加速，适合初学者使用。
* [Weaviate](https://weaviate.io/)：是一个开源向量数据库。
  除了支持**相似度搜索**和**最大边际相关性(MMR，Maximal Marginal Relevance)搜索**外，
  还可以支持**结合多种搜索算法（基于词法搜索、向量搜索）的混合搜索**，
  从而搜索提高结果的相关性和准确性。
* [Qdrant](https://qdrant.tech/)：Qdrant 使用 Rust 语言开发，
  有**极高的检索效率和 RPS(Requests Per Second)**，
  支持**本地运行**、**部署在本地服务器**及 **Qdrant 云**三种部署模式。
  且可以通过为页面内容和元数据制定不同的键来复用数据。

# 参考

* [向量及向量知识库](https://github.com/datawhalechina/llm-universe/blob/main/notebook/C3%20%E6%90%AD%E5%BB%BA%E7%9F%A5%E8%AF%86%E5%BA%93/1.%E5%90%91%E9%87%8F%E5%8F%8A%E5%90%91%E9%87%8F%E7%9F%A5%E8%AF%86%E5%BA%93%E4%BB%8B%E7%BB%8D.md)
* [LangChain Components Vectorstores](https://python.langchain.com/docs/integrations/vectorstores/)
