---
title: NLP-问答系统
author: 王哲峰
date: '2022-04-05'
slug: nlp-qa-system
categories:
  - nlp
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
</style>

<details><summary>目录</summary><p>

- [NLP 问答系统简介](#nlp-问答系统简介)
- [基于关键词匹配的 NLP 问答系统](#基于关键词匹配的-nlp-问答系统)
- [基于相似度匹配的 NLP 问答系统](#基于相似度匹配的-nlp-问答系统)
</p></details><p></p>

# NLP 问答系统简介

# 基于关键词匹配的 NLP 问答系统

![img](images/QA_1.png)

- 基于 FEMA 表抽取实体、关系
- 基于 Neo4j 图数据库存储
- 基于 java SpringBoot 框架做后端接口
- 基于 HanLP 进行实体识别
- 根据用户输入的问题, 进行设备实体、失效模式实体识别, 名词为设备实体, 动词为失效模式实体
- 通过 Cypher 语句在 Neo4j 中查询设备实体匹配数据库中的设备、组件等实体, 
  失效模式实体匹配数据库中的失效模式, 
- 匹配的方式是模糊查询。返回整个节点路径, 
  包括:`设备 -> 一级组件 -> 二级组件 -> 三级组件 -> 失效模式 -> 根本原因 -> 措施`

# 基于相似度匹配的 NLP 问答系统

![img](images/QA_2.png)

- 基于问答对形式的数据
- 基于向量数据库 milvus 和关系型数据库 PostgreSQL
    - milvus 存储问题向量
    - PostgreSQL 存储问题答案
- 基于 Python `fastapi` 做后端接口
- 基于 bert-as-service 进行文本的 embedding 向量化
- 用户输入问题, 首先将问题向量化, 然后使用 milvus 提供的 search 函数, 
  查询 milvus 数据库中的 top_k 个与该向量最为相似的答案, 返回 ID, 
  然后根据 ID 到 PostgreSQL 查询问题答案

