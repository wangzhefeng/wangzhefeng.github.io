---
title: Python shutile
author: 王哲峰
date: '2023-01-09'
slug: shutile
categories:
  - Python
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

- [shutile 简介](#shutile-简介)
- [shutile 使用](#shutile-使用)
  - [目录和文件操作](#目录和文件操作)
  - [归档操作](#归档操作)
- [参考](#参考)
</p></details><p></p>

# shutile 简介

`shutile` 模块提供了一系列对文件和文件集合的高阶操作, 特别是提供了一些文件拷贝和删除的函数. 
对于单个文件的操作, 使用 `os` 模块. 

- 常用 API:
    - copyfileobj
    - copyfile
    - copymode
    - copystat
    - copy
    - copy2
    - ignore_patterns
    - copytree
    - rmtree
    - move

# shutile 使用



## 目录和文件操作



## 归档操作


# 参考

* https://docs.python.org/zh-cn/3/library/shutil.html

