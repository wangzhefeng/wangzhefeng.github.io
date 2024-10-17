---
title: 如何读论文？
author: wangzf
date: '2023-05-21'
slug: how-to-read-papers
categories:
  - 论文阅读
tags:
  - paper
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

- [论文的基本结构](#论文的基本结构)
- [三遍读论文](#三遍读论文)
  - [第一遍](#第一遍)
  - [第二遍](#第二遍)
  - [第三遍](#第三遍)
- [参考](#参考)
</p></details><p></p>

# 论文的基本结构

一般来说，一篇论文的基本结构：

1. 标题(title)
2. 作者(author)
3. 摘要(abstract)
4. 导言(introduct)
5. 算法/方法(method)
6. 实验(exp)
7. 结论(conclusion)

绝大多数的论文是这样一个八股文的形式，如果要从头读到尾，当然是可以的。
但是这个世界上论文有很多，如果都是从头读到尾，时间上可能划不来。
而且适合你的文章可能就是那么一小部分，因此需要快速地找到适合你的论文，然后对它进行精读

# 三遍读论文

对于每一遍花费的时间会不一样：

* 第一遍花的时间最少，做海选
* 第二遍的时候对相关的论文做一步精选
* 第三遍，重点研读论文

## 第一遍

第一遍首先要关注的内容：

1. 标题
    - 标题说明论文跟你是不是相关
2. 摘要
    - 摘要就是简单地介绍一下论文在做什么
3. 结论
    - 结论通常跟摘要是一样的，但是通常是吧摘要里面可能提出的一两个问题，
      用一些实际的结论、实际的数字进行证明

读完这三个部分，就可以大概知道这篇论文是在讲什么东西，这时可以跳到实验和方法部分：

4. 实验
    - 跳到一些实验部分，看一下一些关键的图和表
5. 算法/方法
    - 瞄一眼方法里面的图和表在干什么

通常这一边粗读需要十几分钟，就可以知道这篇文章大概在讲什么、质量怎么样、结果怎么样、方法看上去怎么样，
最重要的是不是适合自己，这时可以决定我要不要再继续读下去，还是放在一边

* 如果不再继续读下去，可能是因为论文质量不高，或者跟自己没有太多关系，自己不感兴趣
* 如果要再继续读下去，那就进行第二遍

## 第二遍

第二遍需要对整个文章过一遍，知道每一部分在干什么事情。这时可以沿着从标题一直往下读到最后。
这时候不要太注意很多细节，一些公式，特别是证明，或者一些很细节的部分可以忽略掉。
但是主要是要搞清楚那些重要的图和表里面它每一个字在干什么事情。比如说：

* 方法里面整个流程图长什么样、结构图长什么样
* 实验里面每一张图的 X 轴是什么、Y 轴是什么，每一个点是什么意思
* 作者提出的方法和别的方法是怎么对比的，之间差距有多大。这个时候可能还是没有特别搞得懂它在干什么，
  这个不要紧，可以留到之后。但是可以在中间可以把那些相关的文献给圈出来

在这一遍可以大概对整个论文的各个部分都有一个大概的了解。这一遍读完，
就可以决定要不要继续往下精度

* 如果决定到此为止，知道了论文在解决什么问题，结果怎么样，大概用了什么方法。
  但是觉得文章写得太难了，读不懂，可以先去读它引用的之前的那些文章，门槛低一些，然后再回来读这篇文章。
  另外，如果不需要了解那么深，知道论文就行了，那也可以到此为止
* 如果决定继续往下走，那就进行第三遍

## 第三遍

第三遍可以认为是最后一遍，也是最详细的一遍。在这一遍，需要知道里面的每一句话在干什么、每一段在说什么。
可以在读这篇文章的时候，想象如果是我来实现这个事情，应该可以用什么东西实现这个东西，脑补一下
比如实验部分能不能做的更好，那些还未实现的部分，自己可以做什么事情

读完这一遍之后，就可以对整个论文的细节都比较了解了，然后关上文章，可以做到能回忆出很多细节的部分。
然后再基于它做研究，或者之后再提到这篇论文的时候，可以详细地复述一遍

# 参考

* [李沐 YT 视频：如何读论文？](https://www.youtube.com/watch?v=txjl_Q4jCyQ&list=PLFXJ6jwg0qW-7UM8iUTj3qKqdhbQULP5I)