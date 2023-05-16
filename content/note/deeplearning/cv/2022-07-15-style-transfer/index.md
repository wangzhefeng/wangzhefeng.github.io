---
title: CNN-风格迁移
subtitle: style-transfer
author: 王哲峰
date: '2022-07-15'
slug: style-transfer
categories:
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
</style>

<details><summary>目录</summary><p>

- [参考](#参考)
</p></details><p></p>

深度神经网络的训练依赖于大量高质量的打标数据, 但实际研究工作中很难有这样好又多的数据供大家尝试, 
而迁移学习正是为了解决这种没有数据的尴尬而产生的一种方法论



# 参考

* [A Neural Algorithm of Artistic Style](https://www.jianshu.com/p/9f03b61fdeac)
* [neural style GitHub](https://github.com/jcjohnson/neural-style)
