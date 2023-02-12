---
title: 特征工程
subtitle: Feature Engine
list_pages: true
# order_by: title
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

- [特征工程](#特征工程)
- [结构化数据](#结构化数据)
- [非结构化数据](#非结构化数据)
- [文档](#文档)
</p></details><p></p>

## 特征工程

在机器学习中, 所有数据最终都会转化为数值型特征, 所有特征工程都会归结为某种数值型特征工程技术。
特征工程, 顾名思义, 是对原始数据进行一系列工程处理, 将其提炼为特征, 作为输入供算法和模型使用

> * Garbage in, garbage out.
> * 对于一个机器学习问题, 数据和特征往往决定了结果的上限, 
>   而模型、算法的选择及优化则是在逐步接近这个上限.
> * 没有最好的模型, 只有最合适的模型.
> * 一个模型所能提供的信息一般来源于两个方面, 一是训练数据中蕴含的信息；
>   二是在模型的形成过程中(包括构造、学习、推理等), 人们提供的先验信息.

## 结构化数据

* **数值型特征**
    - 特征合理性检查
        - 量级
        - 正负
    - 特征尺度
        - 尺度: 
            - 最大值, 最小值
            - 是否横跨多个数量级
    - 特征分布
        - 对数变换
        - Box-Cox变换
    - 特征组合
        - 交互特征
        - 多项式特征
    - 特征选择
        - PCA
* **类别型特征**
    - 分类任务目标变量
    - 类别特征
* **时间序列数据**
    - 时间序列插值
    - 时间序列降采样
    - 时间序列聚合计算
    - 时间序列平滑
* **样本采样**
    - 欠采样
    - 过采样
    - 过采样和欠采样结合

## 非结构化数据

* **文本数据**
    - 扁平化
    - 过滤
    - 分块
* **图像数据**
* **音频数据**
* **视屏数据**

## 文档

