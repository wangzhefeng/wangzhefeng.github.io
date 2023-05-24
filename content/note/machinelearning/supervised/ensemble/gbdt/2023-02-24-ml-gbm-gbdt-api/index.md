---
title: GBDT API
author: 王哲峰
date: '2023-02-24'
slug: ml-gbm-gbdt-api
categories: 
    - machinelearning
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

- [GBDT 参数类型](#gbdt-参数类型)
- [GBDT 调参策略](#gbdt-调参策略)
- [参考](#参考)
</p></details><p></p>

# GBDT 参数类型

* 决策树参数
    - `min_samples_split`
        - 要分裂的树节点需要的最小样本数量, 若低于某个阈值, 则在此节点不分裂
        - 用于控制过拟合, 过高会阻止模型学习, 并导致欠拟合
        - 需要使用CV进行调参
    - `min_samples_leaf`
        - 叶子节点中所需的最小样本数, 若低于某个阈值, 则此节点的父节点将不分裂, 此节点的父节点作为叶子结点
        - 用于控制过拟合, 同 `min_samples_split` 
        - 一般选择一个较小的值用来解决不平衡类型样本问题
    - `min_weight_fraction_leaf`
        - 类似于 `min_sample_leaf` 
        - 一般不进行设置, 上面的两个参数设置就可以了
    - `max_depth`
        - 一棵树的最大深度
        - 用于控制过拟合, 过大会导致模型比较复杂, 容易出现过拟合
        - 需要使用 CV 进行调参
    - `max_leaf_nodes`
        - 一棵树的最大叶子节点数量
        - 一般不进行设置, 设置 `max_depth` 就可以了
    - `max_features`
        - 在树的某个节点进行分裂时的考虑的最大的特征个数
        - 一般进行随机选择, 较高的值越容易出现过拟合, 但也取决于具体的情况
        - 一般取特征个数的平方根(跟随机森林的选择一样)
* Boosting参数
    - `learning_rate`
        - 学习率
    - `n_estimators`
        - 树的个数
    - `subsample`
        - 构建每棵数时选择的样本数
* 其他参数
    - `loss`: 损失函数
    - `init`
    - `random_state`
    - `verbose`
    - `warm_start`
    - `presort`

# GBDT 调参策略

* 一般参数调节策略: 
    - 选择一个相对来说较高的 learning rate, 先选择默认值 0.1(0.05-0.2)
    - 选择一个对于这个 learning rate 最优的树的数量(合适的数量为: 40-70)
        - 若选出的树的数量较小, 可以减小 learning rate 重新跑 GridSearchCV
        - 若选出的树的数量较大, 可以增大初始 learning rate 重新跑 GridSearchCV
    - 调节基于树的参数
    - 降低 learning rate, 增加学习器的个数得到更稳健的模型
* 对于 learning rate 的调节, 对其他树参数设置一些默认的值
    - `min_samples_split = 500`
        - 0.5-1% of total samples
        - 不平衡数据选择一个较小值
    - `min_samples_leaf = 50`
        - 凭感觉选择, 考虑不平衡数据, 选择一个较小值
    - `max_depth = 8`
        - 基于数据的行数和列数选择, 5-8
    - `mat_features = 'sqrt'`
    - `subsample = 0.8`
* 调节树参数
    - 调节 `max_depth` , `min_samples_split`
    - 调节 `min_samples_leaf`
    - 调节 `max_features`



# 参考


