---
title: XGBoost API
author: 王哲峰
date: '2023-02-24'
slug: ml-gbm-xgboost-api
categories:
  - machinelearning
tags:
  - model
  - api
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

- [XGBoost 参数](#xgboost-参数)
  - [通用参数](#通用参数)
  - [Tree Booster 参数](#tree-booster-参数)
  - [Linear Booster 参数](#linear-booster-参数)
  - [学习任务参数](#学习任务参数)
  - [参数调节](#参数调节)
- [XGBoost API](#xgboost-api)
  - [核心数据结构](#核心数据结构)
  - [Learning API](#learning-api)
  - [Scikit-Learn API](#scikit-learn-api)
  - [数据可视化 API](#数据可视化-api)
</p></details><p></p>

# XGBoost 参数

## 通用参数

> 控制整个模型的通用性能

* `booster`：基本学习器类型，默认 `gbtree`
    - `gbtree`：基于树的模型
    - `gblinear`：线性模型
    - `dart`：TODO
* `nthread`：用于运行的并行线程数，默认为最大可用线程数
* `verbosity`：打印消息的详细程度
    - `0`：静默
    - `1`：警告
    - `2`：信息
    - `3`：调试

## Tree Booster 参数

> 控制每步迭代中每个基学习器(树模型)

需要调参：

* `eta`(`learning_rate`)
    - 学习率，在更新中收缩每步迭代中基本学习期的权重，使模型更加稳健，防止过度拟合
    - 默认 0.3
    - 范围：`$[0, 1]$`
    - 典型值一般设置为：`$[0.01, 0.2]$` 或 `$[0.05, 0.3]$`
* `n_estimators`
    - 树模型基学习器的数量
* `max_depth`
    - 一棵树的最大深度。增加此值将使模型更复杂，并且更可能过度拟合
    - 默认 6
    - 范围：`$[0，\infty)$`
    - 典型值一般设置为：`$[3, 10]$`
* `min_child_weight`
    - 子节点中所有样本的权重和的最小值，如果新分裂的节点的样本权重和小于 `min_child_weight` 则停止分裂，
      可以用来减少过拟合，但是也不能太高，太高会导致欠拟合
    - 默认 1
    - 范围：`$[0，\infty)$`
    - 典型值一般设置为：1
* `gamma`(`min_split_loss`)
    - 树模型节点当分裂结果使得损失函数减少时，才会进行分裂。
      分裂节点时，损失函数减小值只有大于等于 gamma 时节点才分裂。
      gamma 值越大，算法越保守，越不容易过拟合，但性能就不一定能保证，需要平衡
    - 默认 0
    - 范围：`$[0，\infty)$`
    - 典型值一般设置为：`$[0.1, 0.2]$`
* `subsample`
    - 构建每棵树对样本的采样率
    - 默认 1
    - 范围：`$(0, 1]$`
    - 典型值一般设置为：`$[0.5, 0.9]$`
* `colsample_bytree`
    - 列采样率，也就是特征采样率
    - 默认 1
    - 范围：`$(0，1]$`
    - 典型值一般设置为：`$[0.5, 0.9]$`
* `scale_pos_weight`
    - 控制正负权重的平衡，在类别高度不平衡的情况下，将参数设置大于 0，可以加快收敛。
      Kaggle 竞赛一般设置 `sum(negativ instances) / sum(positive instances)`
* `lambda`(reg_lambda)：
    - 叶节点权重得分的 L2 正则参数。L2 正则化权重项，增加此值将使模型更加保守
    - 默认 1
* `alpha`(reg_alpha)：
    - 叶节点权重得分的l1正则参数。权重的 L1 正则化项，增加此值将使模型更加保守。适用于高维数据中, 能使得算法加速
    - 默认 0

一般不需要调参：

* `max_leaf_nodes`
    - 树中叶节点的最大数量
* `max_delta_step`：
    - 允许每个叶子输出的最大增量步长。如果将该值设置为 0，则表示没有约束。如果将其设置为正值，
      则可以帮助使更新步骤更加保守。通常不需要此参数，但是当类极度不平衡时，
      它可能有助于逻辑回归。将其设置为 `$[1, 10]$` 的值可能有助于控制更新
    - 默认 0
    - 范围：`$[0，\infty)$`
* `sampling_method`
    - 用于对训练样本进行采样的方法
    - 默认 `uniform`
    - `uniform`：每个训练实例的选择概率均等。通常将 subsample>=0.5 设置为良好的效果
    - `gradient_based`：每个训练实例的选择概率与规则化的梯度绝对值成正比，
      具体来说就是，subsample 可以设置为低至 0.1，而不会损失模型精度
* `tree_method`
    - XGBoost 中使用的树构建算法 
    - 默认 `auto`
    - auto：使用启发式选择最快的方法
    - exact：对于小型数据集，精确的贪婪算法，枚举所有拆分的候选点
    - approx：对于较大的数据集，使用分位数和梯度直方图的近似贪婪算法
    - hist：更快的直方图优化的近似贪婪算法(LightGBM 也是使用直方图算法)
    - gpu_hist：GPU hist 算法的实现
- `num_parallel_tree`
    - 每次迭代期间构造的并行树的数量。此选项用于支持增强型随机森林
    - 默认 1
- `monotone_constraints`
    - 可变单调性的约束，在某些情况下，如果有非常强烈的先验信念认为真实的关系具有一定的质量，
      则可以使用约束条件来提高模型的预测性能
    - 例如 `params_constrained['monotone_constraints'] = "(1,-1)"`，
      `(1,-1)` 表示对第一个预测变量施加增加的约束，对第二个预测变量施加减小的约束

## Linear Booster 参数

- `lambda`(reg_lambda)：
    - L2 正则化权重项。增加此值将使模型更加保守。归一化为训练示例数
    - 默认 0
- `alpha`(reg_alpha)：
    - 权重的 L1 正则化项。增加此值将使模型更加保守。归一化为训练示例数
    - 默认 0
- `updater`：默认 `shotgun`
    - `shotgun`：基于 shotgun 算法的平行坐标下降算法。使用 "hogwild" 并行性，因此每次运行都产生不确定的解决方案
    - `coord_descent`：普通坐标下降算法。同样是多线程的，但仍会产生确定性的解决方案
- `feature_selector`：特征选择和排序方法，默认 `cyclic`
    - `cyclic`：通过每次循环一个特征来实现的
    - `shuffle`：类似于 `cyclic`，但是在每次更新之前都有随机的特征变换
    - `random`：一个随机(有放回)特征选择器
    - `greedy`：选择梯度最大的特征(贪婪选择)
    - `thrifty`：近似贪婪特征选择(近似于 `greedy`)
- `top_k`：要选择的最重要特征数(在 `greedy` 和 `thrifty` 内)

## 学习任务参数

> 控制模型优化的表现

* `objective`：默认 `reg:squarederror`，表示最小平方误差
    - 回归
        - `reg:squarederror`：最小平方误差
        - `reg:squaredlogerror`：对数平方损失
        - `reg:logistic`：逻辑回归
        - `reg:pseudohubererror`：使用伪 Huber 损失进行回归，这是绝对损失的两倍可微选择
        - `reg:tweedie`：使用对数链接进行 Tweedie 回归
    - 二分类
        - `binary:logistic`：二元分类的逻辑回归，输出分类概率
        - `binary:logitraw`：用于二进制分类的逻辑回归，逻辑转换之前的输出得分
        - `binary:hinge`：二进制分类的铰链损失。这使预测为0或1，而不是产生概率。(SVM就是铰链损失函数)
    - 生存分析
        - `survival:cox`：针对正确的生存时间数据进行 Cox 回归(负值被视为正确的生存时间)
        - `survival:aft`：用于检查生存时间数据的加速故障时间模型
        - `aft_loss_distribution`：survival:aft 和 aft-nloglik 度量标准使用的概率密度函数
    - 多分类
        - `multi:softmax`：使用 softmax 目标函数进行多类分类，需要设置 `num_class`(类数)，返回分类标签
        - `multi:softprob`：与 softmax 相同，但输出向量，可以进一步重整为矩阵。
          结果包含属于每个类别的每个数据点的预测概率
    - 排序
        - `rank:pairwise`：使用 LambdaMART 进行成对排名，从而使成对损失最小化
        - `rank:ndcg`：使用 LambdaMART 进行列表式排名，使标准化折让累积收益(NDCG)最大化
        - `rank:map`：使用 LambdaMART 进行列表平均排名，使平均平均精度(MAP)最大化
        - `reg:gamma`：使用对数链接进行伽马回归。输出是伽马分布的平均值
    - `count:poisson`：计数数据的泊松回归，泊松分布的输出平均值
    - 自定义损失函数和评价指标
* `eval_metric`：验证数据的评估指标，将根据目标分配默认指标(回归均方根，分类误差，排名的平均平均精度)，
  用户可以添加多个评估指标
    - 回归
        - `rmse`，均方根误差
        - `rmsle`：均方根对数误差
        - `mae`：平均绝对误差
        - `mphe`：平均伪 Huber 错误
    - 二分类
        - `logloss`：负对数似然
        - `error`：二(进制)分类错误率
    - 多分类
        - `merror`：多类分类错误率
        - `mlogloss`：多类 `logloss`
    - `auc`：曲线下面积
    - `aucpr`：PR 曲线下的面积
    - `ndcg`：归一化累计折扣
    - `map`：平均精度
* `seed`：随机数种子，默认 0

## 参数调节

参数调优的一般策略:

1. 首先, 选择一个相对较大的 `eta`/`learning_rate`, 比如: 0.1(一般范围在: `$[0.05, 0.3]$`)
    - 根据这个选定的 `learning_rate`，对树的数量 `n_estimators` 进行 CV 调优，
      选择一个最优的 `n_estimators`
2. 依次调节树参数 
    - `max_depth`
        - `$[3, 10]$`
    - `min_child_weight`
        - 类别不平衡数据选择一个较小值
    - `gamma`
        - `$[0.1, 0.2]$`
    - `subsample`
        - `$[0.5, 0.9]$`
    - `colsample_bytree`
        - `$[0.5, 0.9]$`
    - `scale_pos_weight`
        - 类别不平衡数据选择一个较小值
3. 调节正则化参数
    - `lambda`: L2
    - `alpha`: L1
4. 减小 `learning_rate`, 增加决策树数量 `n_estimators`，并且优化其他参数
5. 调节对模型结果影响最大的参数
   - `max_depth`
   - `min_child_weight`

# XGBoost API

## 核心数据结构

```python
xgb.DMatrix(
    data,
    label = None,
    missing = None,
    weight = None,
    silent = False,
    feature_name = None,
    feature_type = None,
    nthread = None
)
```

```python
xgb.Booster(
    params = None, 
    cache = (),
    model_file = None
)
```

## Learning API

```python
xgb.train(
    params, 
    dtrain,
    num_boost_round = 10,
    evals = (),
    obj = None,
    feval = None,
    maximize = False,
    early_stopping_rounds = None,
    evals_result = None,
    verbose_eval = True,
    xgb_model = None,
    callbacks = None,
    learning_rate = None
)
```

```python
xgb.cv(
    params, 
    dtrain,
    num_boost_round = 10,
    # ---------
    # cv params
    nfold = 3,
    stratified = False,
    folds = None,
    metrics = (),
    # ---------
    obj = None,
    feval = None,
    maximize = False,
    early_stopping_rounds = None,
    fpreproc = None,
    as_pandas = True,
    verbose_eval = None,
    show_stdv = True,
    seed = 0,
    callbacks = None,
    shuffle = True
)
```

## Scikit-Learn API

```python
xgb.XGBRegressor(
    max_depth = 3,
    learning_rate = 0.1, 
    n_estimators = 100,
    verbosity = 1,
    silent = None,
    objective = "reg:squarederror",
    booster = "gbtree",
    n_jobs = 1,
    nthread = None,
    gamma = 0,
    min_child_weight = 1,
    max_delta_step = 0,
    subsample = 1,
    colsample_bytree = 1,
    colsample_bylevel = 1,
    colsample_bynode = 1,
    reg_alpha = 0,
    reg_lambda = 1,
    scale_pos_weight = 1,
    base_score = 0.5,
    random_state = 0,
    seed = None,
    missing = None,
    importance_type = "gain",
    **kwargs
)

xgbr.fit(
    X, y,
    sample_weight = None,
    eval_set = None,
    eval_metric = None, 
    early_stopping_rounds = None,
    verbose = True,
    xgb_model = None,
    sample_weight_eval_set = None,
    callbacks = None
)
```

```python
xgb.XGBClassifier(
    max_depth = 3,
    learning_rate = 0.1,
    n_estimators = 100,
    verbosity = 1,
    silent = None,
    objective = "binary:logistic",
    booster = "gbtree",
    n_jobs = 1,
    nthread = None,
    gamma = 0,
    min_child_weight = 1,
    max_delta_step = 0,
    subsample = 1,
    colsample_bytree = 1,
    colsample_bylevel = 1,
    colsample_bynode = 1,
    reg_alpha = 0,
    reg_lambda = 1,
    scale_pos_weight = 1,
    base_score = 0.5,
    random_state = 0,
    seed = None, 
    missing = None,
    **kwargs
)
xgbc.fit(
    X, 
    y,
    sample_weight = None,
    eval_set = None,
    eval_metric = None,
    early_stopping_rounds = None,
    verbose = True,
    xgb_model = None,
    sample_weight_eval_set = None,
    callbacks = None
)
```

## 数据可视化 API

```python
xgb.plot_importance(
    booster,
    ax = None,
    height = 0.2, 
    xlim = None,
    ylim = None,
    title = "Feature importance",
    xlabel = "F score",
    ylabel = "Features",
    importance_type = "weight"
)
```

