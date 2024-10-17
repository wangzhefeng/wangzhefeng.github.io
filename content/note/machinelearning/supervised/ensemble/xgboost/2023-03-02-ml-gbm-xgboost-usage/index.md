---
title: XGBoost 使用
author: wangzf
date: '2023-03-02'
slug: ml-gbm-xgboost-usage
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

- [参数](#参数)
    - [通用参数](#通用参数)
    - [Tree Booster 参数](#tree-booster-参数)
    - [Linear Booster 参数](#linear-booster-参数)
    - [DART Booster 参数](#dart-booster-参数)
    - [学习任务参数](#学习任务参数)
    - [计算参数](#计算参数)
        - [外存计算](#外存计算)
        - [GPU 计算](#gpu-计算)
- [参数调节](#参数调节)
    - [参数调优的一般策略](#参数调优的一般策略)
    - [调参指导](#调参指导)
- [Python API](#python-api)
    - [数据接口](#数据接口)
        - [数据格式](#数据格式)
        - [DMatrix](#dmatrix)
    - [模型接口](#模型接口)
    - [绘图 API](#绘图-api)
- [参考](#参考)
</p></details><p></p>

# 参数

## 通用参数

> 控制整个模型的通用性能

* `booster`：基本学习器类型，默认 `gbtree`
    - `gbtree`：基于树的模型
    - `gblinear`：线性模型。`gblinear` 使用带 l1，l2 正则化的线性回归模型作为基学习器。
      因为 boost 算法是一个线性叠加的过程，而线性回归模型也是一个线性叠加的过程。
      因此叠加的最终结果就是一个整体的线性模型，xgboost 最后会获得这个线性模型的系数
    - `dart`：DART booster
* `num_pbuffer`：指定了 prediction buffer 的大小。
    - 通常设定为训练样本的数量
    - 该参数由 xgboost 自动设定，无需用户指定
    - 该 buffer 用于保存上一轮 boostring step 的预测结果
* `num_feature`：样本的特征数量。
    - 通常设定为特征的最大维数。
    - 该参数由 xgboost 自动设定，无需用户指定
* `nthread`：用于运行的并行线程数量，如果未设定该参数，则默认为最大可用线程数；
* `silent`：如果为 0(默认值)，则表示打印运行时的信息；如果为 1，则表示 silent mode，不打印这些信息
* `verbosity`：打印消息的详细程度
    - `0`：静默
    - `1`：警告
    - `2`：信息
    - `3`：调试

## Tree Booster 参数

> 控制每步迭代中每个基学习器(树模型)

需要调参：

* `eta`(`learning_rate`)
    - 学习率，在更新中收缩每步迭代中基本学习器的权重，使模型更加稳健，防止过度拟合
    - 默认 `0.3`
    - 范围：`$[0, 1]$`
    - 典型值一般设置为：`$[0.01, 0.2]$` 或 `$[0.05, 0.3]$`
* `gamma`(`min_split_loss`)
    - 也称为最小划分损失(`min_split_loss`)树模型节点当分裂结果使得损失函数减少时，才会进行分裂。
      分裂节点时，损失函数减小值只有大于等于 `gamma` 时节点才分裂。
      `gamma` 值越大，算法越保守，越不容易过拟合，但性能就不一定能保证，需要平衡。
    - 默认 `0`
    - 范围：`$[0，\infty)$`
    - 典型值一般设置为：`$[0.1, 0.2]$`
* `min_child_weight`
    - 子节点中所有样本的权重和的最小值，如果新分裂的节点的样本权重和小于 `min_child_weight` 则停止分裂，
      可以用来减少过拟合，但是也不能太高，太高会导致欠拟合。
    - 默认 `1`
    - 范围：`$[0，\infty)$`
    - 典型值一般设置为：`1`
* `n_estimators`
    - 树模型基学习器的数量
* `max_depth`
    - 一棵树的最大深度。增加此值将使模型更复杂，并且更可能过度拟合
    - 默认 6
    - 范围：`$[0，\infty)$`
    - 典型值一般设置为：`$[3, 10]$`
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

重要：

* `lambda`(reg_lambda)：
    - L2 正则化权重项。增加此值将使模型更加保守。归一化为训练示例数
    - 默认 `0`
* `alpha`(reg_alpha)：
    - 权重的 L1 正则化项。增加此值将使模型更加保守。归一化为训练示例数
    - 默认 `0`
* `lambda_bias`：
    - L2 正则化系数(基于 `bias` 的正则化)，没有基于 `bias` 的 L1 正则化，因为它不重要
    - 默认为 `0`

不重要：

* `updater`：默认 `shotgun`
    - `shotgun`：基于 shotgun 算法的平行坐标下降算法。使用 "hogwild" 并行性，因此每次运行都产生不确定的解决方案
    - `coord_descent`：普通坐标下降算法。同样是多线程的，但仍会产生确定性的解决方案
* `feature_selector`：特征选择和排序方法，默认 `cyclic`
    - `cyclic`：通过每次循环一个特征来实现的
    - `shuffle`：类似于 `cyclic`，但是在每次更新之前都有随机的特征变换
    - `random`：一个随机(有放回)特征选择器
    - `greedy`：选择梯度最大的特征(贪婪选择)
    - `thrifty`：近似贪婪特征选择(近似于 `greedy`)
* `top_k`：要选择的最重要特征数(在 `greedy` 和 `thrifty` 内)

## DART Booster 参数

* `sample_type`： 它指定了丢弃时的策略：
    - `'uniform'`： 随机丢弃子树（默认值）
    - `'weighted'`： 根据权重的比例来丢弃子树
* `normaliz_type`：它指定了归一化策略：
    - `'tree'`： 新的子树将被缩放为 `$\frac{1}{K+v}$`；被丢弃的子树被缩放为 `$\frac{v}{K+v}$`。
      其中 `$v$` 为学习率，`$K$` 为被丢弃的子树的数量
    - `'forest'`：新的子树将被缩放为 `$\frac{1}{1+v}$`；被丢弃的子树被缩放为 `$\frac{v}{1+v}$`。
      其中 `$v$` 为学习率
* `rate_drop`：dropout rate，指定了当前要丢弃的子树占当前所有子树的比例。范围为 `[0.0,1.0]`， 默认为 `0.0`。
* `one_drop`：如果该参数为 `true`，则在 dropout 期间，至少有一个子树总是被丢弃。默认为 `0`。
* `skip_drop`：它指定了不执行 dropout 的概率，其范围是 `[0.0,1.0]`， 默认为 `0.0`。
  如果跳过了 dropout，则新的子树直接加入到模型中（和 xgboost 相同的方式）

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

## 计算参数

### 外存计算

### GPU 计算




# 参数调节

## 参数调优的一般策略

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

## 调参指导

1. 当出现过拟合时，有两类参数可以缓解：
    - 第一类参数：用于直接控制模型的复杂度。包括 `max_depth`、`min_child_weight`、`gamma` 等参数
    - 第二类参数：用于增加随机性，从而使得模型在训练时对于噪音不敏感。包括 `subsample`、`colsample_bytree`
    - 也可以直接减少步长 `eta`，但是此时需要增加 `num_round` 参数。
2. 当遇到数据不平衡时（如广告点击率预测任务），有两种方式提高模型的预测性能：
    - 如果关心的是预测的 AUC：
        - 可以通过 `scale_pos_weight` 参数来平衡正负样本的权重
        - 使用 AUC 来评估
    - 如果关心的是预测的正确率：
        - 不能重新平衡正负样本
        - 设置 `max_delta_step` 为一个有限的值（如 `1`），从而有助于收敛

# Python API

## 数据接口

### 数据格式

1. XGBoost 的数据存储在 `DMatrix` 中；
2. XGBoost 支持直接从下列格式的文件中加载数据：
    - `libsvm` 文本格式文件。其格式为：
    
    ```
    [label] [index1]:[value1] [index2]:[value2] ...
    [label] [index1]:[value1] [index2]:[value2] ...
    ...
    ```

    - `xgboost binary buffer` 文件

```python
import xgboost as xgb

dtrain = xgb.DMatrix("train.svm.txt")  # libsvm 格式
dtest = xgb.DMatrix("test.svm.buffer")  # xgboost binary buffer 文件
```


3. XGBoost 也支持从二维的 Numpy array 中加载数据

```python
data = np.random.rand(5, 10)
label = np.random.randint(2, size = 5)
dtrain = xgb.DMatrix(data, label = label)
```

4. 也可以从 `scipy.sparse.array` 中加载数据

```python
csr = scipy.sparse.csr_matrix((dat, (row, col)))
dtrain = xgb.DMatrix(csr)
```

### DMatrix




## 模型接口

## 绘图 API






# 参考

* [XGBoost 使用](https://www.huaxiaozhuan.com/%E5%B7%A5%E5%85%B7/xgboost/chapters/xgboost_usage.html)
