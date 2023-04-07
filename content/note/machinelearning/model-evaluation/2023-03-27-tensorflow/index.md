---
title: tensorflow
author: 王哲峰
date: '2023-03-27'
slug: tensorflow
categories:
  - deeplearning
tags:
  - model
---


# 评价指标

损失函数除了作为模型训练时的优化目标，也能够作为模型好坏的一种评价指标。
但是，通常还会从其他角度评估模型的好坏，这就是评价指标

通常损失函数都可以作为评价指标，如 MAE、MSE、CategoricalCrossentropy 等也是常用的评价指标。
但评价指标不一定可以作为损失函数，例如 AUC、Accuracy、Precision，因为评价指标不要求连续可导，
而损失函数通常要求连续可导

TensorFlow 编译模型时，可以通过列表形式指定多个评价指标

## TensorFlow 内置评价指标

### 回归指标

* `MeanSquaredError` 类
    - 函数形式 `mse`
    - 均方误差，MSE
* `RootMeanSquaredError` 类
    - 函数形式 `rmse`
    - 均方根误差，RMSE
* `MeanAbsoluteError` 类
    - 函数形式 `mae`
    - 平均绝对误差，MAE
* `MeanAbsolutePercentageError` 类
    - 函数形式 `mape`
    - 平均百分比误差，MAPE

### 二分类指标

* `Accuracy` 类
    - 函数形式 `acc`
    - 准确率，可以用字符串 `Accuracy` 表示
    - `$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$`
    - 要求 `y_true` 和 `y_pred` 都为类别序号编码
* `BinaryAccuracy` 类
    - 函数形式 `binary_accuracy`
* `CategoricalAccuracy` 类
    - 函数形式 `categorical_accuracy`
    - 分类准确率，与 `Accuracy` 含义相同
    - 要求 `y_true` 为 ont-hot 编码形式
* `Precision` 类
    - 精确率
    - `$Precision = \frac{TP}{TP + FP}$`
* `Recall` 类
    - 召回率
    - `$Recall = \frac{TP}{TP + FN}$`
* `TurePositives` 类
    - 真正例
* `TrueNegatives` 类
    - 真负例
* `FalsePositives` 类
    - 假正例
* `FalseNegatives` 类
    - 假负例
* `AUC` 类
    - ROC 曲线(TPR vs FPR)下的面积
    - 直观解释为: 随机抽取后一个正样本和一个负样本，正样本的预测值大于负样本的概率
* `PrecisionAtRecall` 类
* `SensitivityAtSpecificity` 类
* `SpecificityAtSensitivity` 类

### 多分类指标

- `SparseCategoricalAccuracy` 类
    - 稀疏分类准确率，与 `Accuracy` 含义相同
    - 要求 `y_true` 为序号编码形式
- `TopKCategoricalAccuracy` 类
    - 多分类 TopK 准确率
    - 要求 `y_true` 为 one-hot 编码形式
- `SparseTopKCategoricalAccuracy` 类
    - 稀疏多分类 TopK 准确率
    - 要求 `y_true` 为序号编码形式 

### 图像分割指标

- `MeanIoU` 类
    - Intersection Over Union
    - 常用于图像分割

### 其他指标

* Mean
* Sum

## 创建自定义评价指标

如果有需要，也可以自定义评价指标。
自定义评价指标需要接收两个张量 `y_true`、`y_pred` 作为输入参数，
并输出一个标量作为评价指标

### 类形式评价指标 

自定义评价指标需要继承 `tf.keras.metrics.Metric` 类, 
并重写 `__init__`、`update_state`、`result` 三个方法实现评价指标的计算逻辑，
总而得到评价指标的类的实现形式

由于训练的过程通常是分批次训练的，而评价指标要跑完一个 epoch 才能够得到整体的指标结果。
因此，类形式的评价指标更为常见。即需要编写初始化方法以创建与计算指标结果相关的一些中间变量，
编写 `update_state` 方法在每个 `batch` 后更新相关中间变量的状态，
编写 `result` 方法输出最终指标结果

* 示例 1

```python
import tensorflow as tf
from tensorflow.keras import metrics


class SparseCategoricalAccuracy(metrics.Metric):

    def __init__(self):
        super().__init__()
        self.total = self.add_weight(
            name = "total", 
            dtype = tf.int32, 
            initializer = tf.zeros_initializer()
        )
        self.count = self.add_weight(
            name = "total", 
            dtype = tf.int32, 
            initializer = tf.zeros_initializer()
        )

    def update_state(self, y_true, y_pred, sample_weight = None):
        values = tf.cast(
            tf.equal(
                y_true, 
                tf.argmax(y_pred, axis = 1, output_type = tf.int32)
            ), 
            tf.int32
        )
        self.total.assign_add(tf.shape(y_true)[0])
        self.count.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.count / self.total
```

* 示例 2

以金融风控领域常用的 KS 指标为例，示范自定义评估指标。
KS 指标适合二分类问题，其计算方式为 

`$$KS = max(TPR - FPR)$$`

其中:

* `$TPR = TP / (TP + FN)$` 
* `$FPR = FP / (FP + TN)$`

TPR 曲线实际上就是正样本的累积分布曲线(CDF)，FPR 曲线实际上就是负样本的累积分布曲线(CDF)。
KS 指标就是正样本和负样本累积分布曲线差值的最大值

```python
import tensorflow as tf
from tensorflow.keras import metrics


class KS(metrics.Metric):

    def __init__(self, name = "ks", **kwargs):
        super(KS, self).__init__(name = name, **kwargs)
        self.true_positives = self.add_weight(
            name = "tp",
            shape = (101,),
            initializer = "zeros",
        )
        self.false_positives = self.add_weight(
            name = "fp",
            shape = (101,),
            initializer = "zeros",
        )
    
    @tf.function
    def update_state(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, (-1,)), tf.bool)
        y_pred = tf.cast(100 * tf.reshape(y_pred, (-1,)), tf.int32)

        for i in tf.range(0, tf.shape(y_true)[0]):
            if y_true[i]:
                self.true_positives[y_pred[i]].assign(self.true_positives[y_pred[i]] + 1.0)
            else:
                self.false_positives[y_pred[i]].assign(self.false_positive[y_pred[i]] + 1.0)
        
        return (self.true_positives, self.false_positives)

    @tf.function
    def result(self):
        cum_positive_ratio = tf.truediv(
            tf.cumsum(self.true_positives),
            tf.reduce_sum(self.true_positives)
        )
        cum_negative_ratio = tf.truediv(
            tf.cumsum(self.false_positives),
            tf.reduce_sum(self.false_positives)
        )
        ks_value = tf.reduce_max(tf.abs(cum_positive_ratio - cum_negative_ratio))

        return ks_value


y_true = tf.constant(
    [[1],[1],[1],[0],[1],[1],[1],
     [0],[0],[0],[1],[0],[1],[0]]
)
y_pred = tf.constant(
    [[0.6],[0.1],[0.4],[0.5],[0.7],[0.7],[0.7],
     [0.4],[0.4],[0.5],[0.8],[0.3],[0.5],[0.3]]
)

myks = KS()
myks.update_state(y_true, y_pred)
tf.print(myks.result())
```

### 函数形式评价指标

如果编写函数形式的评价指标，则只能取 epoch 中各个 batch 计算的评价指标结果的平均值作为整个 epoch 上的评价指标结果，
这个结果通常会偏离整个 epoch 数据一次计算的结果

* 示例 1

```python
import tensorflow.keras.backend as K

def mean_pred(y_true, y_pred):
   return K.mean(y_pred)

model.compile(
    optimizers = "rmsprop",
    loss = "binary_accuracy",
    metrics = ["accuracy", mean_pred],
)
```

* 示例 2

```python
import tensorflow as tf

@tf.function
def ks(y_ture, y_pred):
    y_true = tf.reshape(y_true, (-1,))
    y_pred = tf.reshape(y_pred, (-1,))
    # 样本数量
    length = tf.shape(y_true)[0]
    # 排序
    t = tf.math.top_k(y_pred, k = length, sorted = False)
    y_pred_sorted = tf.gather(y_pred, t.indices)
    y_true_sorted = tf.gather(y_true, t.indices)

    cum_positive_ratio = tf.truediv(
        tf.cumsum(y_true_sorted),
        tf.reduce_sum(y_true_sorted)
    )
    cum_negative_ratio = tf.truediv(
        tf.cumsum(1 - y_true_sorted),
        tf.reduce_sum(1 - y_true_sorted)
    )
    ks_value = tf.reduce_max(tf.abs(cum_positive_ratio - cum_negative_ratio))
    
    return ks_value

y_true = tf.constant(
    [[1],[1],[1],[0],[1],[1],[1],
     [0],[0],[0],[1],[0],[1],[0]]
)
y_pred = tf.constant(
    [[0.6],[0.1],[0.4],[0.5],[0.7],[0.7],[0.7],
     [0.4],[0.4],[0.5],[0.8],[0.3],[0.5],[0.3]]
)
tf.print(ks(y_true,y_pred))
```

## 评价指标的使用—compile() & fit()

API:

```python
from tensorflow.keras import metrics
from tensorflow.keras.metrics import binary_accuracy
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.metrics import sparse_categorical_accuracy
from tensorflow.keras.metrics import top_k_categorical_accuracy
from tensorflow.keras.metrics import sparse_top_k_categorical_accuracy
from tensorflow.keras.metrics import mae

from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.losses import mean_absolute_error
from tensorflow.keras.losses import mean_absolute_percentage_error
from tensorflow.keras.losses import mean_squared_logarithmic_error
from tensorflow.keras.losses import squared_hinge
from tensorflow.keras.losses import hinge
from tensorflow.keras.losses import categorical_hinge
from tensorflow.keras.losses import logcosh
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.losses import kullback_leibler_divergence
from tensorflow.keras.losses import poisson
from tensorflow.keras.losses import cosine_proximity
```

Metrics Name:

```python
metrics = ["acc", "accuracy"]
```

## 评价指标的使用—单独使用

* TODO

