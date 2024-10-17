---
title: 模型验证
author: wangzf
date: '2022-11-13'
slug: model-validation
categories:
  - machinelearning
tags:
  - machinelearning
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

- [模型验证介绍](#模型验证介绍)
- [模型验证工具](#模型验证工具)
  - [evidently](#evidently)
    - [安装](#安装)
    - [使用-数据统计](#使用-数据统计)
    - [使用-机器学习](#使用-机器学习)
  - [deepchecks](#deepchecks)
    - [安装](#安装-1)
    - [使用](#使用)
  - [TFDV](#tfdv)
    - [安装](#安装-2)
    - [使用](#使用-1)
- [模型交叉验证](#模型交叉验证)
  - [Hold Out](#hold-out)
  - [K-Fold](#k-fold)
    - [K-Fold](#k-fold-1)
    - [分层 K-Fold](#分层-k-fold)
    - [分层分组 K-Fold](#分层分组-k-fold)
- [参考](#参考)
</p></details><p></p>

# 模型验证介绍

机器学习模型项目不是一次性的,它是一个持续的过程。如果存在任何异常，则需要对生产中的模型进行持续监控

# 模型验证工具

## evidently

Evidently 是一个用于分析和监控机器学习模型的开源 Python 包。开发该软件包的目的是建立一个易于监控的机器学习仪表盘，
并检测数据中的漂移。它是专门为生产而设计的，所以在有数据管道的情况下使用它会更好。然而，即使在开发阶段，仍然可以使用它

### 安装

```bash
$ pip install evidently
```

### 使用-数据统计

```python
import pandas as pd
from evidently.dashboard import Dashboard
from evidently.tabs import DataDriftTab
```

可以尝试检测数据集中发生的数据漂移。数据漂移是指参考数据或之前时间线中的数据与当前数据在统计上存在差异的现象

```python
train = pd.read_csv("churn-bigml-80.csv")
test = pd.read_csv("churn-bigml-20.csv")
train.drop(["State", "International plan", "Voice mail plan"], axis = 1, inplace = True)
test.drop(["State", "International plan", "Voice mail plan"], axis = 1, inplace = True)
train["Churn"] = train["Churn"].apply(lambda x: 1 if x == True else 0)
test["Churn"] = test["Churn"].apply(lambda x: 1 if x == True else 0)
```

数据准备好后，将构建仪表板来检测任何漂移。显然需要我们独立导入每个标签；对于数据漂移，我们将使用 `DataFloftTab`：

```python
data_drift_report = Dashboard(tabs = [DataDriftTab()])
data_drift_report.calculate(train, test, column_mapping = None)
data_drift_report.save("reports/my_report.html")
```

有一个监视器仪表板。我们可以在这个仪表板中看到每个特征分布和数据漂移的统计测试。
在我们的样本中，训练数据和测试数据之间没有显示任何漂移，这意味着所有数据分布都是相似的

### 使用-机器学习

可以用来创建一个机器学习分类仪表板来监控机器学习的健康状况。例如，让我们使用之前的数据训练一个分类模型

```python
from sklearn.neighbors import KNeighborsClassifier

# data
X_train = train.drop('Churn', axis =1)
X_test = test.drop('Churn', axis =1)
y_train = train['Churn']
y_test = test['Churn']

# model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
```

在拟合模型之后，我们需要实际结果和预测结果。我们还需要训练和测试数据集完好无损

```python
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)
X_train['target'] = y_train
X_train['prediction'] = train_predictions
X_test['target'] = y_test
X_test['prediction'] = test_predictions
```

监视器还需要映射使用的列。在本例中，将使用中的 `ColumnMapping` 进行设置：

```python
from evidently.pipeline.column_mapping import ColumnMapping


churn_column_mapping = ColumnMapping()
churn_column_mapping.target = 'target'
churn_column_mapping.prediction = 'prediction'
churn_column_mapping.numerical_features = train.drop('Churn', axis = 1).columns
```

所有准备工作完成后，设置分类器监视器仪表板：

```python
from evidently.tabs.base_tab import Verbose
from evidently.tabs import ClassificationPerformanceTab


churn_model_performance_dashboard = Dashboard(
    tabs = [
        ClassificationPerformanceTab(verbose_level = Verbose.FULL)
    ]
)
churn_model_performance_dashboard.calculate(
    X_train, 
    X_test, 
    column_mapping = churn_column_mapping
)
churn_model_performance_dashboard.save("reports/classification_churn.html")
```


如上面所示，我们可以监控机器学习模型指标和每个特征的预测，可以知道在接收新数据时是否存在差异

## deepchecks

deepchecks 是一个 Python 工具包，只需用几行代码就可以验证机器学习模型。
许多 API 可用于检测数据漂移、标签漂移、列车测试比较、评估模型等。
deepchecks 非常适合在研究阶段和模型投产前使用

deepchecks 完整报告包含许多信息，例如混淆矩阵、简单模型比较、混合数据类型、数据漂移等。
检查机器学习模型所需的所有信息都可以在单个代码运行中获得

### 安装

```bash
$ pip isntall deepchecks
```

### 使用

```python
import pandas as pd

from deepchecks.datasets.classification import iris
from deepchecks import Dataset
from deepchecks.suites import full_suite

from sklearn.ensemble import RandomForestClassifier
```

加载数据，拆分训练、测试数据，并加载机器学习模型:

```python
# data
df_train, df_test = iris.load_data(
    data_format = "Dataframe", 
    as_train_test = True
)
label_col = "target"

# model
rf_clf = iris.load_fitted_model()
```

如果 Deepchecks 将 pandas 数据帧转换为 deepchecks 数据集对象，数据会更好处理：

```python
ds_train = Dataset(df_train, label = label_col, cat_features = [])
ds_test = Dataset(df_test, label = label_col, cat_features = [])
```

进行数据验证：

```python
suite = full_suite()
suite.run(
    train_dataset = ds_train, 
    test_dataset = ds_test, 
    model = rf_clf
)
```

## TFDV

> tensorflow-data-validation

TFDV(TensorFlow Data Validation) 是 TensorFlow 开发人员开发的用于管理数据质量问题的 python 包。
它用于自动描述数据统计、推断数据模式以及检测传入数据中的任何异常

### 安装

```bash
$ pip install tesnsorflow-data-validation
```

### 使用

```python
import tensorflow_data_validation as tfdv
```

加载数据：

```python
stats = tfdv.generate_statistics_from_csv(
    data_location = "churn-bigml-80.csv"
)
```

可以将统计对象的统计信息可视化：

```python
tfdv.visualize_statistics(stats)
```

TFDV 包不仅限于生成统计可视化，还有助于检测传入数据中的任何变化。
为此，需要推断原始或参考数据模式：

```python
schema = tfdv.infeer_schema(stats)
tfdv.display_schema(schema)
```

该模式将用于针对任何传入数据进行验证，如果传入数据没有在模式中推断出任何列或类别，
那么 TFDV 将通知异常的存在。将使用以下代码和测试数据来实现这一点：

```python
new_csv_stats = tfdv.generate_statistics_from_csv(
    data_location = "churn-bigml-0.csv"
)
anomalies = tfdv.validate_statistics(
    statistics = new_csv_stats, 
    schema = schema
)
tfdv.display_anomalies(anomalies)
```

# 模型交叉验证

## Hold Out

## K-Fold

### K-Fold

### 分层 K-Fold

### 分层分组 K-Fold



# 参考

* [机器学习模型验证，这 3 个 Python 包可轻松解决 95% 的需求](https://mp.weixin.qq.com/s?__biz=MzA3MTM5MDYyMA==&mid=2656763105&idx=1&sn=2c481069a1d2849b1f816075ba122df9&chksm=84801812b3f79104a6ec6f26db92dc452342ebcce6fcb6ff2928d966d9ea90d182baed1a2b52&scene=132#wechat_redirect)
* [交叉验证常见的6个错误](https://mp.weixin.qq.com/s/OirhNWfpz-mRJpSb_CwIXA)
* [综述论文：机器学习中的模型评价、模型选择与算法选择](https://mp.weixin.qq.com/s/F7mgeYGzTxO6jFeIbqHbqw)
* [验证策略设计](https://mp.weixin.qq.com/s?__biz=Mzk0NDE5Nzg1Ng==&mid=2247493351&idx=1&sn=d4144b64da66d4bc91ed0716596476ce&chksm=c32aff68f45d767e7392de9fc12814a229130d6ca0c41764ba1fbcb437a909a7560a22d9cd09&scene=21#wechat_redirect)

