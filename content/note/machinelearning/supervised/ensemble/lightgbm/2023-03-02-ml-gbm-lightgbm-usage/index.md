---
title: LightGBM 使用
author: 王哲峰
date: '2023-03-02'
slug: ml-gbm-lightgbm-usage
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
</style>

<details><summary>目录</summary><p>

- [LightGBM 调用方法](#lightgbm-调用方法)
  - [定义数据集](#定义数据集)
  - [训练模型](#训练模型)
  - [模型保存与加载](#模型保存与加载)
  - [查看特征重要性](#查看特征重要性)
  - [继续训练](#继续训练)
  - [动态调整模型超参数](#动态调整模型超参数)
  - [自定义损失函数](#自定义损失函数)
  - [网格搜索](#网格搜索)
  - [贝叶斯优化](#贝叶斯优化)
- [LightGBM 调参参数](#lightgbm-调参参数)
  - [参数设置方式](#参数设置方式)
    - [命令行参数](#命令行参数)
    - [参数配置文件](#参数配置文件)
    - [Python 参数字典](#python-参数字典)
  - [参数类型](#参数类型)
    - [Booster 参数](#booster-参数)
  - [参数初始化](#参数初始化)
    - [回归](#回归)
    - [分类](#分类)
    - [排序](#排序)
  - [调参技巧](#调参技巧)
- [参考](#参考)
</p></details><p></p>

# LightGBM 调用方法

在 Python 中 LightGBM 提供了两种调用方式，分为为原生的 API 和 Scikit-learn API，
两种方式都可以完成训练和验证。当然原生的 API 更加灵活，看个人习惯来进行选择

## 定义数据集

读取数据：

```python
import pandas as pd

df_train = pd.read_csv(
    'https://cdn.coggle.club/LightGBM/examples/binary_classification/binary.train', 
    header = None, 
    sep = '\t'
)
df_test = pd.read_csv(
    'https://cdn.coggle.club/LightGBM/examples/binary_classification/binary.test', 
    header = None, 
    sep = '\t'
)
W_train = pd.read_csv(
    'https://cdn.coggle.club/LightGBM/examples/binary_classification/binary.train.weight', 
    header = None
)[0]
W_test = pd.read_csv(
    'https://cdn.coggle.club/LightGBM/examples/binary_classification/binary.test.weight', 
    header = None
)[0]

y_train = df_train[0]
y_test = df_test[0]
X_train = df_train.drop(0, axis = 1)
X_test = df_test.drop(0, axis = 1)

num_train, num_feature = X_train.shape
```

LightGBM Dataset：

```python
import lightgbm as lgb

lgb_train = lgb.Dataset(
    X_train, 
    y_train, 
    weight = W_train, 
    free_raw_data = False
)
lgb_eval = lgb.Dataset(
    X_test, 
    y_test, 
    reference = lgb_train, 
    weight = W_test, 
    free_raw_data = False
)
```

## 训练模型

```python
# 模型超参数
params = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": 0,
}

# generate features names
feature_name = ["feature_" + str(col) for col in range(num_feature)]

# model
gbm = lgb.train(
    params,
    lgb_train,
    num_boost_round = 10,
    valid_sets = lgb_train,  # eval training data
    feature_name = feature_name,
    categorical_feature = [21],
)
```

## 模型保存与加载

Txt：

```python
# txt
gbm.save_model("model.txt")
```

JSON：

```python
# json
print("Dumping model to JSON...")
model_json = gbm.dump_model()
with open("model.json", "w+") as f:
    json.dump(model_json, f, indent = 4)
```

## 查看特征重要性

```python
# feature names
print(f"Feature names: {gbm.feature_name()}")

# feature importances
print(f"Feature importances: {list(gbm.feature_importance())}")
```

## 继续训练

```python
gbm = lgb.train(
    params,
    lgb_train,
    num_boost_round = 10,
    init_model = "model.txt",
    valid_sets = lgb_eval,
)
print("Finished 10-20 rounds with model file...")
```

## 动态调整模型超参数

```python
gbm = lgb.train(
    params,
    lgb_train,
    num_boost_round = 10,
    init_model = gbm,
    learning_rates = lambda iter: 0.05 * (0.99 ** iter),
    valid_sets = lgb_eval
)
print("Finished 20-30 rounds with decay learning rates...")


gbm = lgb.train(
    params,
    lgb_train,
    num_boost_round = 10,
    init_model = gbm,
    valid_sets = lgb_eval,
    callbacks = [
        lgb.reset_parameter(bagging_fraction = [0.7] * 5 + [0.6] * 5)
    ]
)
print('Finished 30 - 40 rounds with changing bagging_fraction...')
```

## 自定义损失函数

```python
def loglikelihood(preds, train_data):
    """
    self-defined objective function: log likelihood loss
        - f(preds: array, train_data: Dataset) -> grad: array, hess: array
    """
    labels = train_data.get_label()
    preds = 1. / (1. + np.exp(-preds))
    grad = preds - labels
    hess = preds * (1. - preds)

    return grad, hess


def binary_error(preds, train_data):
    """
    self-defined eval metric: binary error
        - f(preds: array, train_data: Dataset) -> name: string, eval_result: float, is_higher_better: bool
    NOTE: when you do customized loss function, the default prediction value is margin
        This may make built-in evalution metric calculate wrong results
        For example, we are doing log likelihood loss, the prediction is score before logistic transformation
        Keep this in mind when you use the customization
    """
    labels = train_data.get_label()
    preds = 1. / (1. + np.exp(-preds))

    return 'error', np.mean(labels != (preds > 0.5)), False


gbm = lgb.train(
    params,
    lgb_train,
    num_boost_round = 10,
    init_model = gbm,
    fobj = loglikelihood,
    feval = binary_error,
    valid_sets = lgb_eval
)
print('Finished 40 - 50 rounds with self-defined objective function and eval metric...')
```

## 网格搜索

```python
# model
lgbc = lgb.LGBMClassifier(silent = False)

# hyperparameters
param_dist = {
    "max_depth": [4, 5, 7],
    "learning_rate" : [0.01, 0.05, 0.1],
    "num_leaves": [300, 900, 1200],
    "n_estimators": [50, 100, 150]
}

# grid search cv
grid_search = GridSearchCV(
    lgbc, 
    n_jobs = -1, 
    param_grid = param_dist, 
    cv = 5, 
    scoring = "roc_auc", 
    verbose = 5
)
grid_search.fit(X_train, y_train)

# best hyper parameters
print(grid_search.best_estimator_)
print(grid_search.best_score_)
```

## 贝叶斯优化

```python
import warnings
import time
warnings.filterwarnings("ignore")
from bayes_opt import BayesianOptimization

def lgb_eval(max_depth, learning_rate, num_leaves, n_estimators):
    params = {
        "metric": 'auc'
    }
    params['max_depth'] = int(max(max_depth, 1))
    params['learning_rate'] = np.clip(0, 1, learning_rate)
    params['num_leaves'] = int(max(num_leaves, 1))
    params['n_estimators'] = int(max(n_estimators, 1))

    cv_result = lgb.cv(
        params, 
        d_train, 
        nfold = 5, 
        seed = 0, 
        verbose_eval = 200,
        stratified = False
    )

    return 1.0 * np.array(cv_result['auc-mean']).max()

lgbBO = BayesianOptimization(
    lgb_eval, 
    {
        'max_depth': (4, 8),
        'learning_rate': (0.05, 0.2),
        'num_leaves' : (20, 1500),
        'n_estimators': (5, 200)
    }, 
    random_state = 0
)

lgbBO.maximize(init_points = 5, n_iter = 50, acq = 'ei')
print(lgbBO.max)
```

# LightGBM 调参参数

## 参数设置方式

* 命令行参数
* 参数配置文件
* Python 参数字典

### 命令行参数

```bash
$ 
```

### 参数配置文件

```python

```

### Python 参数字典

```python

```

## 参数类型

* 核心参数
    - Booster 参数
* 学习控制参数
* IO 参数
* 目标参数
* 度量参数
* 网络参数
* GPU 参数
* 模型参数
* 其他参数

### Booster 参数

```python
param = {
   'num_levels': 31,
   'num_trees': 100,
   'objective': 'binary',
   'metirc': ['auc', 'binary_logloss']
}
```


## 参数初始化

### 回归


### 分类


### 排序


## 调参技巧

<font size=0.01em>

|  参数 | 提高模型训练速度 | 提高准确率 | 加强拟合 | 处理过拟合(降低拟合) |
|------|---------------|----|----|----|
| 训练数据 | | 增加训练数据 | 增加训练数据 | 增加训练数据 |
| `dart` | 尝试使用 `dart`(带dropout的树) | 尝试使用 `dart`(带dropout的树) | |
| `bagging_fraction`</br>`bagging_freq` | 使用 bagging 样本采样，设置`bagging_fraction`</br>`bagging_freq` | | | 使用 bagging 样本采样，设置`bagging_fraction`</br>`bagging_freq` |
| `feature_fraction` | 使用子特征采样，</br>设置`feature_fraction` | | | 使用子特征采样，</br>设置</br>`feature_fraction` |
| `learning_rate` | | 使用较小的`learning_rate` 和较大的`num_iterations` | 使用较小的`learning_rate` 和较大的`num_iterations` | |
| `max_bin`</br>(会减慢训练速度) | 使用较小的</br>`max_bin` | 使用较大的</br>`max_bin` | 使用较大的</br>`max_bin` | 使用较小的</br>`max_bin` |
| `num_leaves`| 使用较小的</br>`num_leaves` | 使用较大的</br>`num_leaves` | 使用较大的</br>`num_leaves` | 使用较小的</br>`num_leaves` |
| `max_depth` | | | | 尝试`max_depth` 避免树生长太深 |
| `min_data_in_leaf`和`min_sum_`</br>`hessian_in_leaf`| | | | 使用`min_data_in_leaf` 和`min_sum_`</br>`hessian_in_leaf` |
| `lambda_l1`、`lambda_l2`、`min_gain_to_split` | | | | 尝试使用`lambda_l1`、`lambda_l2` 和`min_gain_to_split` 做正则化 |
| `path_smooth` | | | | 尝试提高</br>`path_smooth` |
| `extra_trees` | | | | 尝试使用`extra_trees` |
| `save_binary` | 使用`save_binary`加速数据加载速度 | | | |
| 并行           | 使用并行训练                     | | | |

</font>

# 参考

* [LightGBM各种操作](https://mp.weixin.qq.com/s/iLgVeAKXR3EJW2mHbEWiFQ)

