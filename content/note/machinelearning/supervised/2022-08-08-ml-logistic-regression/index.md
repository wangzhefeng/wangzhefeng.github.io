---
title: Logistic Regression
author: 王哲峰
date: '2022-08-08'
slug: ml-logistic-regression
categories:
  - machinelearning
tags:
  - machinelearning
  - model
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

- [模型推导-从线性模型出发](#模型推导-从线性模型出发)
- [角度2 模型推导-TODO](#角度2-模型推导-todo)
- [Loss Function 推导](#loss-function-推导)
- [Loss Function 优化方法](#loss-function-优化方法)
- [Logistic Regression 实现](#logistic-regression-实现)
  - [模型类型](#模型类型)
  - [模型形式](#模型形式)
  - [模型学习算法](#模型学习算法)
  - [Scikit API](#scikit-api)
    - [LogisticRegression](#logisticregression)
    - [SGDClassifier](#sgdclassifier)
    - [LogisticRegressionCV](#logisticregressioncv)
- [TODO](#todo)
</p></details><p></p>

# 模型推导-从线性模型出发

**线性模型:**

`$$y=f(x)=\omega \cdot x + b, y \in R$$`

**二分类模型:**

`$$y = f(x), y \in \{0, 1\}$$`

**伯努利分布:**

`$$y \sim b(0, p)$$`

假设事件发生(`$y=1$`)的概率为:

`$$p = P(y = 1)$$`

那么事件不发生(`$y=0$`)的概率为: 

`$$1-p = P(y = 0)$$`

**the odds of experiencing an event:**: 

`$$odds = \frac{p}{1-p}$$`

取对数: 

`$$log(odds)= \log\Big(\frac{p}{1-p}\Big)$$`

其中: 

`$$log(odds) \in [-\infty, +\infty]$$`

**线性模型:**

`$$\log\Big(\frac{p}{1-p}\Big) = g(x) = \omega \cdot x + b, \log\Big(\frac{p}{1-p}\Big) \in [-\infty, +\infty]$$`

因此: 

`$$p = \frac{1}{1+e^{-g(x)}}$$`

**Logistic Regression 模型:**

`$$y = f(x), y \in \{0, 1\}$$`

`$$\left\{\begin{array}{ll} P(y=1|x) =  \sigma(x) \\ P(y=0|x) = 1-\sigma(x) \end{array} \right.$$`

其中 `$\sigma(x)$`  为 sigmoid 函数: 

`$$\sigma(x) = \frac{1}{1+e^{-(\omega \cdot x + b)}}$$`

# 角度2 模型推导-TODO

# Loss Function 推导

**Logistic Regression 模型:**

`$$y = f(x), y \in \{0, 1\}$$`

`$$\left\{\begin{array}{ll} P(y=1|x) =  \sigma(x) \\ P(y=0|x) = 1-\sigma(x) \end{array} \right.$$`

其中 `$\sigma(x)$` 为sigmoid函数: 

`$$\sigma(x) = \frac{1}{1+e^{-(\omega \cdot x + b)}}$$`

**极大似然估计思想:**

给定数据集: `$\{(x_i, y_i)\}$`, 其中: `$i = 1, 2, \ldots, N$`, `$x_i \in R^p$`, `$y_i \in \{0, 1\}$` ；

似然函数为: 

`$$l=\prod_{i=1}^{N}[\sigma(x_i)]^{y_{i}}[1-\sigma{x_i}]^{1-y_i}$$`

则对数似然函数为: 

`$$\begin{eqnarray}L(\omega) & & {}= \log(l) \nonumber\\
             & & {}= \log\prod_{i=1}^{N}[\sigma(x_i)]^{y_i}[1-\sigma(x_i)]^{1-y_i} \nonumber\\
             & & {}= \sum_{i=1}^{N}\log[\sigma(x_i)]^{y_i}[1-\sigma(x_i)]^{1-y_i} \nonumber\\
             & & {}= \sum_{i=1}^{N}[\log[\sigma(x_i)]^{y_i}+\log[1-\sigma(x_i)]^{1-y_i}] \nonumber\\
             & & {}= \sum_{i=1}^{N}[y_i\log\sigma(x_i)+(1-y_i)\log[1-\sigma(x_i)]] \nonumber\\
             & & {}= \sum_{i=1}^{N}[y_i\log\frac{\sigma(x_i)}{1-\sigma(x_i)}+log[1-\sigma(x_i)]] \nonumber\\
             & & {}= \sum_{i=1}^{N}[y_i(\omega \cdot x_i)-\log(1+e^{\omega\cdot x_i})] \nonumber\\
             & & {}= \sum_{i=1}^{N}[y_i\log P(Y=1|x)+(1-y_i)\log(1-P(Y=1|x))] \nonumber\\
             & & {}= \sum_{i=1}^{N}[y_i\log \hat{y}_i+(1-y_i)\log(1-\hat{y}_i)] \nonumber
   \end{eqnarray}$$`

**Loss Function:**

`$$L(\omega) = - \sum_{i=1}^{N} [y_{i} \log \hat{y}_{i} + (1-y_{i}) \log(1- \hat{y}_{i})]$$`

# Loss Function 优化方法

- 梯度下降法
- 拟牛顿法

# Logistic Regression 实现

## 模型类型

- binray classification
- multiclass classification
   - One-vs-Rest classification
- Multinomial classification

## 模型形式

Logistic Regression with L1 正则化

`$$\min_{w, c} \|w\|_1 + C \sum_{i=1}^n \log(\exp(- y_i (X_i^T w + c)) + 1)$$`

Logistic Regression with L2 正则化

`$$\min_{w, c} \frac{1}{2}w^T w + C \sum_{i=1}^n \log(\exp(- y_i (X_i^T w + c)) + 1)$$`

## 模型学习算法

- liblinear
   - 坐标下降算法(coorinate descent algorithm, CD)
   - 算法稳健
- newton-cg
- lbfgs
   - 近似于Broyden-Fletcher-Goldfarb-Shanno算法的优化算法, 属于准牛顿方法
   - 适用于小数据集, 高维数据集
- sag
   - 随机平均梯度下降(Stochastic Average Gradient descent)
   - 适用于大数据集, 高维数据集
- saga
   - sag算法的变体
   - 适用于大数据集, 高维数据集
- SGDClassifier with log loss
   - 适用于大数据集, 高维数据集

## Scikit API

```python
   from sklearn.linear_model import LogisticRegression
   from sklearn.linear_model import LogisticRegressionCV
   from sklearn.linear_model import SGDClassifier 
```

### LogisticRegression

```python
lr = LogisticRegression(penalty = "l2", 
                        dual = False,
                        tol = 0.0001,
                        C = 1.0,
                        fit_intercept = True,
                        intercept_scaling = 1,
                        class_weight = None,
                        random_state = None, 
                        solver = "warn",
                        max_iter = 100,
                        multi_class = "warn",
                        verbose = 0,
                        warm_start = False,
                        n_jobs = None)
# Method
lr.fit()
lr.predict()
lr.predict_proba()
lr.predict_log_proba()
lr.decision_function()
lr.density()
lr.get_params()
lr.set_params()
lr.score()
lr.sparsify()

# Attributes
lr.classes_
lr.coef_
lr.intercept_
lr.n_iter_
```

- 多分类
   - multi_class = "ovr": 使用one-vs-rest模式
   - multi_class = "multinomial": 使用cross-entropy loss
      - 仅支持: `solver in ["lbfgs", "sag", "newton-cg"]`
- 其他
   - `dual = True, penalty = "l2"`
   - `solver in ["newton-cg", "sag", "lbfgs"], penalty = "l2"`
   - `solver = "liblinear", penalty in ["l2", "l1"]`

### SGDClassifier

```python
# 使用SGD算法训练的线性分类器: SVM, Logistic Regression
sgdc_lr = SGDClassifier(loss = 'log',
                        penalty = "l2",
                        alpha = 0.0001,
                        l1_ratio = 0.15,
                        fit_intercept = True,
                        max_iter = None, 
                        tol = None,
                        shuffle = True,
                        verbose = 0,
                        epsilon = 0.1, 
                        n_jobs = None,
                        random_state = None,
                        learning_rate = "optimal",
                        eta0 = 0.0,
                        power_t = 0.5, 
                        early_stopping = False,
                        validation_fraction = 0.1,
                        n_iter_no_change = 5,
                        class_weight = None, 
                        warm_start = False,
                        aveage = False,
                        n_iter = None)
```

### LogisticRegressionCV

```python
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# data
digits = datasets.load_digits()
X, y = digits.data, digits.target
X = StandardScaler().fit_transform(X)
y = (y > 4).astype(np.int)

for i, C in enmerate((1, 0.1, 0.01)):
    clf_l1_LR = LogisticRegression(C = C, penalty = "l1", )
```

# TODO

Logistic Regression 的目的是从特征学习出一个 0/1 分类模型 `$f(\cdot)$`：

`$$y = f(z), y \in \{0, 1\}$$`

Logistic Regression 模型是将特征变量的线性组合作为自变量: 

`$$z=\omega^{T}x + b$$`

由于自变量 `$x$` 取值的范围是 `$[-\infty, +\infty]$`，
因此需要使用 Logistic 函数(Sigmoid 函数)将自变量 `$z=\omega^{T}x + b$` 映射到范围 `$[0, 1]$` 上。
映射后的值被认为是 `$y=1$` 的概率。假设: 

`$$h_{\omega,b}(x)=\sigma(\omega^{T}x + b)$$`

其中：

* `$\sigma(z)$` 是 Sigmoid 函数 

`$$\sigma(z)=\frac{1}{1+e^{-z}}$$`

因此 Logistic Regression 模型的形式为： 

`$$\begin{cases}
P(y=1|x, \omega) = h_{\omega, b}(x) =\sigma(\omega^{T}x+b) \\ 
P(y=0|x, \omega) = 1 - h_{\omega, b}(x) =1-\sigma(\omega^{T}x+b) 
\end{cases}$$`

当要判别一个新来的数据点 `$x_{test}$` 属于哪个类别时, 
只需要求解 `$h_{\omega, b}(x_{test}) = \sigma(\omega^{T}x_{test} + b)$`: 

`$$ y_{test}=\left\{
\begin{array}{rcl}
1    &      & h_{\omega,b}(x_{test}) \geq 0.5 & \Leftrightarrow & \omega^{T}x_{test}+b \geq 0\\
0    &      & h_{\omega,b}(x_{test}) < 0.5 & \Leftrightarrow & \omega^{T}x_{test}+b < 0\\
\end{array} \right.$$`

Logistic Regression 的目标就是从数据中学习得到 `$\omega, b$`, 
使得正例 `$y=1$` 的特征 `$\omega^{T}x+b$` 远大于 `$0$`, 
负例 `$y=0$` 的特征 `$\omega^{T}x + b$` 远小于 `$0$`. 
