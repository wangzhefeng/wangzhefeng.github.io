---
title: 模型降维和特征选择
author: 王哲峰
date: '2022-09-13'
slug: model-decomposition
categories:
  - feature engine
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
</style>

<details><summary>目录</summary><p>

- [降维和特征选择](#降维和特征选择)
- [特征选择](#特征选择)
  - [过滤法](#过滤法)
    - [方差选择法](#方差选择法)
    - [相关系数法](#相关系数法)
    - [卡方检验](#卡方检验)
    - [互信息法](#互信息法)
  - [打包法](#打包法)
    - [递归特征消除法](#递归特征消除法)
  - [嵌入式方法](#嵌入式方法)
    - [基于正则化](#基于正则化)
    - [基于树模型](#基于树模型)
- [模型降维](#模型降维)
  - [主成分分析](#主成分分析)
    - [PCA 实现示例](#pca-实现示例)
  - [线性判别分析](#线性判别分析)
    - [LDA 实现示例](#lda-实现示例)
  - [t-SNE](#t-sne)
  - [独立分量分析](#独立分量分析)
  - [其他](#其他)
- [特征重要性评估](#特征重要性评估)
  - [随机森林](#随机森林)
- [参考](#参考)
</p></details><p></p>

# 降维和特征选择

在机器学习中，特征降维和特征选择是两个常见的概念。特征降维和特征选择的目的都是使数据的特征维数降低，
但实际上两者的区别是很大，它们的本质是完全不同的

特征选择从数据集中选择最重要特征的子集，特征选择不会改变原始特征的含义和数值，只是对原始特征进行筛选。
而降维将数据转换为低维空间，会改变原始特征中特征的含义和数值，可以理解为低维的特征映射。
这两种策略都可以用来提高机器学习模型的性能和可解释性，但它们的运作方式是截然不同的

![img](images/dr_fs.png)

# 特征选择

在数据集中选择一个特征子集用于机器学习模型的过程被称为特征选择。
特征选择的目的是发现对预测目标变量最相关和最重要的特征，
可以精简掉无用的特征，以降低最终模型的复杂性，它的最终目的是得到一个简约模型，
在不降低预测准确率或对预测准确率影响不大的情况下提高计算速度

使用特征选择有很多优点:

* 改进的模型可解释性：通过 减少模型中的特征数量，可以更容易地掌握和解释变量和模型预测之间的关系
* 降低过拟合的危险：当一个模型包含太多特征时，它更有可能过拟合，这意味着它在训练数据上表现良好，
  但在新的未知数据上表现不佳。通过选择最相关特征的子集，可以帮助限制过拟合的风险
* 改进模型性能：通过从模型中删除不相关或多余的特征，可以提高模型的性能和准确性

通常来说，从两个方面考虑来选择特征：

* 特征是否发散
    - 如果一个特征不发散，例如方差接近于 0，也就是说样本在这个特征上基本上没有差异，
      这个特征对于样本的区分并没有什么用
* 特征与目标的相关性
    - 这点比较显见，与目标相关性高的特征，应当优选选择。
      除方差法外，其他方法均从相关性考虑

根据特征选择的形式可以将特征选择方法分为三种：

* Filter：过滤法
    - 按照发散性或者相关性对各个特征进行评分，设定阈值或者待选择阈值的个数，选择特征
* Wrapper：包装法
    - 根据目标函数（通常是预测效果评分），每次选择若干特征，或者排除若干特征
* Embedded：嵌入法
    - 先使用某些机器学习的算法和模型进行训练，得到各个特征的权值系数，
      根据系数从大到小选择特征。类似于 Filter 方法，但是是通过训练来确定特征的优劣

可以使用 `sklearn` 中的 `feature_selection` 库来进行特征选择

## 过滤法

> Filter，过滤法

对特征进行预处理, 除去那些不太可能对模型有用处的特征。
一般通过计算特征与相应变量之间的相关性或互信息, 
然后过滤掉那些在某个阈值之下的特征

缺点：没有考虑模型, 可能无法为模型选择出正确的特征

### 方差选择法

使用方差选择法，先要计算各个特征的方差，然后根据阈值，选择方差大于阈值的特征






可以使用 `sklearn.feature_selection` 库的 `VarianceThreshold` 类来选择特征：

```python
from sklearn.feature_selection import VarianceThreshold

# 方差选择法，返回值为特征选择后的数据，参数 threshold 为方差的阈值
VarianceThreshold(threshold = 3).fit_transform(df.data)
```

### 相关系数法


### 卡方检验


### 互信息法





## 打包法

> Wrapper，打包法/包装法：试验特征的各个子集

### 递归特征消除法

递归消除特征法使用一个基模型来进行多轮训练，每轮训练后，
消除若干权值系数的特征，再基于新的特征集进行下一轮训练

可以使用 `sklearn.feature_selection` 库的 `RFE` 类来选择特征：

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# 递归特征消除法，返回特征选择后的数据
# 参数estimator为基模型
# 参数 n_features_to_select 为选择的特征个数
RFE(
    estimator = LogsiticRegression(), 
    n_features_to_select = 2
).fit_transform(
    iris.data, 
    iris.target
)
```

## 嵌入式方法

> Embedded，嵌入法

嵌入式方法将特征选择作为模型训练过程的一部分

* L1 正则化可以添加到任意线性模型的训练目标中，L1 正则化鼓励模型使用更少的特征，所以也称为稀疏性约束
    - LASSO
* 特征选择是决策树与生俱来的功能, 因为它在每个训练阶段都要选择一个特征来对树进行分割
    - 决策树
    - GBM
    - XGBoost
    - LightGBM
    - CatBoost
    - RandomForest

### 基于正则化

使用带惩罚项的基模型，除了筛选出特征外，同时也进行了降维

使用 `sklearn.feature_selection` 库的 `SelectFromModel` 类结合带 L1 惩罚项的逻辑回归模型：

```python
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

SelectFromModel(
    LogisticRegression(penalty = "l1", C = 0.1)
).fit_transform(iris.data, iris.target)
```

L1 惩罚项降维的原理在于保留多个对目标值具有同等相关性的特征中的一个，所以没选到的特征不代表不重要。
所以可结合 L2 惩罚项来优化，具体操作为：若一个特征在 L1 中的权值为 1，
选择在 L2 中权值差别不大且在 L1 中权值为 0 的特征构成同类集合，
将这一集合中的特征平分 L1 中的权值，故需要构建一个新的逻辑回归模型

```python
from sklearn.linear_model import LogisticRegression 

class LR(LogisticRegression): 

    def __init__(self, 
                 threshold = 0.01, 
                 dual = False, 
                 tol = 1e-4, 
                 C = 1.0,
                 fit_intercept = True, 
                 intercept_scaling = 1, 
                 class_weight = None,
                 random_state = None, 
                 solver = 'liblinear', 
                 max_iter = 100,
                 multi_class = 'ovr', 
                 verbose = 0, 
                 warm_start = False, 
                 n_jobs = 1):
        # 权值相近的阈值
        self.threshold = threshold 
        LogisticRegression.__init__(
            self, 
            penalty='l1', 
            dual = dual, 
            tol = tol, 
            C = C, 
            fit_intercept = fit_intercept, 
            intercept_scaling = intercept_scaling, 
            class_weight = class_weight, 
            random_state = random_state, 
            solver = solver, 
            max_iter = max_iter, 
            multi_class = multi_class, 
            verbose = verbose, 
            warm_start = warm_start, 
            n_jobs = n_jobs
        ) 
        # 使用同样的参数创建 L2 逻辑回归
        self.l2 = LogisticRegression(
            penalty = 'l2', 
            dual = dual, 
            tol = tol, 
            C = C, 
            fit_intercept = fit_intercept, 
            intercept_scaling = intercept_scaling, 
            class_weight = class_weight, 
            random_state = random_state, 
            solver = solver, 
            max_iter = max_iter, 
            multi_class = multi_class, 
            verbose = verbose, 
            warm_start = warm_start, 
            n_jobs = n_jobs
        ) 

    def fit(self, X, y, sample_weight = None): 
        # 训练 L1 逻辑回归
        super(LR, self).fit(X, y, sample_weight = sample_weight)
        self.coef_old_ = self.coef_.copy() 
        # 训练 L2 逻辑回归
        self.l2.fit(X, y, sample_weight = sample_weight) 

        cntOfRow, cntOfCol = self.coef_.shape 
        # 权值系数矩阵的行数对应目标值的种类数目
        for i in range(cntOfRow):
            for j in range(cntOfCol): 
                coef = self.coef_[i][j] 
                # L1 逻辑回归的权值系数不为 0
                if coef != 0: 
                    idx = [j] 
                    # 对应在 L2 逻辑回归中的权值系数
                    coef1 = self.l2.coef_[i][j] 
                    for k in range(cntOfCol): 
                        coef2 = self.l2.coef_[i][k] 
                        # 在 L2 逻辑回归中，权值系数之差小于设定的阈值，且在 L1 中对应的权值为0
                        if abs(coef1-coef2) < self.threshold and j != k and self.coef_[i][k] == 0: 
                            idx.append(k) 
                    # 计算这一类特征的权值系数均值
                    mean = coef / len(idx) 
                    self.coef_[i][idx] = mean 
        return self
```

使用 `sklearn.feature_selection` 库的 `SelectFromModel` 类结合带 L1 以及 L2 惩罚项的逻辑回归模型，
来选择特征的代码如下：

```python
from sklearn.feature_selection import SelectFromModel

# 带 L1 和 L2 惩罚项的逻辑回归作为基模型的特征选择
# 参数 threshold 为权值系数之差的阈值
SelectFromModel(LR(threshold = 0.5, C = 0.1)).fit_transform(iris.data, iris.target)
```

### 基于树模型

树模型中 GBDT 也可用来作为基模型进行特征选择

使用 `sklearn.feature_selection` 库的 `SelectFromModel` 类结合 GBDT 模型

```python
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier

SelectFromModel(
    GradientBoostingClassifier()
).fit_transform(
    iris.data, 
    iris.target
)
```

# 模型降维

降低数据集中特征的维数，同时保持尽可能多的信息的技术被称为降维。
可以最大限度地降低数据复杂性并提高模型性能

当特征选择完成后，可以直接训练模型了，但是可能由于特征矩阵过大，导致计算量大、训练时间长，
因此降低特征矩阵维度也是必不可少的。常见的降维方法除了以上提到的基于 L1 惩罚项的模型以外，
另外还有主成分分析法（PCA）和线性判别分析（LDA），线性判别分析本身也是一个分类模型

PCA 和 LDA 有很多的相似点，其本质是要将原始的样本映射到维度更低的样本空间中，
但是 PCA 和 LDA 的映射目标不一样：

* PCA 是为了让映射后的样本具有最大的发散性
* LDA 是为了让映射后的样本有最好的分类性能
 
所以说 PCA 是一种无监督的降维方法，而 LDA 是一种有监督的降维方法

## 主成分分析

主成分分析 (PCA) 是一种统计方法，可识别一组不相关的变量，将原始变量进行线性组合，称为主成分。
第一个主成分解释了数据中最大的方差，然后每个后续成分解释主键变少。PCA 经常用作机器学习算法的数据预处理步骤，
因为它有助于降低数据复杂性并提高模型性能

### PCA 实现示例

使用 `sklearn.decomposition` 库的 `PCA` 类选择特征

```python
from sklearn.decomposition import PCA

# 主成分分析法，返回降维后的数据
# 参数 n_components 为主成分数目
PCA(n_components = 2).fit_transform(iris.data)
```

## 线性判别分析

线性判别分析（LDA）是一种用于分类工作的统计工具。它的工作原理是确定数据属性的线性组合，
最大限度地分离不同类别。为了提高模型性能，LDA 经常与其他分类技术(如逻辑回归或支持向量机)结合使用

### LDA 实现示例

使用 `sklearn.lda` 库的 `LDA` 类选择特征

```python
from sklearn.lda import LDA

# 线性判别分析法，返回降维后的数据
# 参数 n_components 为降维后的维数
LDA(n_components = 2).fit_transform(iris.data, iris.target)
```

## t-SNE

t-SNE(t-分布随机邻居嵌入)是一种非线性降维方法，特别适用于显示高维数据集。
它保留数据的局部结构来，也就是说在原始空间中靠近的点在低维空间中也会靠近。
t-SNE 经常用于数据可视化，因为它可以帮助识别数据中的模式和关系

## 独立分量分析

独立分量分析（Independent Component Analysis，ICA）实际上也是对数据在原有特征空间中做的一个线性变换。
相对于 PCA 这种降秩操作，ICA 并不是通过在不同方向上方差的大小，即数据在该方向上的分散程度来判断那些是主要成分，
那些是不需要到特征。而 ICA 并没有设定一个所谓主要成分和次要成分的概念，ICA 认为所有的成分同等重要，
而我们的目标并非将重要特征提取出来，而是找到一个线性变换，使得变换后的结果具有最强的独立性。
PCA 中的不相关太弱，我们希望数据的各阶统计量都能利用，即我们利用大于 2 的统计量来表征。而 ICA 并不要求特征是正交的

## 其他

* 多维缩放
* 自编码器

# 特征重要性评估

## 随机森林

```python
import numpy as np
import pandas as pd
try:
    from sklearn.cross_validation import train_test_split
except:
    from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# data
data_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
df = pd.read_csv(data_url, header = None)
df.columns = [
    'Class label', 'Alcohol', 'Malic acid', 'Ash', 
    'Alcalinity of ash', 'Magnesium', 'Total phenols', 
    'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 
    'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline'
]

# data split
x, y = df.iloc[:, 1:].values, df.iloc[:, 0].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)
feat_labels = df.columns[1:]

# model
rf_clf = RandomForestClassifier(
    n_estimators = 10000,
    random_state = 0,
    n_jobs = -1,
)
rf_clf.fit(x_train, y_train)

importances = rf_clf.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(x_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))

# features selection
threshold = 0.15
x_selected = x_train[:, importances > threshold]
print(x_selected.shape)
```

# 参考

* [利用随机森林评估特征重要性原理与应用](https://mp.weixin.qq.com/s/2O9k0FSY15aHRSZ8D6B5Gg)
* [用机器学习神器sklearn做特征工程](https://mp.weixin.qq.com/s/AwjEfC2wLhUF9Ecgt0kocw)
* [IRIS（鸢尾花）数据集](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html#sklearn.datasets.load_iris)
* [通常使用哑编码的方式将定性特征转换为定量特征](http://www.ats.ucla.edu/stat/mult_pkg/faq/general/dummy.html)
* [L1惩罚项降维的原理在于保留多个对目标值具有同等相关性的特征中的一个](http://www.zhihu.com/question/28641663/answer/41653367)
* [PCA是为了让映射后的样本具有最大的发散性；而LDA是为了让映射后的样本有最好的分类性能](http://www.cnblogs.com/LeftNotEasy/archive/2011/01/08/lda-and-pca-machine-learning.html)
* [《使用sklearn优雅地进行数据挖掘》](http://www.cnblogs.com/jasonfreak/p/5448462.html)
* [FAQ: What is dummy coding?](http://www.ats.ucla.edu/stat/mult_pkg/faq/general/dummy.htm)
* [卡方检验](http://wiki.mbalib.com/wiki/%E5%8D%A1%E6%96%B9%E6%A3%80%E9%AA%8C)
* [干货：结合Scikit-learn介绍几种常用的特征选择方法](http://dataunion.org/14072.html)
* [机器学习中，有哪些特征选择的工程方法？](http://www.zhihu.com/question/28641663/answer/41653367)
* [机器学习中的数学(4)-线性判别分析（LDA）, 主成分分析(PCA)](http://www.cnblogs.com/LeftNotEasy/archive/2011/01/08/lda-and-pca-machine-learning.html)
* [特征选择方法总结](https://mp.weixin.qq.com/s?__biz=MzUyNzA1OTcxNg==&mid=2247486395&idx=1&sn=df2f5838314fb89d4f7eca03b5339a2a&chksm=fa0415d0cd739cc6a01bd0f9ece6a8d4a3285b61bbee39d66acbff1f9579f43f3281bfcc4914&scene=178&cur_album_id=1577157748566310916#rd)
