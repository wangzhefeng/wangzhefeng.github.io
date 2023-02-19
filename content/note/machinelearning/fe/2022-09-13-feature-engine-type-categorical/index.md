---
title: Categorical
author: 王哲峰
date: '2022-09-13'
slug: feature-engine-type-categorical
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

- [类别特征编码介绍](#类别特征编码介绍)
  - [无序类别特征](#无序类别特征)
  - [有序类别特征](#有序类别特征)
- [标签编码](#标签编码)
  - [LabelEncoder](#labelencoder)
  - [LabelBinarizer](#labelbinarizer)
- [哈希编码](#哈希编码)
  - [HashingEncoder](#hashingencoder)
- [独热编码](#独热编码)
  - [get\_dummies](#get_dummies)
- [计数编码](#计数编码)
  - [CountEncoder](#countencoder)
- [直方图编码](#直方图编码)
  - [HistEncoder](#histencoder)
- [WOE 编码](#woe-编码)
  - [WOEEncoder](#woeencoder)
  - [Information Value](#information-value)
- [Target 编码](#target-编码)
  - [Leave-One-Out Mean-Target 编码](#leave-one-out-mean-target-编码)
  - [K-fold Mean-Target 编码](#k-fold-mean-target-编码)
  - [Bayesian Target 编码](#bayesian-target-编码)
  - [Beta Target 编码](#beta-target-编码)
    - [BetaEncoder](#betaencoder)
- [平均编码](#平均编码)
  - [MeanEncoder](#meanencoder)
- [模型编码](#模型编码)
  - [LightGBM GS 编码](#lightgbm-gs-编码)
  - [CatBoost Ordered TS 编码](#catboost-ordered-ts-编码)
    - [CatBoostEncoder](#catboostencoder)
- [M Estimator Encoding](#m-estimator-encoding)
- [James Stein Encoding](#james-stein-encoding)
- [有序字典编码](#有序字典编码)
  - [map](#map)
- [分段编码](#分段编码)
  - [Python Code](#python-code)
- [Hashing](#hashing)
  - [FeatureHasher](#featurehasher)
- [Helmert Contrast](#helmert-contrast)
  - [HelmertEncoder](#helmertencoder)
- [Sum Encoding](#sum-encoding)
- [人工编码](#人工编码)
  - [人工转化编码](#人工转化编码)
  - [人工组合编码](#人工组合编码)
- [分箱计数](#分箱计数)
- [参考](#参考)
</p></details><p></p>

# 类别特征编码介绍

类别型特征原始输入通常是字符串形式, 除了基于决策树模型的少数模型能够直接处理字符串形式的输入, 
其他模型需要将类别型特征转换为数值型特征

在很多表格类的问题中，高基数的特征类别处理一直是一个困扰着很多人的问题，究竟哪一种操作是最好的，
很难说，不同的数据集有不同的特性，可能某一种数据转化操作这 A 数据集上取得了提升，
但在 B 数据集上就不行了，但是知道的技巧越多，我们能取得提升的概率往往也会越大。
此处我们会介绍几种常见的处理类别特征的方法。按照不同的划分标准，类别型特征可以分为：

* 按照类别是否有序
    - 无序类别特征
    - 有序类别特征
* 按照类别数量
    - 高基数类
    - 低基数类

总结来说，关于类别特征，有以下心得：

![img](images/encoder.png)

1. 统计类编码常常不适用于小样本，因为统计意义不明显
2. 当训练集和测试集分布不一致时，统计类编码往往会有预测偏移问题，所以一般会考虑结合交叉验证
3. 编码后特征数变多的编码方法，不适用于高基类的特征，会带来稀疏性和训练成本
4. 没有完美的编码方法，但感觉标签编码、平均编码、WOE 编码和模型编码比较常用

## 无序类别特征


## 有序类别特征

有序类别特征，故名思意，就是有相对顺序的类别特征。例如：

* 年龄段特征："1-10, 11-20, 21-30, 31-40" 等年龄段
* 评分特征："high, medium, low"

有序类别特征和无序的类别特征有些许区别，例如：标签编码等，
如果我们直接按照原先的标签编码进行转化就会丢失特征相对大小的信息，
这对于梯度提升树模型会带来负向的效果，因为序列信息可能和标签有着强烈的相关性，
比如回购的问题，有 “high, medium, low” 三种评分，用户购物之后如果给商品打了 “high“，
那么他大概率还会回购，但是如果打了 “low”，那么大概率是不会回购了，
传统的标签编码就直接丢失了这种信息，那么相较于无序类别特征的编码，有哪些变化呢？

* 标签编码 -> 字典编码
* 独热编码 -> ~~很少不用~~
* 计数编码、Target 编码、WOE 编码、人工编码使用方式不变

# 标签编码

> Label Encoder 标签编码

无序的类别变量，在很多时候是以字符串形式的出现的，例如：颜色：红色，绿色，黑色...；
形状：三角形，正方形，圆形...。梯度提升树模型是无法对此类特征进行处理的，直接将其输入到模型就会报错

标签编码就是简单地赋予不同类别以不同的数字标签，属于硬编码。优点是简单直白，
网上很多说适用于有序类别型特征，不过如果是分类任务且类别不多的情况下，
LightGBM 只要指定 `categorical_feature` 也能有较好的表现。
但不建议用在高基类特征上，而且标签编码后的自然数对于回归任务来说是线性不可分的

另外 Label Binarizer 能用来从多类别列表创建标签矩阵，
它将一个类别列表转换成一个列数与输入集合中唯一值的列数完全相同的矩阵

## LabelEncoder

Python 中提供了 `sklearn.preprocessing.LabelEncoder` 编码方法，
`LabelEncoder` 可以将类型为 `object` 的变量转变为数值形式。
`LabelEncoder` 默认会先将 `object` 类型的变量进行排序，
然后按照大小顺序进行 `$0, 1, 2, \ldots, N-1$` 的编码，
此处 `$N$` 为该特征中不同变量的个数

```python
from sklearn import preprocessing

# data
df = pd.DataFrame({
    "color": ["red", "blue", "black", "green"]
})

# 编码
le = preprocessing.LabelEncoder()
df["color_labelencode"] = le.fit_transform(df["color"].values)

df
```

```
    color	color_labelencode
0	red	    3
1	blue	1
2	black	0
3	green	2
```

## LabelBinarizer

```python
from sklearn import preporcessing

# data
df = pd.DataFrame({
    "color": ["red", "blue", "black", "green"]
})

# 编码
lb = preprocessing.LabelBinarizer()
df["color_labelbinarizer"] = lb.fit_transform(df["color"].values)
```

# 哈希编码

> * Hash Encoder 哈希编码
> * Binary Encoder，二进制编码

哈希编码是使用二进制对标签编码做哈希映射。好处在于哈希编码器不需要维持类别字典，
若后续出现训练集未出现的类别，哈希编码也能适用。但按位分开哈希编码，模型学习相对比较困难

> 二进制编码主要分为两步，先用序号编码给每个类别赋予一个类别 ID， 
  然后将类别 ID 对应的二进制编码作为结果 
> 二进制编码本质上是利用二进制对 ID 进行哈希映射，最终得到 0/1 特征向量， 
  且维数少于 One-Hot Encoding, 节省了存储空间

## HashingEncoder

```python
# !pip install category_encoders

import category_encoders as ce

# data
df = pd.DataFrame({
    'gender': [2, 1, 1]
})

# hash encoder
ce_encoder = ce.HashingEncoder(cols = ['gender']).fit(df)
x_trans = ce_encoder.transform(df)

x_trans
```

```
col_0  col_1  col_2  col_3  col_4  col_5  col_6  col_7
0      0      0      0      0      1      0      0      0
1      0      0      0      1      0      0      0      0
2      0      0      0      1      0      0      0      0
```

# 独热编码

> One-Hot Encoding，独热编码

独热编码采用 `$d$` 位状态寄存器来对 `$d$` 个状态进行编码，
简单来说就是利用 0 和 1 表示类别状态对每个类别使用二进制编码

一个具有 `$n$` 个观测值和 `$d$` 个不同值的单一类别变量被转换成具有 `$n$` 个观测值的 `$d$` 的二元变量，
每个二元变量使用 `$(0, 1)$` 进行表示，它转换后的变量叫哑变量（dummy variables），
其中 1 代表某个输入属于该类别，以表示特定行是否属于该类别

独热编码能很好解决标签编码对于回归任务中线性不可分的问题

* 缺点：它处理不好高基数特征，基类越大会带来过很多列的稀疏特征，消耗内存和训练时间
* 优点：
    - 独热编码能很好解决标签编码对于回归任务中线性不可分的问题
    - 很多时候对高基数的类别特征直接进行 One-Hot 编码的效果往往可能不如直接 `LabelEncoder` 来的好。
      但是当我们的类别变量中有一些变量是人为构造的，加入了很多噪音，这个时候将其展开，
      那么模型可以更加快的找到那些非构建的类别

## get_dummies

```python
from sklearn import preprocessing

df = pd.DataFrame({
    "color": ["red", "blue", "black", "green"]
})

# 编码
x_dummies = pd.get_dummies(
    columns = df["color"].values, 
    data = df
)
```

# 计数编码

> * Count Encoder，计数编码
> * Frequency Encoder，频次编码

计数编码通过计算特征变量中每个值的出现次数来表示该特征的信息。
频度统计对于低频具有归一化作用，能够使类别特征中低频的特征数据的共性被挖掘出来

计数编码也叫频次编码。就是用分类特征下不同类别的样本数去编码类别。
清晰地反映了类别在数据集中的出现次数，缺点是忽略类别的物理意义，
比如说两个类别出现频次相当，但是在业务意义上，模型的重要性也许不一样

计数编码是数据竞赛中使用最为广泛的技术，在 90% 以上的数据建模的问题中都可以带来提升。
因为在很多的时候，频率的信息与我们的目标变量往往存在有一定关联，例如：

* 在音乐推荐问题中，对于乐曲进行计数编码可以反映该乐曲的热度，而热度高的乐曲往往更受大家的欢迎
* 在购物推荐问题中，对于商品进行计数编码可以反映该商品的热度，而热度高的商品大家也更乐于购买
* 微软设备被攻击概率问题中，预测设备受攻击的概率，那么设备安装的软件是非常重要的信息，
  此时安装软件的计数编码可以反映该软件的流行度，越流行的产品的受众越多，
  那么黑客往往会倾向对此类产品进行攻击，这样黑客往往可以获得更多的利益

## CountEncoder

示例 1:

```python
import pandas as pd

# data
df = pd.DataFrame({
    '区域' : ['西安', '太原', '西安', '太原', '郑州', '太原'], 
    '10月份销售' : [
        '0.477468', '0.195046', '0.015964', 
        '0.259654', '0.856412', '0.259644'
    ],
    '9月份销售' : [
        '0.347705', '0.151220', '0.895599', 
        '0236547', '0.569841', '0.254784'
    ]
})

# 统计
df_counts = df['区域'].value_counts().reset_index()
df_counts.columns = ['区域', '区域频度统计']
print(df_count)
```

```
   区域  区域频度统计
0  太原       3
1  西安       2
2  郑州       1
```

```python
df = df.merge(df_counts, on = ['区域'], how = 'left')
print(df)
```

```
   区域    10月份销售     9月份销售  区域频度统计
0  西安  0.477468  0.347705       2
1  太原  0.195046  0.151220       3
2  西安  0.015964  0.895599       2
3  太原  0.259654   0236547       3
4  郑州  0.856412  0.569841       1
5  太原  0.259644  0.254784       3
```

示例 2：

```python
df = pd.DataFrame({
    'color': [
        'red', 'red', 'red', 'blue', 'blue', 
        'black', 'green', 'green', 'green'
    ]
})
df['color_cnt'] = df['color'].map(df['color'].value_counts())
df
```

```
    color	color_cnt
0   red	    3
1   red	    3
2   red	    3
3   blue    2
4   blue    2
5   black   1
6   green   3
7   green   3
8   green   3
```

示例 3：

```python
import pandas as pd
import category_encoders as ce

# data
df = pd.DataFrame({
    "cat_feat": ["A", "B", "B", "A", "B", "A"]
})

# encoder
count_encoder = ce.count.CountEncoder(cols = ["cat_feat"]).fit(df)
df_trans = count_encoder.transform(df)
df_trans
```

# 直方图编码

> Bin Encoder：直方图编码

直方图编码属于目标编码的一种，适用于分类任务。它先将类别属性分类，然后在对应属性下，
统计不同类别标签的样本占比进行编码

优点：

* 直方图编码能清晰看出特征下不同类别对不同预测标签的贡献度

缺点：

* 使用了标签数据，若训练集和测试集的类别特征分布不一致，那么编码结果容易引发过拟合
* 直方图编码出的特征数量是分类标签的类别数量，若标签类别很多，可能会给训练带来空间和时间上的负担

直方图编码样例如下图所示：

![img](images/hist.png)

## HistEncoder

```python
import pandas as pd

class HistEncoder:
    """
    直方图码器

    params:
        df         (pd.DataFrame): 待编码的 dataframe 数据
        encode_feat_name    (str): 编码的类别特征名，当前代码只支持单个特征编码，若要批量编码，请自行实现
        label_name          (str): 类别标签
    """

    def __init__(self, df, encode_feat_name, label_name):
        self.df = df.copy()
        self.encode_feat_name = encode_feat_name
        self.label_name = label_name

    def fit(self):
        '''
        用训练集获取编码字典
        '''
        # 分子：类别特征下给定类别，在不同分类标签下各类别的数量
        self.df['numerator'] = 1
        numerator_df = self.df.groupby([self.encode_feat_name, self.label_name])['numerator'].count().reset_index()

        # 分母：分类标签下各类别的数量
        self.df['denumerator'] = 1
        denumerator_df = self.df.groupby(self.encode_feat_name)['denumerator'].count().reset_index()

        # 类别特征类别、分类标签类别：直方图编码映射字典
        encoder_df = pd.merge(numerator_df, denumerator_df, on = self.encode_feat_name)
        encoder_df['encode'] = encoder_df['numerator'] / encoder_df['denumerator'] 

        self.encoder_df = encoder_df[[self.encode_feat_name, self.label_name, 'encode']]

    def transform(self, test_df):
        '''对测试集编码'''
        # 依次编码出: hist特征1， hist特征2， ...
        test_trans_df = test_df.copy()
        for label_cat in test_trans_df[self.label_name].unique():
            hist_feat = []
            for cat_feat_val in test_trans_df[self.encode_feat_name].values:
                try:
                    encode_val = encoder_df[
                        (encoder_df[self.label_name] == label_cat) & (encoder_df[self.encode_feat_name] == cat_feat_val)
                    ]['encode'].item()
                    hist_feat.append(encode_val)
                except:
                    hist_feat.append(0)
            encode_fname = self.encode_feat_name + '_en{}'.format(str(label_cat))  # 针对类别特征-类别label_cat的直方图编码特征名
            test_trans_df[encode_fname] = hist_feat  # 将编码的特征加入到原始数据中    
        return test_trans_df
```

```python
# 初始化数据
df = pd.DataFrame({
    'cat_feat': ['A', 'A', 'B', 'A', 'B', 'A'], 
    'label': [0, 1, 0, 2, 1, 2]
})
encode_feat_name = 'cat_feat'
label_name = 'label'

# 直方图编码
he = HistEncoder(df, encode_feat_name, label_name)
he.fit()
df_trans = he.transform(df)

>>df
cat_feat  label
0  A  0
1  A  1
2  B  0
3  A  2
4  B  1
5  A  2

>>df_trans
cat_feat  label  cat_feat_en0  cat_feat_en1  cat_feat_en2
0  A  0  0.25  0.25  0.5
1  A  1  0.25  0.25  0.5
2  B  0  0.50  0.50  0.0
3  A  2  0.25  0.25  0.5
4  B  1  0.50  0.50  0.0
5  A  2  0.25  0.25  0.5
```

# WOE 编码

> 证据权重，Weight of Evidence

WOE(Weight of Evidence，证据权重)编码适用于二分类任务。
WOE 开发的主要目标是创建一个预测模型，表明自变量相对于因变量的预测能力，
是另一种关于分类自变量和因变量之间关系的方案

由于它是从信用评分领域演变而来的，用于评估信贷和金融行业的贷款违约风险。
它通常被描述为区分好客户和坏客户的衡量标准。“坏客户”是指拖欠贷款的客户，
“优质客户”指的是谁偿还贷款的客户，可以把 WoE 理解成：
每个分组内坏客户分布相对于优质客户分布之间的差异性

WoE 的数学定义是优势比的自然对数, 即: 

`$$WoE = \sum_{i}^{\#group} ln(\frac{p_{y_{i}}}{p_{n_{i}}}) = \sum_{i}^{\#group} ln(\frac{\frac{\#y_{i}}{\#y_{T}}}{\frac{\#n_{i}}{\#n_{T}}})$$`

其中：

* `$\#group$`：分组的数量
* `$p_{y_{i}}`：组内违约用户数占比，即组内 `label = 1` 的样本占比数
* `$p_{n_{i}}$`： 组内正常用户占比，即组内 `label = 0` 的样本数占比
* `$\frac{\#y_{i}}{\#y_{T}}$`：组内违规用户数/所有违规用户数
* `$\frac{\#n_{i}}{\#n_{T}}$`：组内正常用户数/所有正常用户数

在实践中，可以直接通过下面的步骤计算得到 WOE 的结果：

* 对于一个连续变量可以将数据先进行分箱，对于类别变量无需做任何操作
* 计算每个类内（group）中正样本和负样本出现的次数
* 计算每个类内（group）正样本和负样本的百分比 events% 以及 non events%
* 按照公式计算 WOE

这些方法都是有监督编码器， 或者是考虑目标变量的编码方法, 因此在预测任务中通常是更有效的编码器。
但是，当需要执行无监督分析时，这些方法并不一定适用

WoE 存在几个问题：

1. 分母可能为 0
2. 没有考虑不同类别数量的大小带来的影响，可能某类数量多，但最后计算出的 WOE 跟某样本数量少的类别的 WOE 一样
3. 只针对二分类问题
4. 训练集和测试集可能存在 WOE 编码差异（通病）

## WOEEncoder

* https://github.com/Sundar0989/WOE-and-IV/blob/master/WOE_IV.ipynb

```python
nominator = (stats["sum"] + regularization) / (_sum + 2 * regularization)
denominator = ((stats["count"] - stats["sum"]) + regularization) / (_count - _sum + 2 * regularization)
woe = np.log(nominator / denominator)
```

```python
from category_encoders import WOEEncoder

df = pd.DataFrame({
    "cat": ["a", "b", "a", "b", "a", "a", "b", "c", "c"],
    "target": [1, 0, 0, 1, 0, 0, 1, 1, 0],
})
X = df["cat"]
y = df.target

encoded_df = woe.fit_transform(X, y)
print(encoder_df)
```

## Information Value

WoE 是另一个衡量指标 Information Value 的关键组成部分，该指标用来衡量特征如何为预测提供信息 

`$$IV = \sum_{i}^{\#group}(p_{y_{i}} - p_{n_{i}})\times ln(\frac{p_{y_{i}}}{p_{n_{i}}})$$`

WOE和IV的区别和联系[2]是：

1. WOE describes the relationship between a predictive variable and a binary target variable.
2. IV measures the strength of that relationship.

扩展：IV 常会被用来评估变量的预测能力，用于筛选变量

| 信息价值 IV | 可变预测性 |
|----|----|
| 小于 0.02 | 对预测没有用 |
| 0.02 到 0.1 | 预测能力弱 |
| 0.1 到 0.3 | 中等预测能力 |
| 0.3 到 0.5 | 强大的预测能力 |
| 0.02 到 0.1 | 可疑的预测能力 |




# Target 编码

> 目标编码

Target 编码是 2006 年提出的一种结合标签进行编码的技术，它将类别特征替换为从标签衍生而来的特征，
这可以更直接地表示分类变量和目标变量之间的关系，在类别特征为高基数的时候非常有效。
该技术在非常多的数据竞赛中都取得了非常好的效果，但特别需要注意过拟合的问题。
在 kaggle 竞赛中成功的案例有 owen zhang 的 Leave-One-Out Mean-Target 编码，
和莫斯科 GM 的基于 K-fold 的 Mean-Target 编码

![img](images/target-encoding.png)

但这种编码方法也有一些缺点：

* 首先, 它使模型更难学习均值编码变量和另一个变量之间的关系, 仅基于列与目标的关系就在列中绘制相似性
* 而最主要的是，这种编码方法对目标变量非常敏感，这会影响模型提取编码信息的能力

由于该类别的每个值都被相同的数值替换，因此模型可能会过拟合其见过的编码值(例如将 0.8 与完全不同的值相关联, 而不是 0.79)，
这是把连续尺度上的值视为严重重复的类的结果。因此，需要仔细监控 y 变量，以防出现异常值。要实现这个目的，
就要使用 `category_encoders` 库。由于目标编码器是一种有监督方法，所以它同时需要 X 和 y 训练集 

## Leave-One-Out Mean-Target 编码

Leave-One-Out Mean-Target 编码的思路相对简单，每次编码时，不考虑当前样本的情况，
用其它样本对应的标签的均值作为编码，而测试集则用全部训练集样本的均值进行编码，案例如下：

![img](images/LOO_MT.png)

| Split | UserID | Y | Mean(Y)       |
|-------|--------|---|---------------|
| Train | A1     | 0 | 2 / 3 = 0.667 |
| Train | A1     | 1 | 1 / 3 = 0.333 |
| Train | A1     | 1 | 1 / 3 = 0.333 |
| Train | A1     | 0 | 2 / 3 = 0.667 |
| Test  | A1     | - | 2 / 4 = 0.5   |
| Test  | A1     | - | 2 / 4 = 0.5   |
| Test  | A2     | 0 | ...           |
| ...   | ..     | . | ...           |

```python
import pandas as pd
from sklearn import preprocessing
from category_encoders.leave_one_out import LeaveOneOutEncoder as looe

# train and test data
df_train = pd.DataFrame({
    "color": [
        "red", "red", "red", "red", 
        "red", "red", "black", "black"
    ],
    'label': [1, 0, 1, 1, 0, 1, 1, 0]
})
df_test = pd.DataFrame({
    "color": ["red", "red", "black"]
})

# leave one out mean-target encoder
loo = LeaveOneOutEncoder()
loo.fit_transform(df_train["color"], df_train["label"])
loo.transform(df_test["color"])
```

```
    color
0	0.6
1	0.8
2	0.6
3	0.6
4	0.8
5	0.6
6	0.0
7	1.0


    color
0	0.666667
1	0.666667
2	0.500000
```

```python
import pandas as pd
import category_encoders as ce

# data
data = [
    ['1', 120],
    ['2', 120],
    ['3', 140],
    ['2', 100], 
    ['3', 70], 
    ['1', 100],
    ['2', 60],
    ['3', 110], 
    ['1', 100],
    ['3', 70]
]
df = pd.DataFrame(data, columns = ["Dept", "Yearly Salary"])

# 编码
tenc = ce.TargetEncoder()
df_dep = tenc.fit_transform(df["Dept"], df["Year Salary"])
df_dep = df_dep.rename({"Dept": "Value"}, axis = 1)
df_new = df.join(df_dep)
```

## K-fold Mean-Target 编码

K-fold Mean-Target 编码的基本思想来源于 Mean Target 编码。
K-fold Mean-Target 编码的训练步骤如下：

1. 先将训练集划分为 K 折
2. 在对第 A 折的样本进行编码时，删除 K 折中 A 折，并用剩余的数据计算如下公式

`$$Mean_A = mean(Y)$$`

3. 然后利用上面计算得到的值对第 A 折进行编码
4. 最后，依次对所有折进行编码即可

首先我们先理解一下上面的公式，最原始的 Mean-target 编码是非常容易导致过拟合的，
这其中过拟合的最大的原因之一在于对于一些特征列中出现次数很少的值过拟合了，
比如某些值只有 1 个或者 2 到 3 个，但是这些样本对应的标签全部是 1，怎么办，
他们的编码值就应该是 1，但是很明显这些值的统计意义不大，大家可以通过伯努利分布去计算概率来理解。
而如果我们直接给他们编码了，就会误导模型的学习。那么我们该怎么办呢？老办法，加正则！

于是我们就有了上面的计算式子，式子是值出现的次数，是它对应的概率，是全局的均值，
那么当为 0 同时比较小的时候， 就会有大概率出现过拟合的现象，此时我们调大就可以缓解这一点，
所以很多时候都需要不断地去调整的值

```python
from sklearn import base
from category_encoders.target_encoder import TargetEncoder
from sklearn.model_selection import KFold

# data
df = pd.DataFrame({
    'Feature': [
        'A', 'B', 'B', 'B', 'B', 'A', 'B', 
        'A', 'A', 'B', 'A', 'A', 'B', 'A',
        'A', 'B', 'B', 'B', 'A', 'A'],
    'Target': [
        1, 0, 0, 1, 1, 1, 0, 
        0, 0, 0, 1, 0, 1, 0, 
        1, 0, 0, 0, 1, 1
    ]
})


class KFoldTargetEncoderTrain(base.BaseEstimator, base.TransformerMixin):

    def __init__(self, colnames, targetName, n_fold = 5, verbosity = True, discardOriginal_col = False):
        self.colnames = colnames
        self.targetName = targetName
        self.n_fold = n_fold
        self.verbosity = verbosity
        self.discardOriginal_col = discardOriginal_col

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        assert(type(self.targetName) == str)
        assert(type(self.colnames) == str)
        assert(self.colnames in X.columns)
        assert(self.targetName in X.columns)

        mean_of_target = X[self.targetName].mean()
        kf = KFold(n_splits = self.n_fold, shuffle = False, random_state = 2019)

        col_mean_name = self.colnames + '_' + 'Kfold_Target_Enc'
        X[col_mean_name] = np.nan

        for tr_ind, val_ind in kf.split(X):
            X_tr, X_val = X.iloc[tr_ind], X.iloc[val_ind] 
            X.loc[X.index[val_ind], col_mean_name] = X_val[self.colnames].map(
                X_tr.groupby(self.colnames)[self.targetName].mean()
            )

        X[col_mean_name].fillna(mean_of_target, inplace = True)

        if self.verbosity:
            encoded_feature = X[col_mean_name].values
            print('Correlation between the new feature, {} and, {} is {}.'.format(
                col_mean_name,
                self.targetName,
                np.corrcoef(X[self.targetName].values, encoded_feature)[0][1]
            ))

        if self.discardOriginal_col:
            X = X.drop(self.targetName, axis = 1)
            
        return X
    

class KFoldTargetEncoderTest(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self, train, colNames, encodedName):
        self.train = train
        self.colNames = colNames
        self.encodedName = encodedName
        
    def fit(self, X, y = None):
        return self

    def transform(self, X):
        mean = self.train[[self.colNames,self.encodedName]].groupby(self.colNames).mean().reset_index() 
        dd = {}
        for index, row in mean.iterrows():
            dd[row[self.colNames]] = row[self.encodedName]
        
        X[self.encodedName] = X[self.colNames]
        X = X.replace({self.encodedName: dd})

        return X
```

训练集编码：

```python
targetc   = KFoldTargetEncoderTrain('Feature','Target',n_fold=5)
new_train = targetc.fit_transform(df)
new_train
```

```
    Feature	Target	Feature_Kfold_Target_Enc
0   A       1       0.555556
1   B       0       0.285714
2   B       0       0.285714
3   B       1       0.285714
4   B       1       0.250000
5   A       1       0.625000
6   B       0       0.250000
7   A       0       0.625000
8   A       0       0.714286
9   B       0       0.333333
10  A       1       0.714286
11  A       0       0.714286
12  B       1       0.250000
13  A       0       0.625000
14  A       1       0.625000
15  B       0       0.250000
16  B       0       0.375000
17  B       0       0.375000
18  A       1       0.500000
19  A       1       0.500000
```

测试集编码：

```python
test_targetc = KFoldTargetEncoderTest(
    new_train, 
    'Feature', 
    'Feature_Kfold_Target_Enc'
)
new_test = test_targetc.fit_transform(test)
```

## Bayesian Target 编码

> 贝叶斯目标编码 Bayesian Target Encoding

贝叶斯目标编码(Bayesian Target Encoding)是一种使用目标作为编码方法的数学方法. 
仅使用均值可能是一种欺骗性度量标准, 因此贝叶斯目标编码试图结合目标变量分布的其他统计度量. 
例如其方差或偏度(称为高阶矩「higher moments」). 然后通过贝叶斯模型合并这些分布的属性, 
从而产生一种编码, 该编码更清楚类别目标分布的各个方面, 但是结果的可解释性比较差

## Beta Target 编码

> Beta 目标编码，Beta Target Encoding

Beta Target Encoding 可以提取更多的特征，不仅仅是均值，还可以是方差等。
没有进行 N Fold 提取特征，所以可能在时间上提取会更快一些。
另外使用 Beta Target Encoding 相较于直接使用 LightGBM 建模的效果可以得到大幅提升

利用 Beta 分布作为共轭先验，可以方便地对二元目标变量进行建模。
Beta 分布用 `$\alpha$` 和 `$\beta$` 来参数化，
`$\alpha$` 和 `$\beta$` 可以被当作是重复 Binomial 实验中的正例数和负例数。
Beta 分布中许多有用的统计数据可以用 `$\alpha$` 和 `$\beta$` 表示，例如：

平均值：

`$$\mu = \frac{\alpha}{\alpha + \beta}$$`

方差：

`$$\sigma^{2} = \frac{\alpha\beta}{(\alpha + \beta)^{2}(\alpha + \beta + 1)}$$`

因为 Beta Target Encoding 是类别编码的一种，所以适用于高基数类别特征的问题，
是可以转成 N-fold 的形式的，所以非常建议在使用时考虑转成 N-fold 提取的形式，
是非常值得尝试的方案之一

### BetaEncoder

```python
import numpy as np 
import pandas as pd
from sklearn.preprocessing import LabelEncoder

'''
代码摘自原作者：https://www.kaggle.com/mmotoki/beta-target-encoding
'''
class BetaEncoder(object):
        
    def __init__(self, group):
        
        self.group = group
        self.stats = None
        
    # get counts from df
    def fit(self, df, target_col):
        # 先验均值
        self.prior_mean = np.mean(df[target_col]) 
        stats           = df[[target_col, self.group]].groupby(self.group)
        # count和sum
        stats           = stats.agg(['sum', 'count'])[target_col]    
        stats.rename(columns={'sum': 'n', 'count': 'N'}, inplace=True)
        stats.reset_index(level=0, inplace=True)           
        self.stats      = stats
        
    # extract posterior statistics
    def transform(self, df, stat_type, N_min=1):
        
        df_stats = pd.merge(df[[self.group]], self.stats, how='left')
        n        = df_stats['n'].copy()
        N        = df_stats['N'].copy()
        
        # fill in missing
        nan_indexs    = np.isnan(n)
        n[nan_indexs] = self.prior_mean
        N[nan_indexs] = 1.0
        
        # prior parameters
        N_prior     = np.maximum(N_min-N, 0)
        alpha_prior = self.prior_mean*N_prior
        beta_prior  = (1-self.prior_mean)*N_prior
        
        # posterior parameters
        alpha       =  alpha_prior + n
        beta        =  beta_prior  + N-n
        
        # calculate statistics
        if stat_type=='mean':
            num = alpha
            dem = alpha+beta
                    
        elif stat_type=='mode':
            num = alpha-1
            dem = alpha+beta-2
            
        elif stat_type=='median':
            num = alpha-1/3
            dem = alpha+beta-2/3
        
        elif stat_type=='var':
            num = alpha*beta
            dem = (alpha+beta)**2*(alpha+beta+1)
                    
        elif stat_type=='skewness':
            num = 2*(beta-alpha)*np.sqrt(alpha+beta+1)
            dem = (alpha+beta+2)*np.sqrt(alpha*beta)

        elif stat_type=='kurtosis':
            num = 6*(alpha-beta)**2*(alpha+beta+1) - alpha*beta*(alpha+beta+2)
            dem = alpha*beta*(alpha+beta+2)*(alpha+beta+3)
            
        # replace missing
        value = num/dem
        value[np.isnan(value)] = np.nanmedian(value)
        return value
N_min = 1000
feature_cols = []    

# encode variables
for c in cat_cols:

    # fit encoder
    be = BetaEncoder(c)
    be.fit(train, 'deal_probability')

    # mean
    feature_name = f'{c}_mean'
    train[feature_name] = be.transform(train, 'mean', N_min)
    test[feature_name]  = be.transform(test,  'mean', N_min)
    feature_cols.append(feature_name)

    # mode
    feature_name = f'{c}_mode'
    train[feature_name] = be.transform(train, 'mode', N_min)
    test[feature_name]  = be.transform(test,  'mode', N_min)
    feature_cols.append(feature_name)
    
    # median
    feature_name = f'{c}_median'
    train[feature_name] = be.transform(train, 'median', N_min)
    test[feature_name]  = be.transform(test,  'median', N_min)
    feature_cols.append(feature_name)    

    # var
    feature_name = f'{c}_var'
    train[feature_name] = be.transform(train, 'var', N_min)
    test[feature_name]  = be.transform(test,  'var', N_min)
    feature_cols.append(feature_name)        
    
    # skewness
    feature_name = f'{c}_skewness'
    train[feature_name] = be.transform(train, 'skewness', N_min)
    test[feature_name]  = be.transform(test,  'skewness', N_min)
    feature_cols.append(feature_name)    
    
    # kurtosis
    feature_name = f'{c}_kurtosis'
    train[feature_name] = be.transform(train, 'kurtosis', N_min)
    test[feature_name]  = be.transform(test,  'kurtosis', N_min)
    feature_cols.append(feature_name)  
```

# 平均编码

> 平均编码，Mean Encoder

平均编码是基于目标编码的改进版。它的改动如下：

1. 权重公式：其实没有本质上的区别，可自行修改函数内的参数
2. 由于目标编码使用了标签，为了缓解编码带来模型过拟合问题，平均编码加入了 K-fold 编码思路，
   若分为 5 折，则用 1-4 折先 fit 后，再 transform 第 5 折，依次类推，将类别特征分 5 次编码出来。
   坏处是耗时

## MeanEncoder

```python
class MeanEncoder:

    def __init__(self, 
                 categorical_features, 
                 n_splits = 5, 
                 target_type = 'classification', 
                 prior_weight_func = None):
        """
        Param:
            categorical_features: list of str, the name of the categorical columns to encode
            n_splits: the number of splits used in mean encoding
            target_type: str, 'regression' or 'classification'
            prior_weight_func:
        a function that takes in the number of observations, and outputs prior weight
        when a dict is passed, the default exponential decay function will be used:
           k: the number of observations needed for the posterior to be weighted equally as the prior
           f: larger f -> smaller slope
        """
        self.categorical_features = categorical_features
        self.n_splits = n_splits
        self.learned_stats = {}

        if target_type == 'classification':
            self.target_type = target_type
            self.target_values = []
        else:
            self.target_type = 'regression'
            self.target_values = None

        if isinstance(prior_weight_func, dict):
            self.prior_weight_func = eval('lambda x: 1 / (1 + np.exp((x - k) / f))', dict(prior_weight_func, np=np))
        elif callable(prior_weight_func):
            self.prior_weight_func = prior_weight_func
        else:
            self.prior_weight_func = lambda x: 1 / (1 + np.exp((x - 2) / 1))

    @staticmethod
    def mean_encode_subroutine(X_train, y_train, X_test, variable, target, prior_weight_func):
        X_train = X_train[[variable]].copy()
        X_test = X_test[[variable]].copy()

        if target is not None:
            nf_name = '{}_pred_{}'.format(variable, target)
            X_train['pred_temp'] = (y_train == target).astype(int)  # classification
        else:
            nf_name = '{}_pred'.format(variable)
            X_train['pred_temp'] = y_train  # regression
        prior = X_train['pred_temp'].mean()

        col_avg_y = X_train.groupby(by=variable, axis=0)['pred_temp'].agg([('mean', 'mean'), ('beta', 'size')])
        col_avg_y['beta'] = prior_weight_func(col_avg_y['beta'])
        col_avg_y[nf_name] = col_avg_y['beta'] * prior + (1 - col_avg_y['beta']) * col_avg_y['mean']
        col_avg_y.drop(['beta', 'mean'], axis=1, inplace=True)

        nf_train = X_train.join(col_avg_y, on=variable)[nf_name].values
        nf_test = X_test.join(col_avg_y, on=variable).fillna(prior, inplace=False)[nf_name].values

        return nf_train, nf_test, prior, col_avg_y

    def fit_transform(self, X, y):
        """
        :param X: pandas DataFrame, n_samples * n_features
        :param y: pandas Series or numpy array, n_samples
        :return X_new: the transformed pandas DataFrame containing mean-encoded categorical features
        """
        X_new = X.copy()
        if self.target_type == 'classification':
            skf = StratifiedKFold(self.n_splits)
        else:
            skf = KFold(self.n_splits)

        if self.target_type == 'classification':
            self.target_values = sorted(set(y))
            self.learned_stats = {'{}_pred_{}'.format(variable, target): [] for variable, target in
                                  product(self.categorical_features, self.target_values)}
            for variable, target in product(self.categorical_features, self.target_values):
                nf_name = '{}_pred_{}'.format(variable, target)
                X_new.loc[:, nf_name] = np.nan
                for large_ind, small_ind in skf.split(y, y):
                    nf_large, nf_small, prior, col_avg_y = MeanEncoder.mean_encode_subroutine(
                        X_new.iloc[large_ind], y.iloc[large_ind], X_new.iloc[small_ind], variable, target, self.prior_weight_func)
                    X_new.iloc[small_ind, -1] = nf_small
                    self.learned_stats[nf_name].append((prior, col_avg_y))
        else:
            self.learned_stats = {'{}_pred'.format(variable): [] for variable in self.categorical_features}
            for variable in self.categorical_features:
                nf_name = '{}_pred'.format(variable)
                X_new.loc[:, nf_name] = np.nan
                for large_ind, small_ind in skf.split(y, y):
                    nf_large, nf_small, prior, col_avg_y = MeanEncoder.mean_encode_subroutine(
                        X_new.iloc[large_ind], y.iloc[large_ind], X_new.iloc[small_ind], variable, None, self.prior_weight_func)
                    X_new.iloc[small_ind, -1] = nf_small
                    self.learned_stats[nf_name].append((prior, col_avg_y))
        return X_new

    def transform(self, X):
        """
        :param X: pandas DataFrame, n_samples * n_features
        :return X_new: the transformed pandas DataFrame containing mean-encoded categorical features
        """
        X_new = X.copy()

        if self.target_type == 'classification':
            for variable, target in product(self.categorical_features, self.target_values):
                nf_name = '{}_pred_{}'.format(variable, target)
                X_new[nf_name] = 0
                for prior, col_avg_y in self.learned_stats[nf_name]:
                    X_new[nf_name] += X_new[[variable]].join(col_avg_y, on=variable).fillna(prior, inplace=False)[
                        nf_name]
                X_new[nf_name] /= self.n_splits
        else:
            for variable in self.categorical_features:
                nf_name = '{}_pred'.format(variable)
                X_new[nf_name] = 0
                for prior, col_avg_y in self.learned_stats[nf_name]:
                    X_new[nf_name] += X_new[[variable]].join(col_avg_y, on=variable).fillna(prior, inplace=False)[
                        nf_name]
                X_new[nf_name] /= self.n_splits

        return X_new
```

# 模型编码

目前 GBDT 模型中，只有 LightGBM 和 CatBoost 自带类别编码

## LightGBM GS 编码

LightGBM 的类别编码采用的是 GS(Gradient Statistics)编码。
主要思路是将类别特征转为累积值 `$\frac{sum(gradient)}{hessian}$` (一阶偏导数之和/二阶偏导数之和)再进行直方图特征排序。
使用起来也很简单，定义 LightGBM 数据集时，指定 `categorical_feature` 即可

据官方文档介绍，LightGBM 的 GS 编码比独热编码快大概 8 倍速度。而且文档里也建议，
当类别变量为高基类时，哪怕是简单忽略类别含义或把它嵌入到低维数值空间里，
只要将特征转为数值型，一般会表现的比较好。就个人使用来讲，我一般会对无序类别型变量进行模型编码，
有序类别型变量直接按顺序标签编码即可

虽然 LightGBM 用 GS 编码类别特征看起来挺厉害的，但是存在两个问题：

* 计算时间长：因为每轮都要为每个类别值进行 GS 计算
* 内存消耗大：对于每次分裂，都存储给定类别特征下，它不同样本划分到不同叶节点的索引信息

## CatBoost Ordered TS 编码

CatBoost 使用 Ordered TS 编码，既利用了 TS 省空间和速度的优势，
也使用 Ordered 的方式缓解预测偏移问题

CatBoost 编码器视图解决的是目标泄漏问题，除了目标编码外，还使用了一个排序概念。
它的工作原理与时间序列数据验证类似，当前特征的目标概率仅从它之前的行(观测值)计算，
这意味着目标统计值依赖于观测历史

对于可取值的数量比独热最大量还要大的分类变量，CatBoost 使用了一个非常有效的编码方法，
这种方法和均值编码类似，但可以降低过拟合情况。它的具体实现方法如下：

1. 将输入样本集随机排序，并生成多组随机排列的情况
2. 将浮点型或属性值标记转化为整数
3. 将所有的分类特征值结果都根据以下公式，转化为数值结果

其中 CountInClass 表示在当前分类特征值中，有多少样本的标记值是1；
Prior 是分子的初始值，根据初始参数确定。TotalCount 是在所有样本中（包含当前样本），
和当前样本具有相同的分类特征值的样本数量

![img](images/ts_encoder.png)

CatBoost 处理 Categorical features 总结：

* 首先，他们会计算一些数据的 statistics。计算某个 category 出现的频率，加上超参数，
  生成新的 numerical features。这一策略要求同一标签数据不能排列在一起（即先全是 0 之后全是 1 这种方式），
  训练之前需要打乱数据集
* 第二，使用数据的不同排列（实际上是4个）。在每一轮建立树之前，先扔一轮骰子，决定使用哪个排列来生成树
* 第三，考虑使用categorical features的不同组合。例如颜色和种类组合起来，
  可以构成类似于blue dog这样的feature。当需要组合的categorical features变多时，
  catboost只考虑一部分combinations。在选择第一个节点时，只考虑选择一个feature，例如 A。
  在生成第二个节点时，考虑A和任意一个 categorical feature 的组合，选择其中最好的。
  就这样使用贪心算法生成 combinations
* 第四，除非向gender这种维数很小的情况，不建议自己生成one-hot vectors，最好交给算法来处理。

### CatBoostEncoder

```python
from category_encoders.cat_boost import CatBoostEncoder

cbe = CatBoostEncoder(
    verbose = 0, 
    cols = None, 
    drop_invariant = False, 
    return_df = True, 
    handle_unknown = "value", 
    handle_missing = "value",
    random_state = None,
    sigma = None,
    a = 1
)

target = df["target"]
train = df.drop("target", axis = 1)

# 编码
cbe = CatBoostEncoder()
cbe.fit(train, target)
train_cbe = cbe.transform(train)
```

# M Estimator Encoding

Target Encoder的一个更直接的变体是M Estimator Encoding。它只包含一个超参数 m，
它代表正则化幂。m 值越大收缩越强。建议 m 的取值范围为 1 ~ 100

# James Stein Encoding

James-Stein 为特征值提供以下加权平均值：

* 观察到的特征值的平均目标值
* 平均期望值（与特征值无关）

James-Stein 编码器将平均值缩小到全局的平均值。该编码器是基于目标的。
但是James-Stein 估计器有缺点：它只支持正态分布。

它只能在给定正态分布的情况下定义（实时情况并非如此）。为了防止这种情况，
我们可以使用 beta 分布或使用对数-比值比转换二元目标，
就像在 WOE 编码器中所做的那样，默认使用它，因为它很简单

# 有序字典编码

有序字典编码就是将特征中的每个类别按照相对大小构架字典，再进行转化。
无序类别特征的编码打乱了原始的内在顺序关系，可能增大梯度提升树模型训练的难度，
而有序字典编码的方式则最大程度的保留了所有的信息

## map

```python
import pandas as pd
from sklearn import preprocessing

# data
df = pd.DataFrame({
    "ratings": ["high", "medium", "low"]
})

# LabelEncoder 编码
le = preprocessing.LabelEncoder()
le.fit(df["ratings"].values)
df["traditional_encode"] = le.transform(df["ratings"].values)

# ordered dict 编码
ratings_dict = {
    "low": 0,
    "medium": 1,
    "high": 2,
}
df["selfdefined_encode"] = df["ratings"].map(ratings_dict).values
print(df)
```

```
    ratings	traditional_encode	selfdefined_encode
0   high    0                   2
1   medium  2                   1
2   low     1                   0
```

# 分段编码

分段聚类编码也是一种分箱的策略，它主要基于数据的相对大小并结合业务背景知识对类别特征进行分段分组重新编码。
举个简单但例子，我们现在需要预测学生的幸福指数，现在有一个类别特征：

* 学籍特征：小学一年级，小学二年级，小学三年级，小学四年级，小学五年级，小学六年级，初中一年级，
  初中二年级，初中三年级，高中一年级，高中二年级，高中三年级，大学一年级，大学二年级，大学三年级，大学四年级

我们发现学籍特征是存在相对顺序的，也就是我们的有序类别特征；
与此同时，我们知道，小学初中高中大学这几个阶段幸福的阶段都不一样，
比如小学可能是小学壹年级最不开心，因为刚刚从幼儿园到一年级不适应造成；
而小学初中高中大学在最后一个学年都会很不开心，因为那个时候压力最大，
面临着人生的重要转折。所以这个时候，我们需要对特征进行分段编码，
将学籍编码为小学，初中，高中，大学。还可以将各个不同阶段按照年级的大小进行分段，
分为该阶段的高年级生，低年级生和中间年级的学生

通过对学籍的转化，梯度提升树模型往往可以得到更好的效果。
但这种特征很多时候需要有一定的业务背景才能挖掘到，
不过有很多厉害但朋友也可以通过数据探索分析发现这种规律

分段编码这种方式在数据竞赛中还是非常常见的，例如我们可以：

* 将 24 小时分别编码为：上午，下午，晚上
* 将每个月分为月初，月中，月末等等

基于转化之后的特征再与其它特征进行组合特征往往还能获得更多的提升

## Python Code

```python
import pandas as pd
from sklearn import preprocessing

# data
df = pd.DataFrame({
    'student_status':[
        '小学一年级', '小学二年级', '小学三年级', '小学四年级', '小学五年级', '小学六年级',\
        '初中一年级', '初中二年级', '初中三年级', '高中一年级', '高中二年级', '高中三年级',\
        '大学一年级', '大学二年级', '大学三年级', '大学四年级'
    ]
})


# LabelEncoder
le = preprocessing.LabelEncoder()
le.fit(df['student_status'].values)
df['traditional_encode'] = le.transform(df['student_status'].values) 
```

```python
# 字典自定义编码
map_dic = [
    '小学一年级', '小学二年级', '小学三年级', '小学四年级', '小学五年级', '小学六年级',\
    '初中一年级', '初中二年级', '初中三年级', '高中一年级', '高中二年级', '高中三年级',\
    '大学一年级', '大学二年级', '大学三年级', '大学四年级'
]
map_dic = {
    v: i for i, v in enumerate(map_dic)
}
df['selfdefined_encode'] = df['student_status'].map(map_dic).values
df
```

```python
# 字典自定义编码
df['student_status_1st'] = df['student_status'].map(lambda x: x[:2])
map_dic = {
    '小学': 0, 
    '初中': 1, 
    '高中': 2, 
    '大学': 3
}
df['student_status_1st'] = df['student_status_1st'].map(map_dic).values
df
```

```python
# 字典自定义编码
map_dic = {
    '小学一年级': 0, 
    '小学二年级': 0, 
    '小学三年级': 1, 
    '小学四年级': 1,
    '小学五年级': 2,
    '小学六年级': 2,
    '初中一年级': 0,
    '初中二年级': 1,
    '初中三年级': 2,
    '高中一年级': 0,
    '高中二年级': 1,
    '高中三年级': 2,
    '大学一年级': 0,
    '大学二年级': 1,
    '大学三年级': 1,
    '大学四年级': 2
}
df['student_status_2nd'] = df['student_status'].map(map_dic).values
df
```

# Hashing

散列函数是一种确定性函数, 它可以将一个可能无界的整数映射到一个有限的整数范围 `$\[1, m\]$` 中, 
因为输入域可能大于输出范围, 所以可能有多个值被映射为同样的输出, 这称为碰撞

均匀散列函数可以确保将大致相同数量的数值映射到 `$m$` 个分箱中

* 如果模型中涉及特征向量和系数的内积运算, 那么就可以使用特征散列化
* 特征散列化的一个缺点是散列后的特征失去了可解释性, 只是初始特征的某种聚合

当使用哈希函数时，字符串将被转换为一个惟一的哈希值。
因为它使用的内存很少可以处理更多的分类数据。
对于管理机器学习中的稀疏高维特征，特征哈希是一种有效的方法。
它适用于在线学习场景，具有快速、简单、高效、快速的特点

## FeatureHasher

```python
from sklearn.feature_extraction import FeatureHasher

# data
df = None

# 编码
# n_features contains the number of bits you want in your hash value
h = FeatureHasher(n_features = 3, input_type = "string")

hashed_feature = h.fit_transform(df["nom_0"]).toarray()
new_df = pd.concat([df, pd.DataFrame(hashed_feature)], axis = 1)
new_df.head()
```

# Helmert Contrast

> Helmert Encoding

Hermert Encoding 将一个级别的因变量的平均值与该编码中所有先前水平的因变量的平均值进行比较

反向 Hermert Encoding 是类别编码器中变体的另一个名称，
它将因变量的特定水平平均值与其所有先前水平的平均值进行比较

## HelmertEncoder

```python
import category_encoders as ce

encoder = ce.HelmertEncoder(cols = "Dept")
new_df = encoder.fit_transform(df["Detp"])
new_df = pd.concat([df, new_df], axis = 1)
print(new_df)
```

# Sum Encoding

Sum Encoder 将类别列的特定级别的因变量(目标)的平均值与目标的总体平均值进行比较。
在线性回归(LR)的模型中，Sum Encoder 和 One Hot Encoding 都是常用的方法。
两种模型对 LR 系数的解释是不同的

* Sum Encoder 模型的截距代表了总体平均值(在所有条件下)，而系数很容易被理解为主要效应
* 在 One Hot Encoding 模型中，截距代表基线条件的平均值，系数代表简单效应(一个特定条件与基线之间的差)

# 人工编码

## 人工转化编码

这个需要一些专业背景知识，可以认为是 Label 编码的一种补充，如果我们的类别特征是字符串类型的，例如：

* 城市编号：'10','100','90','888'...

这个时候，我们使用 Labelencoder 会依据字符串排序编码。
在字符串中 '90' > '100'，但我们直观感觉是为 '100' > '90'，
所以需要人为但进行干预编码，如果都是可以直接转化为数值形的，
编码时可以直接转化为数值，或者自己书写一个字典进行映射

## 人工组合编码

这个同样的也设计到部分专业背景知识，有些问题会出现一些脏乱的数据，例如：

* 在一些位置字段中，有的是中文的，有的是英文的，例如 “ShangHai”、“上海”，二者描述的是同一个地方，
  但如果我们不注意就忽略了
  
这个时候，可以先采用字典映射等方式对其进行转化，然后再使用上面所属的 Frequency 等编码重新对其进行处理

# 分箱计数


# 参考

* [频度统计](https://mp.weixin.qq.com/s/yQoaia_jJQsIdBGIe78PQw)
* [类别型特征的编码方法总结](https://mp.weixin.qq.com/s/emw05TSwjd-szqgirbpk9A)
* [特征工程--类别特征篇](https://mp.weixin.qq.com/s?__biz=Mzk0NDE5Nzg1Ng==&mid=2247494138&idx=1&sn=633fe6f67187f3cb46c9ff36a2108417&chksm=c32af075f45d7963858cf135e9f0f5065e0a48ad6b193aac682801b39be28757b099d09089ee&cur_album_id=1701045138849906691&scene=189#wechat_redirect)
* [K-Fold Target Encoding](https://medium.com/@pouryaayria/k-fold-target-encoding-dfe9a594874b)
* [平均数编码：针对某个分类特征类别基数特别大的编码方式](https://www.cnblogs.com/wzdLY/p/9639519.html)
* [平均编码](https://zhuanlan.zhihu.com/p/26308272)

