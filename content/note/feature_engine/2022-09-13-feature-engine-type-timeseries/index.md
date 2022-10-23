---
title: Timeseries
author: 王哲峰
date: '2022-09-13'
slug: feature-engine-type-timeseries
categories:
  - feature engine
tags:
  - ml
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

- [时间序列特征构造基本准则](#时间序列特征构造基本准则)
- [时间序列特征构造方法](#时间序列特征构造方法)
- [时间特征](#时间特征)
  - [连续时间](#连续时间)
  - [离散时间](#离散时间)
- [时间序列历史特征](#时间序列历史特征)
  - [统计值](#统计值)
  - [同期值](#同期值)
- [时间序列交叉特征](#时间序列交叉特征)
- [时间序列隐蔽特征](#时间序列隐蔽特征)
  - [基于线性模型的特征](#基于线性模型的特征)
- [时间序列元特征](#时间序列元特征)
  - [元特征抽取](#元特征抽取)
  - [预测](#预测)
  - [示例](#示例)
</p></details><p></p>

# 时间序列特征构造基本准则

- 构造时序特征时一定要算好时间窗口, 特别是在工作的时候, 需要自己去设计训练集和测试集, 
   千万不要出现数据泄露的情况(比如说预测明天的数据时, 是拿不到今天的特征的)
- 针对上面的情况, 可以尝试将今天的数据进行补齐
- 有些特征加上去效果会变差, 大概率是因为过拟合了
- 有些特征加上去效果出奇好, 第一时间要想到是不是数据泄露了
- 拟合不好的时间(比如说双休日)可以分开建模
- ont-hot 对 xgboost 效果的提升很显著
- 离散化对 xgboost 效果的提升也很显著
- 对标签做个平滑效果可能会显著提升
- 多做数据分析, 多清洗数据

# 时间序列特征构造方法

- **时间特征**
    - 连续时间
        - 持续时间
        - 间隔时间
    - 离散时间
        - 年、季度、季节、月、星期、日、时、分、秒等
        - 节假日、节假日前第n天、节假日第n天、节假日后n天等
        - 一天的哪个时间段(上午、下午、傍晚、晚上等)
        - 年初、年末、月初、月末、周内、周末
        - 是否高峰时段、是否上班、营业
- **时间序列历史特征**
    - 统计值 
        - 四分位数
        - 中位数 
        - 平均数
        - 偏度、峰度
        - 离散系数
    - 同期值
- **时间序列交叉特征**
    - 类别特征与类别特征
        - 笛卡尔积
    - 连续特征与类别特征
        - 离散后笛卡尔积
        - 聚合特征
    - 连续特征与连续特征
        - 一阶差分(同比、环比)
        - 二阶差分

# 时间特征

## 连续时间

- 连续时间
    - 时长
- 间隔时间
    - 距今时长
    - 距离假期的前后时长(节假日前、后可能出现明显的数据波动)

## 离散时间

- 年、季度、季节、月、星期、日、时 等
    - 基本特征, 如果用 Xgboost 模型可以进行 one-hot 编码
    - 如果类别比较多, 可以尝试平均数编码(Mean Encoding)
    - 或者取 cos/sin 将数值的首位衔接起来, 比如说 23 点与 0 点很近, 星期一和星期天很近
- 节假日、节假日第 n 天、节假日前 n 天、节假日后 n 天
    - 数据可能会随着节假日的持续而发生变化, 比如说递减
    - 节假日前/后可能会出现数据波动
    - 不放假的人造节日如 5.20、6.18、11.11 等也需要考虑一下
- 一天的某个时间段
    - 上午、中午、下午、傍晚、晚上、深夜、凌晨等
- 年初、年末、月初、月末、周内、周末
    - 基本特征
- 高峰时段、是否上班、是否营业、是否双休日
    - 主要根据业务场景进行挖掘

```python
# 年、季度、季节、月、星期、日、时、分、秒等
data_df['date'] = pd.to_datetime(data_df['date'], format = '%m/%d/%y')

data_df['year'] = data_df['date'].dt.year
data_df['quarter'] = data_df['date'].dt.quarter
data_df['month'] = data_df['date'].dt.month
data_df['day'] = data_df['date'].dt.day
data_df['hour'] = data_df['date'].dt.hour
data_df['minute'] = data_df['date'].dt.minute
data_df['second'] = data_df['date'].dt.second
data_df['dayofweek'] = data_df['date'].dt.dayofweek
data_df['weekofyear'] = data_df['date'].dt.week

data_df['is_year_start'] = data_df['date'].dt.is_year_start
data_df['is_year_end'] = data_df['date'].dt.is_year_end
data_df['is_quarter_start'] = data_df['date'].dt.is_quarter_start
data_df['is_quarter_end'] = data_df['date'].dt.is_quarter_end
data_df['is_month_start'] = data_df['date'].dt.is_month_start
data_df['is_month_end'] = data_df['date'].dt.is_month_end

# 是否是一天的高峰时段 8-10
data_df['day_high'] = data_df['hour'].apply(lambda x: 0 if 0 < x < 8 else 1)

# 构造时间特征
def get_time_fe(data, col, n, one_hot = False, drop = True):
    '''
    data: DataFrame
    col: column name
    n: 时间周期
    '''
    data[col + '_sin'] = round(np.sin(2*np.pi / n * data[col]), 6)
    data[col + '_cos'] = round(np.cos(2*np.pi / n * data[col]), 6)
    if one_hot:
        ohe = OneHotEncoder()
        X = OneHotEncoder().fit_transform(data[col].values.reshape(-1, 1)).toarray()
        df = pd.DataFrame(X, columns=[col + '_' + str(int(i)) for i in range(X.shape[1])])
        data = pd.concat([data, df], axis=1)
        if drop:
            data = data.drop(col, axis=1)

    return data

    data_df = get_time_fe(data_df, 'hour', n=24, one_hot=False, drop=False)
    data_df = get_time_fe(data_df, 'day', n=31, one_hot=False, drop=True)
    data_df = get_time_fe(data_df, 'dayofweek', n=7, one_hot=True, drop=True)
    data_df = get_time_fe(data_df, 'season', n=4, one_hot=True, drop=True)
    data_df = get_time_fe(data_df, 'month', n=12, one_hot=True, drop=True)
    data_df = get_time_fe(data_df, 'weekofyear', n=53, one_hot=False, drop=True)
```

# 时间序列历史特征

## 统计值

- 基于历史数据构造长中短期的统计值, 包括前 n 天/周期内的: 
    - 四分位数
    - 中位数、平均数、偏差
    - 偏度、峰度
        - 挖掘数据的偏离程度和集中程度
    - 离散系数        
        - 挖掘离散程度

这里可以用自相关系数(autocorrelation)挖掘出周期性. 

除了对数据进行统计外, 也可以对节假日等进行统计, 
以刻画历史数据中所含节假日的情况. (还可以统计未来的节假日的情况)

```python
from pandas.plotting import autocorrelation_plot

# 自相关性系数图
autocorrelation_plot(data['value'])

# 构造过去 n 天的统计数据
def get_statis_n_days_num(data, col, n):
    temp = pd.DataFrame()
    for i in range(n):
        temp = pd.concat([temp, data[col].shift((i + 1) * 24)], axis = 1)
        data['avg_' + str(n) + '_days_' + col] = temp.mean(axis = 1)
        data['median_' + str(n) + '_days_' + col] = temp.median(axis = 1)
        data['max_' + str(n) + '_days_' + col] = temp.max(axis = 1)
        data['min_' + str(n) + '_days_' + col] = temp.min(axis = 1)
        data['std_' + str(n) + '_days_' + col] = temp.std(axis = 1)
        data['mad_' + str(n) + '_days_' + col] = temp.mad(axis = 1)
        data['skew_' + str(n) + '_days_' + col] = temp.skew(axis = 1)
        data['kurt_' + str(n) + '_days_' + col] = temp.kurt(axis = 1)
        data['q1_' + str(n) + '_days_' + col] = temp.quantile(q = 0.25, axis = 1)
        data['q3_' + str(n) + '_days_' + col] = temp.quantile(q = 0.75, axis = 1)
        data['var_' + str(n) + '_days_' + col] = data['std' + str(n) + '_days_' + col] / data['avg_' + str(n) + '_days_' + col]

    return data

data_df = get_statis_n_days_num(data_df, 'num_events', n = 7)
data_df = get_statis_n_days_num(data_df, 'num_events', n = 14)
data_df = get_statis_n_days_num(data_df, 'num_events', n = 21)
data_df = get_statis_n_days_num(data_df, 'num_events', n = 28)
```

## 同期值

- 前 n 个周期/天/月/年的同期值

```python
# n个星期前的同期特征
data_df['ago_7_day_num_events'] = data_df['num_events'].shift(7 * 24)
data_df['ago_14_day_num_events'] = data_df['num_events'].shift(14 * 24)
data_df['ago_21_day_num_events'] = data_df['num_events'].shift(21 * 24)
data_df['ago_28_day_num_events'] = data_df['num_events'].shift(28 * 24)

# 昨天的同期特征
data_df['ago_7_day_num_events'] = data_df['num_events'].shift(1 * 24)
```

# 时间序列交叉特征

- 类别特征间组合构成新特征
    - 笛卡尔积, 比如星期和小时: Mon_10(星期一的十点)
- 类别特征和连续特征
    - 连续特征分桶后进行笛卡尔积
    - 基于类别特征进行 groupby 操作, 类似聚合特征的构造
- 连续特征和连续特征
    - 同比和环比(一阶差分): 反应同期或上一个统计时段的变换大小
    - 二阶差分: 反应变化趋势
    - 比值

```python
# 一阶差分
data_df['ago_28_21_day_num_trend'] = data_df['ago_28_day_num_events'] - data_df['ago_21_day_num_events']
data_df['ago_21_14_day_num_trend'] = data_df['ago_21_day_num_events'] - data_df['ago_14_day_num_events']
data_df['ago_14_7_day_num_trend'] = data_df['ago_14_day_num_events'] - data_df['ago_7_day_num_events']
data_df['ago_7_1_day_num_trend'] = data_df['ago_7_day_num_events'] - data_df['ago_1_day_num_events']
```

***
**Note:**

* 特征交叉一般从重要特征下手, 慢工出细活
***      

# 时间序列隐蔽特征

在时间序列问题中, 经常习惯做非常多的手工统计特征, 包括一些序列的近期的情况, 
序列的趋势信息, 周期信息等等, 除了手动构建特征之外, 其实还存在许多非常隐蔽的特征, 
这些特征是直接通过模型产出的, 这里介绍基于线性模型的特征. 

## 基于线性模型的特征

1. 线性模型的系数特征
    - 通过线性回归的系数来表示近期的趋势特征

```python
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(X, y)
reg.coef_
```

2. 线性模型的残差特征
    - 直接计算线性回归的预测结果与真实值差值(残差), 并计算所有差值的均值作为残差特征

```python
preds = reg.predict(X)
residual = np.mean(np.abs(y - preds))
```

# 时间序列元特征

在时间序列等相关的问题中，除了许多传统的时间序列相关的统计特征之外，
还有一类非常重要的特征，这类特征并不是基于手工挖掘的，而是由机器学习模型产出的，
但更为重要的是，它往往能为模型带来巨大的提升

对时间序列抽取元特征，一共需要进行两个步骤：

* 抽取元特征
* 将元特征拼接到一起重新训练预测得到最终预测的结果

## 元特征抽取

元特征抽取部分，操作如下：

* 先把数据按时间序列分为两块
* 使用时间序列的第一块数据训练模型得到模型 1
* 使用时间序列的第二块数据训练模型得到模型 2
* 使用模型 1 对第二块的数据进行预测得到第二块数据的元特征
* 使用模型 2 对测试集进行预测得到测试集的元特征

![img](images/meta-feature.jpg)


## 预测

将元特征作为新的特征，与原来的数据进行拼接，重新训练新的模型，
并用新训练的模型预测得到最终的结果

![img](images/meta-feature-forecast.jpg)

## 示例

* [ ] TODO