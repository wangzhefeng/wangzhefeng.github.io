---
title: Pandas
subtitle: 数据预处理
author: 王哲峰
date: '2022-11-28'
slug: timeseries-lib-pandas
categories:
  - timeseries
tags:
  - tool
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

- [Pandas 支持的四个与时间相关的概念](#pandas-支持的四个与时间相关的概念)
- [Freq](#freq)
- [Format](#format)
- [Date Time](#date-time)
  - [Timestamp](#timestamp)
  - [DatetimeIndex](#datetimeindex)
    - [DatetimeIndex 对象](#datetimeindex-对象)
    - [slice and index](#slice-and-index)
    - [freq infer](#freq-infer)
    - [DataFrame slice and index](#dataframe-slice-and-index)
  - [to\_datetime](#to_datetime)
    - [pd.Series](#pdseries)
    - [List](#list)
    - [dayfirst](#dayfirst)
    - [format](#format-1)
    - [pd.DataFrame](#pddataframe)
    - [errors](#errors)
    - [unit](#unit)
    - [origin](#origin)
  - [to\_localize](#to_localize)
  - [tz\_convert](#tz_convert)
  - [date\_range](#date_range)
    - [bdate\_range](#bdate_range)
    - [min/max Timestamp](#minmax-timestamp)
- [Time Deltas](#time-deltas)
  - [Timedelta](#timedelta)
  - [TimedeltaIndex](#timedeltaindex)
  - [to\_timedelta](#to_timedelta)
  - [timedelta\_range](#timedelta_range)
- [Time Spans](#time-spans)
  - [Period](#period)
  - [PeriodIndex](#periodindex)
  - [period\_range](#period_range)
- [Date Offset](#date-offset)
  - [DataOffset](#dataoffset)
  - [offsets](#offsets)
- [Time Zone](#time-zone)
  - [to\_localize](#to_localize-1)
  - [tz\_convert](#tz_convert-1)
- [NaT](#nat)
- [Window](#window)
- [Resampling](#resampling)
- [Difference](#difference)
- [Interpolate](#interpolate)
</p></details><p></p>

Pandas 提供了多种功能来支持时间序列数据。以下主要功能对于使用 Pandas 进行时间序列预测非常重要：

* 解析来自各种来源和格式的时间序列信息
* 生成固定频率日期和时间跨度的序列
* 利用时区信息处理和转换日期时间
* 将时间序列重采样或转换为特定频率
* 以绝对或相对时间增量执行日期和时间运算

# Pandas 支持的四个与时间相关的概念

* 日期时间 (Date Time)
    - 带有时区支持的特定日期和时间
* 时间增量 (Time Delta)
    - 用于操纵日期的绝对时间长度
* 时间跨度(Time Spans)
    - 由时间点及其关联的频率定义的持续时间
* 日期偏移 (Date Offset)
    - 涉及日历计算的相对持续时间

|概念          | 标量 Class   | 数组 Class        | 数据类型             | 创建方法             |
|-------------|--------------|------------------|---------------------|---------------------|
| Date Time   | `Timestamp`  | `DatetimeIndex`  | `datetime64[ns]`    | `to_datetime()`     |
|             |              |                  | `datetime64[ns,tz]` | `date_range()`      |
| Time Deltas | `Timedelta`  | `TimedeltaIndex` | `timedelta64[ns]`   | `to_timedelta()`    |
|             |              |                  |                     | `timedelta_range()` |
| Time Spans  | `Period`     | `PeriodIndex`    | `period[freq]`      | `Period()`          |
|             |              |                  |                     | `period_range()`    |
| Date Offset | `DateOffset` | `None`           | `None`              | `DateOffset`        |

# Freq

* `date_range(freq = "")`

|Alias|Description|
|-----|-----------|
|B|business day frequency|
|C|custom business day frequency|
|D|calendar day frequency|
|W|weekly frequency|
|M|month end frequency|
|SM|semi-month end frequency (15th and end of month)|
|BM|business month end frequency|
|CBM|custom business month end frequency|
|MS|month start frequency|
|SMS|semi-month start frequency (1st and 15th)|
|BMS|business month start frequency|
|CBMS|custom business month start frequency|
|Q|quarter end frequency|
|BQ|business quarter end frequency|
|QS|quarter start frequency|
|BQS|business quarter start frequency|
|A, Y|year end frequency|
|BA, BY|business year end frequency|
|AS, YS|year start frequency|
|BAS, BYS|business year start frequency|
|BH|business hour frequency|
|H|hourly frequency|
|T, min|minutely frequency|
|S|secondly frequency|
|L, ms|milliseconds|
|U, us|microseconds|
|N|nanoseconds|

# Format

* `to_datetime(format = "")`

|Directive|Example|
|---------|-------|
|**Year**||
|%Y|0001,2014,9999|
|%y|00,01,99|
|**Month**||
|%m|01,...12|
|%B|January,...,December|
|%b|Jan,...,Dec|
|**Day**||
|%d|01,...,30,31|
|**Week**||
|%w|0,...,6|
|%A|Monday,...,Sunday|
|%a|Mon,...,Sun|
|**Time**||
|%H|00,...,23|
|%I|00,...,12|
|%M|00,...,59|
|%S|00,...,59|
|%p|AM,PM|
|**Timezone**||
|%Z|001,002,...,366|

# Date Time

* class
    - Timestamp
    - DatetimeIndex
* function
    - to_datetime()
    - date_range()
    - bdate_range()
    - infer_freq

## Timestamp

```python
import datetime
import pandas as pd

# datetime object
datetime.datetime(2019, 8, 21)

# scalar datetime
pd.Timestamp(datetime.datetime(2019, 8, 21))
pd.Timestamp("2019-08-21")
pd.Timestamp(2019, 8, 21)

# day_name
wednesday = pd.Timestamp("2019-08-21")
print(wednesday.day_name())
```

```
datetime.datetime(2019, 8, 21, 0, 0)
Timestamp('2019-08-21 00:00:00')
Timestamp('2019-08-21 00:00:00')
Timestamp('2019-08-21 00:00:00')
Wednesday
```

## DatetimeIndex

### DatetimeIndex 对象

```python
dates = [
    pd.Timestamp("2019-08-21"), 
    pd.Timestamp("2019-08-22"), 
    pd.Timestamp("2019-08-23"),
]
ts = pd.Series(data = np.random.randn(3), index = dates)
print(ts)
print("-" * 25)
print(ts.index)
print("-" * 25)
print(type(ts.index))
```

```
2019-08-21   -0.195642
2019-08-22    0.433605
2019-08-23    0.519007
dtype: float64
-------------------------
DatetimeIndex(['2019-08-21', '2019-08-22', '2019-08-23'], dtype='datetime64[ns]', freq=None)
-------------------------
<class 'pandas.core.indexes.datetimes.DatetimeIndex'>
```

### slice and index

```python
start = datetime.datetime(2011, 1, 1)
end = datetime.datetime(2012, 1, 1)
rng = pd.date_range(start, end, freq = "BM") # Business Month 
ts = pd.Series(np.random.randn(len(rng)), index = rng)
print(ts)
print("-" * 100)

print(ts.index)
print("-" * 100)

print(ts[:5].index)
print("-" * 100)

print(ts[::2].index)
print("-" * 100)

print(ts["1/31/2011"])
print("-" * 100)

print(ts[datetime.datetime(2011, 12, 25):])
print("-" * 100)

print(ts['10/31/2011':'12/31/2011'])
print("-" * 100)

print(ts["2011"])
print("-" * 100)

print(ts["2011-6"])
```

```
2011-01-31    0.341942
2011-02-28    0.859577
2011-03-31    0.985406
2011-04-29   -0.224811
2011-05-31   -0.888166
2011-06-30    0.710712
2011-07-29    0.355943
2011-08-31   -2.422465
2011-09-30   -0.769204
2011-10-31    1.886651
2011-11-30    0.456291
2011-12-30    1.312205
Freq: BM, dtype: float64
----------------------------------------------------------------------------------------------------
DatetimeIndex(['2011-01-31', '2011-02-28', '2011-03-31', '2011-04-29',
               '2011-05-31', '2011-06-30', '2011-07-29', '2011-08-31',
               '2011-09-30', '2011-10-31', '2011-11-30', '2011-12-30'],
              dtype='datetime64[ns]', freq='BM')
----------------------------------------------------------------------------------------------------
DatetimeIndex(['2011-01-31', '2011-02-28', '2011-03-31', '2011-04-29',
               '2011-05-31'],
              dtype='datetime64[ns]', freq='BM')
----------------------------------------------------------------------------------------------------
DatetimeIndex(['2011-01-31', '2011-03-31', '2011-05-31', '2011-07-29',
               '2011-09-30', '2011-11-30'],
              dtype='datetime64[ns]', freq='2BM')
----------------------------------------------------------------------------------------------------
0.3419422485361135
----------------------------------------------------------------------------------------------------
2011-12-30    1.312205
Freq: BM, dtype: float64
----------------------------------------------------------------------------------------------------
2011-10-31    1.886651
2011-11-30    0.456291
2011-12-30    1.312205
Freq: BM, dtype: float64
----------------------------------------------------------------------------------------------------
2011-01-31    0.341942
2011-02-28    0.859577
2011-03-31    0.985406
2011-04-29   -0.224811
2011-05-31   -0.888166
2011-06-30    0.710712
2011-07-29    0.355943
2011-08-31   -2.422465
2011-09-30   -0.769204
2011-10-31    1.886651
2011-11-30    0.456291
2011-12-30    1.312205
Freq: BM, dtype: float64
----------------------------------------------------------------------------------------------------
2011-06-30    0.710712
Freq: BM, dtype: float64
```

### freq infer

```python
print(pd.DatetimeIndex(["2018-01-01", "2018-01-03", "2018-01-05"]))
print(pd.DatetimeIndex(["2018-01-01", "2018-01-03", "2018-01-05"], freq = "infer"))
```

```
DatetimeIndex(['2018-01-01', '2018-01-03', '2018-01-05'], dtype='datetime64[ns]', freq=None)
DatetimeIndex(['2018-01-01', '2018-01-03', '2018-01-05'], dtype='datetime64[ns]', freq='2D')
```

### DataFrame slice and index

```python
dft = pd.DataFrame(np.random.randn(100000, 1),
                   columns = ["A"],
                   index = pd.date_range("20130101", periods = 100000, freq = "T")) # minutely 
print(dft)
print("-" * 100)
print(dft["2013"])
print("-" * 100)
print(dft["2013-1":"2013-2"])
print("-" * 100)
print(dft["2013-1":"2013-2-28"])
print("-" * 100)
print(dft["2013-1":"2013-2-28 00:00:00"])
print("-" * 100)
print(dft["2013-1-15":"2013-1-15 12:30:00"])
```

```python
dft2 = pd.DataFrame(
    data = np.random.randn(20, 1),
    columns = ["A"],
    index = pd.MultiIndex.from_product([
        pd.date_range("20130101", periods = 10, freq = "12H"),
        ["a", "b"]
    ]))
print(dft2)
print("-" * 100)

try:
    print(dft2["2013-01-05"])
except:
    print("ERROR")
print("-" * 100)

print(dft2.loc["2013-01-05"])
print("-" * 100)

dft2 = dft2.swaplevel(0, 1).sort_index()
print(dft2)
print("-" * 100)

idx = pd.IndexSlice
print(dft2.loc[idx[:, "2013-01-05"], :])
```

## to_datetime

* `pd.to_datetime(pd.Series/List, dayfirst, format, errors, unit, origin)`

### pd.Series

```python
s = pd.Series(data = ["Jul 31, 2009", "2010-01-10", None])
ts  = pd.to_datetime(s)
print(ts)
```

```
0   2009-07-31
1   2010-01-10
2          NaT
dtype: datetime64[ns]
```

### List

```python
l = [
    '1/1/2018',
    np.datetime64('2018-01-01'), 
    datetime.datetime(2018, 1, 1),
]
ts = pd.to_datetime(l)
ts
```

```
DatetimeIndex(['2018-01-01', '2018-01-01', '2018-01-01'], 
dtype='datetime64[ns]', freq=None)
```

### dayfirst

```python
d = ["04-01-2012 10:00"]
td1 = pd.to_datetime(d)
td2 = pd.to_datetime(d, dayfirst = True)
print(td1)
print(td2)
print("-" * 80)
d2 = ["14-01-2019", "01-14-2019"]
td3 = pd.to_datetime(d2)
td4 = pd.to_datetime(d2, dayfirst = True)
print(td3)
print(td4)
```

```
DatetimeIndex(['2012-04-01 10:00:00'], dtype='datetime64[ns]', freq=None)
DatetimeIndex(['2012-01-04 10:00:00'], dtype='datetime64[ns]', freq=None)
--------------------------------------------------------------------------------
DatetimeIndex(['2019-01-14', '2019-01-14'], dtype='datetime64[ns]', freq=None)
DatetimeIndex(['2019-01-14', '2019-01-14'], dtype='datetime64[ns]', freq=None)
```

### format

```python
print(pd.to_datetime("2010/11/12", format = "%Y/%m/%d"))
print(pd.to_datetime("12-11-2010 00:00", format = "%d-%m-%Y %H:%M"))
```

```
2010-11-12 00:00:00
2010-11-12 00:00:00
```

### pd.DataFrame

```python
df = pd.DataFrame({
    "year": [2015, 2016],
    "month": [2, 3],
    "day": [4, 5],
    "hour": [2, 3]
})
print(df)
print()
print(pd.to_datetime(df))
print()
print(pd.to_datetime(df[["year", "month", "day"]]))
```

```
   year  month  day  hour
0  2015      2    4     2
1  2016      3    5     3

0   2015-02-04 02:00:00
1   2016-03-05 03:00:00
dtype: datetime64[ns]

0   2015-02-04
1   2016-03-05
dtype: datetime64[ns]
```

### errors

```python
try:
    pd.to_datetime(["2019-08-21", "asd"], errors = "raise")
except ValueError as e:
    print(e)

print(pd.to_datetime(["2019-08-21", "asd"], errors = "ignore"))
print(pd.to_datetime(["2019-08-21", "asd"], errors = "coerce"))
```

```
('Unknown string format:', 'asd')
['2019-08-21' 'asd']
DatetimeIndex(['2019-08-21', 'NaT'], dtype='datetime64[ns]', freq=None)
```

### unit

```python
print(pd.to_datetime([
    1349720105, 1349806505, 1349892905, 1349979305, 1350065705
], unit = "s"))
print(pd.to_datetime([
    1349720105000, 1349806505000, 1349892905000, 1349979305000, 1350065705000
], unit = "ms"))
```

```
DatetimeIndex(['2012-10-08 18:15:05', '2012-10-09 18:15:05',
               '2012-10-10 18:15:05', '2012-10-11 18:15:05',
               '2012-10-12 18:15:05'],
              dtype='datetime64[ns]', freq=None)
DatetimeIndex(['2012-10-08 18:15:05', '2012-10-09 18:15:05',
               '2012-10-10 18:15:05', '2012-10-11 18:15:05',
               '2012-10-12 18:15:05'],
              dtype='datetime64[ns]', freq=None)
```

### origin

```python
print(pd.to_datetime([1, 2, 3], unit = "D", origin = pd.Timestamp('1960-01-01')))
print(pd.to_datetime([1, 2, 3], unit = "D"))
```

```
DatetimeIndex(['1960-01-02', '1960-01-03', '1960-01-04'], dtype='datetime64[ns]', freq=None)
DatetimeIndex(['1970-01-02', '1970-01-03', '1970-01-04'], dtype='datetime64[ns]', freq=None)
```

## to_localize

```python
print(pd.to_datetime([1262347200000000000]).tz_localize("US/Pacific"))
print(pd.to_datetime([1262347200000000000]).tz_localize("UTC"))
print(pd.DatetimeIndex([1262347200000000000]).tz_localize("US/Pacific"))
```

```
DatetimeIndex(['2010-01-01 12:00:00-08:00'], dtype='datetime64[ns, US/Pacific]', freq=None)
DatetimeIndex(['2010-01-01 12:00:00+00:00'], dtype='datetime64[ns, UTC]', freq=None)
DatetimeIndex(['2010-01-01 12:00:00-08:00'], dtype='datetime64[ns, US/Pacific]', freq=None)
```

## tz_convert

```python
print(pd.DatetimeIndex([1262347200000000000]).tz_convert('US/Pacific'))
```

```
DatetimeIndex(['2010-01-01 12:00:00-08:00'], dtype='datetime64[ns, US/Pacific]', freq=None)
```

## date_range

* `pd.date_range(start, end, freq, period)`
* `pd.bdate_range(start, end, freq, period)`

```python
start = datetime.datetime(2018, 1, 1)
end = datetime.datetime(2019, 1, 1)

print(pd.date_range(start, end))  # Calendar day 
print(pd.bdate_range(start, end)) # Business day

print(pd.date_range(start, periods = 1000, freq = "M"))   # Month End
print(pd.bdate_range(start, periods = 250, freq = "BQS")) # Business Quarter Start

print(pd.date_range(start, end, freq = "BM")) # Business Month End
print(pd.date_range(start, end, freq = "W")) # Weekly

print(pd.date_range(end = end, periods = 20))     # end and length
print(pd.date_range(start = start, periods = 20)) # start and length

print(pd.date_range("2018-01-01", "2018-01-05", periods = 5)) # start and end and length
print(pd.date_range("2018-01-01", "2018-01-05", periods = 10))# start and end and length
```

```
DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04',
               '2018-01-05', '2018-01-06', '2018-01-07', '2018-01-08',
               '2018-01-09', '2018-01-10',
               ...
               '2018-12-23', '2018-12-24', '2018-12-25', '2018-12-26',
               '2018-12-27', '2018-12-28', '2018-12-29', '2018-12-30',
               '2018-12-31', '2019-01-01'],
              dtype='datetime64[ns]', length=366, freq='D')
DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04',
               '2018-01-05', '2018-01-08', '2018-01-09', '2018-01-10',
               '2018-01-11', '2018-01-12',
               ...
               '2018-12-19', '2018-12-20', '2018-12-21', '2018-12-24',
               '2018-12-25', '2018-12-26', '2018-12-27', '2018-12-28',
               '2018-12-31', '2019-01-01'],
              dtype='datetime64[ns]', length=262, freq='B')
----------------------------------------------------------------------------------------------------
DatetimeIndex(['2018-01-31', '2018-02-28', '2018-03-31', '2018-04-30',
               '2018-05-31', '2018-06-30', '2018-07-31', '2018-08-31',
               '2018-09-30', '2018-10-31',
               ...
               '2100-07-31', '2100-08-31', '2100-09-30', '2100-10-31',
               '2100-11-30', '2100-12-31', '2101-01-31', '2101-02-28',
               '2101-03-31', '2101-04-30'],
              dtype='datetime64[ns]', length=1000, freq='M')
DatetimeIndex(['2018-01-01', '2018-04-02', '2018-07-02', '2018-10-01',
               '2019-01-01', '2019-04-01', '2019-07-01', '2019-10-01',
               '2020-01-01', '2020-04-01',
               ...
               '2078-01-03', '2078-04-01', '2078-07-01', '2078-10-03',
               '2079-01-02', '2079-04-03', '2079-07-03', '2079-10-02',
               '2080-01-01', '2080-04-01'],
              dtype='datetime64[ns]', length=250, freq='BQS-JAN')
----------------------------------------------------------------------------------------------------
DatetimeIndex(['2018-01-31', '2018-02-28', '2018-03-30', '2018-04-30',
               '2018-05-31', '2018-06-29', '2018-07-31', '2018-08-31',
               '2018-09-28', '2018-10-31', '2018-11-30', '2018-12-31'],
              dtype='datetime64[ns]', freq='BM')
DatetimeIndex(['2018-01-07', '2018-01-14', '2018-01-21', '2018-01-28',
               '2018-02-04', '2018-02-11', '2018-02-18', '2018-02-25',
               '2018-03-04', '2018-03-11', '2018-03-18', '2018-03-25',
               '2018-04-01', '2018-04-08', '2018-04-15', '2018-04-22',
               '2018-04-29', '2018-05-06', '2018-05-13', '2018-05-20',
               '2018-05-27', '2018-06-03', '2018-06-10', '2018-06-17',
               '2018-06-24', '2018-07-01', '2018-07-08', '2018-07-15',
               '2018-07-22', '2018-07-29', '2018-08-05', '2018-08-12',
               '2018-08-19', '2018-08-26', '2018-09-02', '2018-09-09',
               '2018-09-16', '2018-09-23', '2018-09-30', '2018-10-07',
               '2018-10-14', '2018-10-21', '2018-10-28', '2018-11-04',
               '2018-11-11', '2018-11-18', '2018-11-25', '2018-12-02',
               '2018-12-09', '2018-12-16', '2018-12-23', '2018-12-30'],
              dtype='datetime64[ns]', freq='W-SUN')
----------------------------------------------------------------------------------------------------
DatetimeIndex(['2018-12-13', '2018-12-14', '2018-12-15', '2018-12-16',
               '2018-12-17', '2018-12-18', '2018-12-19', '2018-12-20',
               '2018-12-21', '2018-12-22', '2018-12-23', '2018-12-24',
               '2018-12-25', '2018-12-26', '2018-12-27', '2018-12-28',
               '2018-12-29', '2018-12-30', '2018-12-31', '2019-01-01'],
              dtype='datetime64[ns]', freq='D')
DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04',
               '2018-01-05', '2018-01-06', '2018-01-07', '2018-01-08',
               '2018-01-09', '2018-01-10', '2018-01-11', '2018-01-12',
               '2018-01-13', '2018-01-14', '2018-01-15', '2018-01-16',
               '2018-01-17', '2018-01-18', '2018-01-19', '2018-01-20'],
              dtype='datetime64[ns]', freq='D')
----------------------------------------------------------------------------------------------------
DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04',
               '2018-01-05'],
              dtype='datetime64[ns]', freq=None)
DatetimeIndex(['2018-01-01 00:00:00', '2018-01-01 10:40:00',
               '2018-01-01 21:20:00', '2018-01-02 08:00:00',
               '2018-01-02 18:40:00', '2018-01-03 05:20:00',
               '2018-01-03 16:00:00', '2018-01-04 02:40:00',
               '2018-01-04 13:20:00', '2018-01-05 00:00:00'],
              dtype='datetime64[ns]', freq=None)
```

### bdate_range

```python
weekmask = "Mon Wed Fri"
holidays = [datetime.datetime(2011, 1, 5), datetime.datetime(2011, 3, 14)]

idx1 = pd.bdate_range(start, end, freq = "C", weekmask = weekmask, holidays = holidays)
print(idx1)

idx2 = pd.bdate_range(start, end, freq = "CBMS", weekmask = weekmask)
print(idx2)
```

```
DatetimeIndex(['2018-01-01', '2018-01-03', '2018-01-05', '2018-01-08',
               '2018-01-10', '2018-01-12', '2018-01-15', '2018-01-17',
               '2018-01-19', '2018-01-22',
               ...
               '2018-12-10', '2018-12-12', '2018-12-14', '2018-12-17',
               '2018-12-19', '2018-12-21', '2018-12-24', '2018-12-26',
               '2018-12-28', '2018-12-31'],
              dtype='datetime64[ns]', length=157, freq='C')
DatetimeIndex(['2018-01-01', '2018-02-02', '2018-03-02', '2018-04-02',
               '2018-05-02', '2018-06-01', '2018-07-02', '2018-08-01',
               '2018-09-03', '2018-10-01', '2018-11-02', '2018-12-03'],
              dtype='datetime64[ns]', freq='CBMS')
```

### min/max Timestamp

```python
print(pd.Timestamp.min)
print(pd.Timestamp.max)
```

```
1677-09-21 00:12:43.145225
2262-04-11 23:47:16.854775807
```



# Time Deltas

* class
    - Timedelta
    - TimedeltaIndex
* function
    - to_timedelta()
    - timedelta_range()

## Timedelta

```python
friday = pd.Timestamp("2019-08-23")
print(friday.day_name())
stariday = friday + pd.Timedelta("1 day")
print(stariday.day_name())
```

```
Friday
Saturday
```

## TimedeltaIndex


## to_timedelta


## timedelta_range



# Time Spans

* class
    - Period
    - PeriodIndex
* function
    - Period()
    - period_range()

## Period

```python
pd.Period("2019-08")
pd.Period("2019-08", freq = "D")
```

```
Period('2019-08', 'M')
Period('2019-08-01', 'D')
```

## PeriodIndex

```python
periods = [
    pd.Period("2019-08"), 
    pd.Period("2019-07"), 
    pd.Period("2019-06"),
]
ts = pd.Series(data = np.random.randn(3), index = periods)
print(ts)
print("-" * 20)
print(ts.index)
print("-" * 20)
print(type(ts.index))
```

```
2019-08   -0.999400
2019-07   -0.213444
2019-06   -1.463501
Freq: M, dtype: float64
--------------------
PeriodIndex(['2019-08', '2019-07', '2019-06'], dtype='period[M]', freq='M')
--------------------
<class 'pandas.core.indexes.period.PeriodIndex'>
```


## period_range

```python
ps = pd.Series(
    data = pd.period_range("1/1/2011", freq = "M", periods = 3)
)
ps
```

```
0   2011-01
1   2011-02
2   2011-03
dtype: object
```

# Date Offset

## DataOffset

```python
ps = pd.Series(data = [pd.DateOffset(1), pd.DateOffset(2)])
ps
```

```
0         <DateOffset>
1    <2 * DateOffsets>
dtype: object
```

## offsets

```python
friday = pd.Timestamp("2019-08-23")
print(friday.day_name())
monday = friday + pd.offsets.BDay()
print(monday.day_name())
```

```
Monday
```

# Time Zone

* tz_localize()
* tz_convert()

## to_localize


## tz_convert


# NaT

```python
print(pd.Timestamp(pd.NaT))
print(pd.Timedelta(pd.NaT))
print(pd.Period(pd.NaT))
print(pd.NaT == pd.NaT)
```

```
NaT
NaT
NaT
False
```


# Window

* pd.DataFrame.rolling().mean()
* pd.DataFrame.expanding().mean()
* pd.DataFrame.ewm().mean()

# Resampling

* pd.DataFrame.resample()
* pd.Series.resample()

```python
idx = pd.date_range("2019-01-01", periods = 6, freq = "H")
ts = pd.Series(
    data = range(len(idx)),
    index = idx,
)
ts
```

```
2019-01-01 00:00:00    0
2019-01-01 01:00:00    1
2019-01-01 02:00:00    2
2019-01-01 03:00:00    3
2019-01-01 04:00:00    4
2019-01-01 05:00:00    5
Freq: H, dtype: int64
```

```python
ts.resample("2H").mean()
```

```
2019-01-01 00:00:00    0.5
2019-01-01 02:00:00    2.5
2019-01-01 04:00:00    4.5
Freq: 2H, dtype: float64
```


# Difference 

* pd.DataFrame.diff()
* pd.Series.diff()

# Interpolate




