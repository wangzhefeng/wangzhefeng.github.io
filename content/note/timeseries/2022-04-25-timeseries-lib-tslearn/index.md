---
title: Lib tslearn
author: 王哲峰
date: '2022-04-25'
slug: timeseries-lib-tslearn
categories:
  - timeseries
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

- [tsfresh 安装](#tsfresh-安装)
- [tsfresh 数据格式](#tsfresh-数据格式)
  - [Flat DataFrame](#flat-dataframe)
  - [Stacked DataFrame](#stacked-dataframe)
  - [Dictionary of flat DataFrame](#dictionary-of-flat-dataframe)
</p></details><p></p>

# tsfresh 安装

```bash
$ pip install tsfresh
```

# tsfresh 数据格式

输入数据格式: 

   - Flat DataFrame
   - Stacked DataFrame
   - dictionary of flat DataFrame

| column_id | column_value | column_sort | column_kind |
|-----------|--------------|-------------|-------------|
| id        | value        | sort        | kind        |

## Flat DataFrame

| id       | time     | x        | y        |          |          |          |          |          |    | A  | t1  | x(A, t1) | y(A, t1) |
|----------|----------|----------|----------|----------|----------|----------|----------|----------|----|----|----|----|----|
| A        | t2       | x(A, t2) | y(A, t2) | A        | t3       | x(A, t3) | y(A, t3) | B        | t1 |    |     |          |          |
| x(B, t1) | y(B, t1) | B        | t2       | x(B, t2) | y(B, t2) | B        | t3       | x(B, t3) |    |    |     |          |          |
| y(B, t3) |          |          |          |          |          |          |          |          |    |    |     |          |          |

## Stacked DataFrame

| id | time | kind | value    |   |    |   |          |   |   | A | t1       | x | x(A, t1) |
|----|------|------|----------|---|----|---|----------|---|---|---|----------|---|----------|
| A  | t2   | x    | x(A, t2) | A | t3 | x | x(A, t3) | A |t1 |y  | y(A, t1) |   |          |
| A  | t2   | y    | y(A, t2) | A | t3 | y | y(A, t3) | B |t1 |x  | x(B, t1) |   |          |
| B  | t2   | x    | x(B, t2) | B | t3 | x | x(B, t3) | B |t1 |y  | y(B, t1) |   |          |
| B  | t2   | y    | y(B, t2) | B | t3 | y | y(B, t3) |   |   |   |          |   |          |

## Dictionary of flat DataFrame

{ "x”:

| id | time | value    |
|----|------|----------|
| A  | t1   | x(A, t1) |
| A  | t2   | x(A, t2) |
| A  | t3   | x(A, t3) |
| B  | t1   | x(B, t1) |
| B  | t2   | x(B, t2) |
| B  | t3   | x(B, t3) |

, "y”:

| id | time | value    |
|----|------|----------|
| A  | t1   | y(A, t1) |
| A  | t2   | y(A, t2) |
| A  | t3   | y(A, t3) |
| B  | t1   | y(B, t1) |
| B  | t2   | y(B, t2) |
| B  | t3   | y(B, t3) |

}

输出数据格式: 

| id | x\ *feature*\ 1 | … | x\ *feature*\ N | y\ *feature*\ 1 | … | y\ *feature*\ N |
|----|-----------------|---|-----------------|-----------------|---|-----------------|
| A  | …               | … | …               | …               | … | …               |
| B  | …               | … | …               | …               | … | …               |
