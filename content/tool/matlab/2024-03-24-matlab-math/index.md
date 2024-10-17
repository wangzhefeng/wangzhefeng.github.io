---
title: MATLAB 数学
author: wangzf
date: '2024-03-24'
slug: matlab-math
categories:
  - MATLAB
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
img {
    pointer-events: none;
}
</style>

<details><summary>目录</summary><p>

- [Matlab 符号运算](#matlab-符号运算)
  - [符号对象的建立](#符号对象的建立)
  - [查找符号变量](#查找符号变量)
  - [替换符号变量](#替换符号变量)
  - [因式分解](#因式分解)
  - [函数展开](#函数展开)
  - [合并同类项](#合并同类项)
  - [函数简化](#函数简化)
  - [计算极限](#计算极限)
  - [计算导数](#计算导数)
  - [计算积分](#计算积分)
  - [符号求和](#符号求和)
  - [代数方程求解](#代数方程求解)
  - [微分方程求解](#微分方程求解)
</p></details><p></p>

# Matlab 符号运算

## 符号对象的建立

* sym

```matlab
符号变量 = sym(A)
```

* syms

```matlab
syms 符号变量1 符号变量2 ... 符号变量n
```

## 查找符号变量

> 查找符号表达式种的符号变量

* `findsym(expr)` 按字母顺序列出符号表达式 `expr` 中的所有符号变量
* `findsym(expr, N)` 按顺序列出符号表达式 `expr` 中离 `$x$` 最近的 N 个符号变量

## 替换符号变量

> 用给定的数据替换符号表达式中的指定符号变量


```matlab
% 用 a 替换字符函数 f 中的字符变量 x
subs(f, x, a)
```

## 因式分解

```matlab
% 符号变量
syms x;

% 符号表达式
f = x^6 + 1;

% 因式分解
factor(f)
```

## 函数展开

```matlab
% 符号变量
syms x;

% 符号表达式
f = (x + 1)^6

% 函数展开
expand(f)
```

## 合并同类项

```matlab
% 按指定变量 v 进行合并
collect(f, v)
```

## 函数简化

```matlab
% y 为 f 的最简短形式，How 中记录的是简化过程中使用的方法
[How, y] = simple(f)
```

## 计算极限

```matlab
% 当变量 x 趋向于 a 时 f 的极限
limit(f, x, a);

% 当默认变量趋向于 a 时 f 的极限
limit(f, a);

% 计算 a=0 时的极限
limit(f);

% 计算右极限
limit(f, x, a, 'right');

% 计算左极限
limit(f, x, a, 'left');
```

## 计算导数

```matlab
% 求符号表达式 f 关于 v 的导数
g = diff(f, v);

% 求符号表达式 f 关于默认变量的导数
g = diff(f);

% 求 f 关于 v 的 n 阶导数
g = diff(f, v, n);
```


## 计算积分

```matlab
% 计算定积分
int(f, v, a, b);

% 计算关于默认变量的定积分
int(f, a, b);

% 计算不定积分
int(f, v);

% 计算关于默认变量的不定积分
int(f);
```

## 符号求和

```matlab
symsum(f, v, a, b)
```

## 代数方程求解

```matlab
% 求方程关于指定自变量的解
solve(f, v)
```

## 微分方程求解

```matlab
y = dsolve('eq1', 'eq2', ... , 'cond1', 'cond2', ... , 'v')
```

其中：

* `y` 为输出的解
* `eq1`、`eq2`、`...` 为微分方程
* `cond1`、`cond2`、`...` 为初值条件
* `v` 为自变量