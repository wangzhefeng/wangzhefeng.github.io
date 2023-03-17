---
title: Python 正则表达式
author: 王哲峰
date: '2023-01-09'
slug: re
categories:
  - Python
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

- [re 库介绍](#re-库介绍)
- [re 库使用](#re-库使用)
  - [re 库常量](#re-库常量)
    - [re 库源码](#re-库源码)
    - [re.IGNORECASE 使用](#reignorecase-使用)
    - [re.ASCII 使用](#reascii-使用)
    - [re.DOTALL 使用](#redotall-使用)
  - [re 库函数](#re-库函数)
  - [re 库异常](#re-库异常)
  - [正则对象 pattern](#正则对象-pattern)
- [参考](#参考)
</p></details><p></p>


# re 库介绍

`re` 库是 Python 处理文本的标准库

Python `re` 库主要定义了: 

- 9 个常量
- 12 个函数
- 1 个异常

# re 库使用

```python
import re
```

## re 库常量

`re` 库 中的常量表示不可更改的变量, 一般用于做标记. \ `re`
模块中有 9 个常量, 常量值都是 `int` 类型: 

- `re.ASCII` or `re.A`
- `re.IGNORECASE` or `re.I`
- `re.LOCALE` or `re.L`
- `re.UNICODE` or `re.U`
- `re. MULTILINE` or `re.M`
- `re.DOTALL` or `re.S`
- `re.VERBOSE` or `re.X`
- `re.TEMPLATE` or `re.T`
- `re.DEBUG`


### re 库源码 

```python
class RegexFlag(enum.IntFlag):
   ASCII = A = sre_compile.SRE_FLAG_ASCII # assume ascii "locale"
   IGNORECASE = I = sre_compile.SRE_FLAG_IGNORECASE # ignore case
   LOCALE = L = sre_compile.SRE_FLAG_LOCALE # assume current 8-bit locale
   UNICODE = U = sre_compile.SRE_FLAG_UNICODE # assume unicode "locale"
   MULTILINE = M = sre_compile.SRE_FLAG_MULTILINE # make anchors look for newline
   DOTALL = S = sre_compile.SRE_FLAG_DOTALL # make dot match newline
   VERBOSE = X = sre_compile.SRE_FLAG_VERBOSE # ignore whitespace and comments
   # sre extensions (experimental, don't rely on these)
   TEMPLATE = T = sre_compile.SRE_FLAG_TEMPLATE # disable backtracking
   DEBUG = sre_compile.SRE_FLAG_DEBUG # dump pattern after compilation

   def __repr__(self):
      if self._name_ is not None:
         return f're.{self._name_}'
      value = self._value_
      members = []
      negative = value < 0
      if negative:
         value = ~value
      for m in self.__class__:
         if value & m._value_:
               value &= ~m._value_
               members.append(f're.{m._name_}')
      if value:
         members.append(hex(value))
      res = '|'.join(members)
      if negative:
         if len(members) > 1:
               res = f'~({res})'
         else:
               res = f'~{res}'
      return res
   __str__ = object.
```

### re.IGNORECASE 使用

- 语法: 
   - `re.IGNORECASE` or `re.I`
- 作用: 
   - 忽略大小写匹配
- 代码: 

```python
text =  "Hello World."
pattern = r"Hello World."
print("默认模式: ", re.findall(pattern, text))
print("忽略大小写模式: ", re.findall(pattern, text, re.I))
```

### re.ASCII 使用

- 语法: 
   - `re.ASCII` or `re.A`
- 作用: 
   - 让 `\w`\ , \ `\W`\ , \ `\b`\ , \ `\B`\ , \ `\d`\ , \ `\D`\ , \ `\s`\ , \ `\S`
     只匹配 ASCII 编码支持的字符, 而不是 Unicode 编码支持的字符
- 代码: 

```python
text = "a测试b测试c"
pattern = r"\w+"
print("Unicode:", re.findall(pattern, text))
print("ASCII:", re.findall(pattern, text, re.A))
```

### re.DOTALL 使用

- 语法: 
   - `re.DOTALL` or `re.S`
- 作用: 
   - 让 `.` 匹配所有字符, 包括换行符
- 代码: 

```python
text = "测试\n测试"
pattern = r".*"
print("默认模式:", re.findall(pattern, text))
print(".匹配所有模式:", re.findall(pattern, text, re.S))
```

## re 库函数


## re 库异常


## 正则对象 pattern


# 参考

1. [re 模块官方文档](https://docs.python.org/zh-cn/3.8/library/re.html)
2. [re 模块库源码](https://github.com/python/cpython/blob/3.8/Lib/re.py)
3. [Python正则表达式](https://mp.weixin.qq.com/s/iZk1CX9VjCcHiXVOEwyWGg)
4. [正则表达式](https://mp.weixin.qq.com/s?__biz=MzI0OTc0MzAwNA==&mid=2247486276&idx=1&sn=ed050c9a691ffd828b86be9edfba73b4&chksm=e98d98b7defa11a1447aca8db5599100cbc8a19c1a8422dd6f84051df643eae1ce49510ef217&scene=21#wechat_redirect)