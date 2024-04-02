---
title: Python 基础数据
author: 王哲峰
date: '2022-07-26'
slug: python-data-basic
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
img {
    pointer-events: none;
}
</style>

<details><summary>目录</summary><p>

- [Python String](#python-string)
   - [字典中的键映射多个值](#字典中的键映射多个值)
</p></details><p></p>

# Python String

```python
# 字符串方法
mystr = "\t WangZhefeng \n"
print(mystr.lower())
print(mystr.upper())
print(mystr.isalpha())
print(mystr.isdigit())
print(mystr.find("Zhe"))
print(mystr.replace("Zhe", "zhe"))
print(mystr.strip())
print(mystr.rstrip())
print(mystr.lstrip())

mystr2 = "aaa,bbb,ccc"
print(mystr2.split(","))
mystr3 = "a b\nc\nd"
print(mystr3.split())
print(mystr3.splitlines())
print("Ni".join(["aaa", "bbb", "ccc"]))
print(' '.join(["A", "dead", "parrot"]))
print(string.ascii_lowercase)
print(string.ascii_uppercase)
print(string.whitespace)

mystr = "xxaaxxaa"
print("SPAM".join(mystr.split("aa")))
print(int("42"))
print(eval("42"))
print(str(42))
print(repr(42))
print(("%d" % 42))
print("{:d}".format(42))
print("42" + str(1))
print(int("42") + 1)
```

## 字典中的键映射多个值

普通做法: 

```python
c = {
   "a": [1, 2, 3],
   "b": [4, 5]
}

d = {
   "a": (1, 2, 3),
   "b": (4, 5)
}

e = {
   "a": {1, 2, 3},
   "b": {4, 5}
}
```

defaultdict:

```python
from collections import defaultdict

c = defaultdict(list)
c["a"].append(1)
c["a"].append(2)
c["a"].append(3)
c["b"].append(4)
c["b"].append(5)

d = defaultdict(set)
d["a"].add(1)
d["a"].add(2)
d["a"].add(3)
d["b"].add(4)
d["b"].add(5)

e = {}
e.setdefault("a", []).append(1)
e.setdefault("a", []).append(2)
e.setdefault("b", []).append(4)
```

比较: 

```python
# 普通做法
d = {}
for key, value in pairs:
   if key not in d:
      d[key] = []
   d[key].append(value)

# defaultdict
d = defaultdict(list)
for key, value in pairs:
   d[key].append(value)
```

