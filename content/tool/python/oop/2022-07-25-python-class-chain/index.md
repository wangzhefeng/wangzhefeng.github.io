---
title: Python 链式调用
author: 王哲峰
date: '2022-07-25'
slug: python-class-chain
categories:
  - python
tags:
  - tool
---


# Python 链式调用

在 Python 中实现一个简单的链式调用就是通过构建方法并返回对象自身或返回归属类

```python
class Chain():

    def __init__(self, name):
        self.name = name
    
    def introduce(self):
        print("hello, my name is %s" % self.name)
        return self
    
    def talk(self):
        print("Can we make a friend?")
        return self
    
    def greet(self):
        print("Hey! How are you?")
        return self

if __name__ == "__main__":
    chain = Chain(name = "jobs")
    chain.introduce()
    print("-" * 20)
    chain.introduce().talk()
    print("-" * 20)
    chain.introduce().talk().greet()
```

# Pandas 链式调用

