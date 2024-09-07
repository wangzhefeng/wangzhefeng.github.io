---
title: 随机漫步
author: 王哲峰
date: '2020-05-06'
slug: random-walk
categories:
  - 数学、统计学
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

- [随机漫步简介](#随机漫步简介)
- [Python 示例](#python-示例)
</p></details><p></p>

# 随机漫步简介

- [百度百科](https://baike.baidu.com/item/%E9%9A%8F%E6%9C%BA%E6%B8%B8%E8%B5%B0/1674146?fromtitle=%E9%9A%8F%E6%9C%BA%E6%BC%AB%E6%AD%A5&fromid=15578433&fr=aladdin)

# Python 示例

```python
import random

"""
随机漫步生成是无规则的，是系统自行选择的结果.
根据设定的规则自定生成，上下左右的方位，每次所经过的方向路径.
"""

class Randomwalk():
    """
    一个生成随机数漫步的类
    """
    def __init__(self, num_point = 1000):
        """
        初始化随机漫步的属性
        """
        self.num_point = num_point
        # 所有随机漫步的开始都是坐标 [0, 0]
        self.x_lab = [0]
        self.y_lab = [0]

    def fill_walk(self):
        """
        计算随机漫步的所有点
        """
        while len(self.x_lab) < self.num_point:
            # 决定前进的方向以及前进的距离
            # x 方向
            x_direction = random.choice([1, -1])
            x_distance = random.choice([0, 1, 2, 3, 4])
            x_step = x_direction * x_distance
            # print(f"x_step = {x_step}")
            # y 方向
            y_direction = random.choice([1, -1])
            y_distance = random.choice([0, 1, 2, 3, 4])
            y_step = y_direction * y_distance
            # print(f"y_step = {y_step}")
            # 拒绝原地不动
            if x_step == 0 and y_step == 0:
                continue
            # 计算下一个点 x 和 y 的值
            next_x = self.x_lab[-1] + x_step
            next_y = self.y_lab[-1] + y_step
            self.x_lab.append(next_x)
            self.y_lab.append(next_y)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    rw = Randomwalk()
    rw.fill_walk()
    
    # (1)绘制随机漫步图
    plt.scatter(rw.x_lab, rw.y_lab, s = 1)
    plt.show()
    
    # (2)给点着色
    while True:
        # 随机游走
        rw = Randomwalk()
        rw.fill_walk()
        #设置绘画窗口大小
        plt.figure(dpi = 128, figsize = (10, 6))
        #隐藏坐标轴
        plt.axes().get_xaxis().set_visible(False)
        plt.axes().get_yaxis().set_visible(False)
        # 绘制散点图，并突出起点(0,0)和终点
        plt.scatter(rw.x_lab, rw.y_lab, c = list(range(rw.num_point)), cmap = plt.cm.YlOrBr, edgecolors = None, s = 15)
        plt.scatter(0, 0, c = 'green', edgecolors = None, s = 100)
        plt.scatter(rw.x_lab[-1], rw.y_lab[-1], c = 'red', edgecolors = None, s = 100)
        # 显示
        plt.show()
        # 交互
        keep_running = input("Make another walk?(y/n): ")
        keep_running = keep_running.lower()
        if keep_running == 'n':
            break
```