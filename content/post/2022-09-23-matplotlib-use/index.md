---
title: Matplotlib
author: 王哲峰
date: '2022-09-23'
slug: matplotlib-use
categories:
  - data analysis
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

- [快速开始](#快速开始)
- [一张统计图的结构](#一张统计图的结构)
- [图形 API](#图形-api)
- [Subplots layout](#subplots-layout)
    - [API](#api)
    - [subplots](#subplots)
- [基本图形](#基本图形)
    - [plot](#plot)
    - [scatter](#scatter)
    - [bar](#bar)
    - [imshow](#imshow)
    - [contour](#contour)
    - [pcolormesh](#pcolormesh)
    - [quiver](#quiver)
    - [pie](#pie)
    - [text](#text)
    - [fill](#fill)
- [高级图形](#高级图形)
    - [step](#step)
    - [boxplot](#boxplot)
    - [errorbar](#errorbar)
    - [hist](#hist)
    - [violinplot](#violinplot)
    - [barbs](#barbs)
    - [eventplot](#eventplot)
    - [hexbin](#hexbin)
- [Scales](#scales)
- [Projections](#projections)
- [Lines](#lines)
- [Markers](#markers)
- [Colors](#colors)
- [Colormaps](#colormaps)
- [Tick locators](#tick-locators)
- [Tick formatters](#tick-formatters)
- [Ornaments](#ornaments)
- [Event handling](#event-handling)
- [Animation](#animation)
- [Styles](#styles)
- [Quick reminder](#quick-reminder)
- [1.安装](#1安装)
- [2.使用库](#2使用库)
- [3.Figure](#3figure)
    - [3.1 Figure class](#31-figure-class)
    - [3.2 Axes class](#32-axes-class)
    - [3.3 Axis class](#33-axis-class)
    - [3.4 Artist class](#34-artist-class)
- [4.函数输入格式](#4函数输入格式)
- [5.面向对象接口、pyplot 接口、GUI应用程序方式](#5面向对象接口pyplot-接口gui应用程序方式)
    - [5.1 面向对象接口](#51-面向对象接口)
    - [5.2 pyplot 接口](#52-pyplot-接口)
    - [5.3 GUI 应用程序中嵌入Matplotlib](#53-gui-应用程序中嵌入matplotlib)
    - [5.4 最佳实践](#54-最佳实践)
- [6.一个简单的🌰](#6一个简单的)
- [7.pyplot 接口](#7pyplot-接口)
    - [7.1 pyplot.plot](#71-pyplotplot)
    - [7.2 plot style](#72-plot-style)
    - [7.3 plot keyword string](#73-plot-keyword-string)
    - [7.4 plot categorical variables](#74-plot-categorical-variables)
    - [7.5 line properties](#75-line-properties)
    - [7.6 多个 figures 和 axes](#76-多个-figures-和-axes)
    - [7.7 处理文本](#77-处理文本)
    - [7.8 对数轴、非线性轴](#78-对数轴非线性轴)
- [8.Image](#8image)
    - [8.1 将 image 数据转换为 Numpy array](#81-将-image-数据转换为-numpy-array)
    - [8.2 将 Numpy array 绘制成图片](#82-将-numpy-array-绘制成图片)
    - [8.2 将伪彩色方案应用于图像](#82-将伪彩色方案应用于图像)
    - [8.3 色标参考](#83-色标参考)
    - [8.4 检查特定数据范围](#84-检查特定数据范围)
    - [8.5 数组插值](#85-数组插值)
- [9.一个 Plot 的生命周期](#9一个-plot-的生命周期)
- [10.Matplotlib 个性化](#10matplotlib-个性化)
    - [10.1 rcParams](#101-rcparams)
    - [10.2 style sheets](#102-style-sheets)
    - [10.3 matplotlibrc file](#103-matplotlibrc-file)
</p></details><p></p>

# 快速开始

```python
def quick_start():
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    X = np.linspace(0, 2 * np.pi, 100)
    Y = np.cos(X)

    fig, ax = plt.subplots()
    ax.plot(X, Y, color = "green")

    fig.savefig(
        os.path.join(
            os.path.dirname(__file__), 
            "images/figure.png"
        )
    )
    fig.show()

quick_start()
```

<img src="images/quick_start.png" width="100%" />

# 一张统计图的结构

<img src="images/anatomy.png" width="100%" />

# 图形 API

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
```

* 图形
    - Figure: `fig`
    - Axes: `fig.subplots` 
    - Line: `ax.plot`
    - Markers: `ax.scatter`
    - Grid: `ax.grid`
    - Legend: `ax.legend`
    - Spine: `ax.spines`
* 标题 
    - Title: `ax.set_title`
* Y 轴
    - y Axis: `ax.yaxis`
    - ylabel: `ax.set_ylabel`
    - Major tick: `ax.yaxis.set_major_locator`
    - Major tick label: `ax.yaxis.set_major_formatter`
    - Minor tick: `ax.yaxis.set_minor_locator`
* X 轴
    - x Axis: `ax.xaxis`
    - xlabel: `ax.set_xlabel`
    - Minor tick label: `ax.xaxis.set_minor_formatter`

# Subplots layout

## API

<img src="images/subplots_layout_api.png" width="70%" />

## subplots

```python
def subplots_layout():
    fig, axs = plt.subplots(3, 3)
    
    fig.savefig(os.path.join(os.path.dirname(__file__), 
        "images/subplots_layout.png"))
    fig.show()
```

![img](images/subplots_layout.png)

* gridsepc
* inset_axes
* make_axes_locatable

# 基本图形

## plot

```python
plot([X], Y, [fmt], color, marker, linestyle)
```

## scatter

```python
scatter(X, Y, [s]izes, [c]olors, markers, cmap)
```

## bar

```python
bar[h](x, height, width, bottom, align, color)
```

## imshow

```python
imshow(Z, cmap, interpolation, extent, origin)
```

## contour

```python
contour[f]([X], [Y], Z, levels, colors, extent, origin)
```

## pcolormesh

```python
pcolormesh([X], [Y], Z, vmin, vmax, cmap)
```

## quiver

```python
quiver([X], [Y], U, V, C, units, angles)
```

## pie

```python
pie(Z, explode, labels, colors, raidus)
```

## text

```python
text(x, y, text, va, ha, size, weight, transform)
```

## fill

```python
fill[_between][x](X, Y1, Y2, color, where)
```

# 高级图形

## step

```python
step(X, Y, [fmt], color, marker, where)
```

## boxplot

```python
boxplot(X, notch, sym, bootstrap, widths)
```

## errorbar

```python
errorbar(X, Y, xerr, yerr, fmt)
```

## hist

```python
hist(X, bins, range, density, weights)
```

## violinplot

```python
violinplot(D, positions, widths, vert)
```

## barbs

```python
barbs([X], [Y], U, V, C, length, pivot, sizes)
```

## eventplot

```python
eventplot(positions, orientation, lineoffsets)
```

## hexbin

```python
hexbin(X, Y, C, gridsize, bins)
```

# Scales

`ax.set_[xy]scale(scale, ...)`

* linear
* log
* symlog
* logit

# Projections

`subplot(..., projection = p)`

* p = "polar"
* p = "3d"
* p = Orthographic() `from cartopy.crs import Cartographic`

# Lines

* `linestyle` or `ls`
    - "-"
    - ":"
    - "--"
    - "-."
    - (0, (0.01, 2))
* `capstyle` or `dash_capstyle`
    - "butt"
    - "round"
    - "projecting"

# Markers

marker:

* "."
* "o"
* "s"
* "P"
* "X"
* "*"
* "p"
* "D"
* "<"
* ">"
* "^"
* "v"
* "1"
* "2"
* "3"
* "4"
* "+"
* "x"
* "|"
* "_"
* 4
* 5
* 6
* 7

markevery:

* 10
* [0, -1]
* (25, 5)
* [0, 25, -1]

# Colors

* "Cn"
* "x"
* "name"
* (R, G, B[, A])
* "#RRGGBB[AA]"
* "x.y"

# Colormaps

* `plt.get_cmap(name)`
    - Uniform
        - viridis
        - magma
        - plasma
    - Sequential
        - Greys
        - YlOrBr
        - Wistia
    - Diverging
        - Spectral
        - coolwarm
        - RdGy
    - Quanlitative
        - tab10
        - tab20
    - Cyclic
        - twilight

# Tick locators

```python
from matplotlib import ticker

ax.[xy]axis.set_[minor|major]_locator(locator)
```

`locator`:

* ticker.NullLocator()
* ticker.MultipleLocator(0.5)
* ticker.LinearLocator(numticks = 3)
* ticker.IndexLocator(base = 0.5, offset = 0.25)
* ticker.AutoLocator()
* ticker.MaxNLocator(n = 4)
* ticker.LogLocator(base = 10, numticks = 15)

# Tick formatters

```python
from matplotlib import ticker

ax.[xy]axis.set_[minor|major]_formatter(formatter)
```

`formatter`:

* ticker.NullFormatter()
* ticker.FixedFormatter(["zeor", "one", "two", ...])
* ticker.FuncFormatter(lambda x, pos: "[%.2f]" % x)
* ticker.FormatStrFormatter(">%d<")
* ticker.ScalarFormatter()
* ticker.StrMethodFormatter("{x}")
* ticker.PercentFormatter(xmax = 5)

# Ornaments

* `ax.legend(...)`
* ax.colorbar()
* ax.annotate()

# Event handling

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

def on_click(event):
    print(event)

fig.canvas.mpl_connect("button_press_event", on_click)
```

# Animation

```python
import matplotlib.animation as mpla

T = np.linspace(0, 2 * np.pi, 100)
S = np.sin(T)
line, = plt.plot(T, S)

def animate(i):
    line.set_ydata(np.sin(T + i / 50))

anim = mpla.FuncAnimation(plt.gcf(), animate, interval = 5)
plt.show()
```

# Styles


```python
plt.style.use(style)
```

style:

* default
* classic
* grayscale
* ggplot
* seaborn
* fast
* bmh
* Solarize_Light2
* seaborn-notebook

# Quick reminder

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ax.grid()
ax.set_[xy]lim(vmin, vmax)
ax.set_[xy]label(label)
ax.set_[xy]ticks(ticks, [labels])
ax.set_[xy]ticklabels(labels)
ax.set_title(title)
ax.tick_params(width = 10, ...)
ax.set_axis_[on|off]()

fig.suptitle(title)

fig.tight_layout()

plt.gcf(), plt.gca()

mpl.rc("axes", linewidth = 1, ...)

[fig|ax].patch.set_alpha(0)

text = r"$frac{-e^{i\pi}}{2^n}"
```

# 1.安装

```shell
pip install matplotlib
```

# 2.使用库

```python
import matplotlib.pyplot as plt
import numpy as np
```

# 3.Figure

> - image 图像
> - graph 图形
> - aritst 可视化元素
> - figure 图形
> - canvas 画布
>
> - axes 数据在图像中的区域
> - data 数据
> 	- plot
> 		- line
> 		- marker
> - Spines 边框
> - axis 坐标轴、坐标轴限制、坐标轴刻度、坐标轴标签、坐标轴刻度标签
> - grid 背景网格
> - legend 图例
> - title 标题

![image-20211201220610305](/Users/zfwang/Library/Application Support/typora-user-images/image-20211201220610305.png)

## 3.1 Figure class

- Figure
	- Axes
	- Artist
	- canvas

```python
fig = plt.fiure()			  # an empty figure with no Axes
fig, ax = plt.subplots()      # a figure with a single Axes
fig, axs = plt.subplots(2, 2) # a figure with a 2x2 grid of Axes
```

## 3.2 Axes class

- Figure

	- Axes: a plot：data 在 image 中的区域

		- title: `axes.Axes.set_title()`
		- xlim: `axes.Axes.set_xlim()`

		- ylim: `axes.Axes.set_ylim()`

		- x-label: `axes.Axes.set_xlabel()`

		- y-label: `axes.Axes.set_ylabel()`

## 3.3 Axis class

- Figure
	- Axes
		- Axis
			- 坐标轴(Axis)
				- X axis
				- Y axis
				- ...
			- 坐标轴标签(Axis label)
				- X axis label
				- Y axis label
			- 坐标轴限制(Axis limit)
				- X axis limit
				- Y axis limit
			- 坐标轴刻度(Tick)
				- 主刻度(Major tick)
				- 副刻度(Minor tick)
				- 刻度位置 Locator
			- 坐标轴刻度标签(Tick label)
				- 主刻度标签(Major tick label)
				- 副刻度标签(Minor tick label)
				- 刻度标签格式 Formatter

## 3.4 Artist class

- 图形中可见的东西都是一个 Artist，包括 Figure、Axes、Axis、Text、Line2D、collections、Patch等对象
- 当一个 figure(图形) 被渲染时，所有的 artist 都被画在 canvas 上
- Artist
	- Figure
		- Axes
			- Axis
			- Text
			- Line2D
			- collections
			- Pathc

# 4.函数输入格式

- numpy.array
	- pandas data object => np.array
	- numpy.matrix => np.array

```python
# pandas.DataFrame 转换
pandas_dataframe = pandas.DataFrame()
array_inputs = pandas_datarframe.values

# numpy.matrix 转换
numpy_matrix = numpy.matrix([1, 2], [3, 4])
array_inputs = numpy.asarray(numpy_matrix)
```

- numpy.ma.masked_array

# 5.面向对象接口、pyplot 接口、GUI应用程序方式

> Matplotlib 的文档和示例同时使用 OO 和 pyplot 方法（它们同样强大），您可以随意使用其中任何一种（但是，最好选择其中之一并坚持使用，而不是混合使用它们）。通常，我们建议将 pyplot 限制为交互式绘图（例如，在 Jupyter 笔记本中），并且更喜欢 OO 风格的非交互式绘图（在旨在作为更大项目的一部分重用的函数和脚本中） .

## 5.1 面向对象接口

- Explicitly create figures and axes, and call methods on them (the "object-oriented (OO) style")
	- plt.subplots()
	- ax.plot()
	- ax.set_xlabel()
	- ax.set_ylabel()
	- ax.set_title()
	- ax.legend()

```python
# data
x = np.linspace(0, 2, 100)

# 创建 Figure class 的 fig 实例
# 创建 Axes class 的  ax 实例
fig, ax = plt.subplots()
ax.plot(x, x, label = "linear")
ax.plot(x, x ** 2, label = "quadratic")
ax.plot(x, x ** 3, label = "cubic")
ax.set_xlabel("x label")
ax.set_ylabel("y label")
ax.set_title("Simple Plot")
ax.legend()
```

![image-20211201234013916](/Users/zfwang/Library/Application Support/typora-user-images/image-20211201234013916.png)

## 5.2 pyplot 接口

- Rely on pyplot to automatically create and manage the figures and axes, and use pyplot functions for plotting.

```python
# data
x = np.linspace(0, 2, 100)

plt.figure(figsize(8, 8))
plt.plot(x, x, label = "linear")
plt.plot(x, x ** 2, label = "quadratic")
plt.plot(x, x ** 3, label = "cubic")
plt.xlabel("x label")
plt.ylabel("y label")
plt.title("Simple Plot")
plt.legend()
```

![image-20211201234013916](/Users/zfwang/Library/Application Support/typora-user-images/image-20211201234013916.png)

## 5.3 GUI 应用程序中嵌入Matplotlib

- 略

## 5.4 最佳实践

- 用不同数据绘制同样的图片
- 方法

```python
def my_plotter(ax, data1, data2, param_dict):
    """
    A helper function to make a graph

    Parameters
    ----------
    ax : Axes
        The axes to draw to

    data1 : array
       The x data

    data2 : array
       The y data

    param_dict : dict
       Dictionary of keyword arguments to pass to ax.plot

    Returns
    -------
    out : list
        list of artists added
    """
   	out = ax.plot(data1, data2, **param_dict)
    return out
```

- 使用

```python
# data
data1, data2, data3, data4 = np.random.randn(4, 100)
# plot
fig, ax = plt.subplots(1, 1)
my_plotter(ax, data1, data2, {"marker": "x"})
```

![image-20211201235123195](/Users/zfwang/Library/Application Support/typora-user-images/image-20211201235123195.png)

```python
# data
data1, data2, data3, data4 = np.random.randn(4, 100)
# plot
fig, ax = plt.subplots(1, 1)
my_plotter(ax, data1, data2, {"marker": "x"})
my_plotter(ax, data3, data4, {"marker": "o"})
```

![image-20211201235141268](/Users/zfwang/Library/Application Support/typora-user-images/image-20211201235141268.png)

# 6.一个简单的🌰

- 方法1

```python
# 面向对象 API
fig, ax = plt.subplots()
ax.plot(
	[1, 2, 3, 4],
    [1, 4, 2, 3]
)
```

![image-20211201203826072](/Users/zfwang/Library/Application Support/typora-user-images/image-20211201203826072.png)

- 方法2

```python
# pyplot API
plt.plot(
	[1, 2, 3, 4],
    [1, 4, 2, 3]
)
plt.show()
```

![image-20211201203834922](/Users/zfwang/Library/Application Support/typora-user-images/image-20211201203834922.png)

# 7.pyplot 接口

- pyplot 是使 matplotlib 像 MATLAB 一样工作的函数集合。每个 `pyplot` 函数都会对图形进行一些更改：例如，创建图形、在图形中创建绘图区域、在绘图区域中绘制一些线条、用标签装饰绘图等

- 在 pyplot 函数调用中保留各种状态，以便跟踪当前图形(figure)和绘图区域(plotting area)等内容，并且绘图函数指向当前轴(axes)
- pyplot 接口方法
	- fig, ax = plt.subplots()：面向对象中创建 figure, axes 实例
	- plt.figure(figsize = ())
	- plt.subplot()
	- plt.bar()
	- plt.scatter()
	- plt.plot()
	- plt.suptitle()
	- plt.title()
	- plt.legend()
	- plt.axis()
	- plt.xlim()
	- plt.ylim()
	- plt.xlabel()
	- plt.ylabel()
	- plt.show()

## 7.1 pyplot.plot

```python
plt.plot([1, 2, 3, 4])   # y = [1, 2, 3, 4] x = [0, 1, 2, 3]
plt.ylabel("some number")
plt.show()
```

![image-20211202000553898](/Users/zfwang/Library/Application Support/typora-user-images/image-20211202000553898.png)

```python
plt.plot(
    [1, 2, 3, 4], 
    [1, 4, 9, 16]
)  # y = [1, 4, 9, 16] x = [1, 2, 3, 4]
plt.ylabel("some number")
plt.show()
```

![image-20211202000605427](/Users/zfwang/Library/Application Support/typora-user-images/image-20211202000605427.png)

## 7.2 plot style

```python
t = np.arange(0., 5., 0.2)
plt.plot(
    t, t, "r--", 
    t, t**2, "bs", 
    t, t**3, "g^"
)
plt.axis([0, 5, 0, 100])  # [xmin, xmax, ymin, ymax]
# plt.xlim([0, 5])
# plt.ylim(0, 100)
plt.show()
```

![image-20211202001450159](/Users/zfwang/Library/Application Support/typora-user-images/image-20211202001450159.png)

## 7.3 plot keyword string

```python
# data
data = {
    "a": np.arange(50),
    "c": np.random.randint(0, 50, 50),
    "d": np.random.randn(50)
}
data["b"] = data["a"] + 10 * np.random.randn(50)
data["d"] = np.abs(data["d"]) * 100

# plot
plt.scatter("a", 
            "b", 
            c = "c",  # color
            s = "d",  # size
            data = data)
plt.xlabel("entry a")
plt.ylabel("entry b")
plt.show()
```

![image-20211202002055655](/Users/zfwang/Library/Application Support/typora-user-images/image-20211202002055655.png)

## 7.4 plot categorical variables

```python
names = ["group_a", "group_b", "group_c"]
values = [1, 10, 100]

plt.figure(figsize = (9,  3))

plt.subplot(131)
plt.bar(names, values)

plt.subplot(132)
plt.scatter(names, values)

plt.subplot(133)
plt.plot(names, values)

plt.suptitle("Categorical Plotting")
plt.show() 
```

![image-20211202003338437](/Users/zfwang/Library/Application Support/typora-user-images/image-20211202003338437.png)

## 7.5 line properties

- 使用关键字参数设置 line properties

```python
plt.plot(x, y, linewidth = 2.0)
```

- 使用 Line2D setter 方法设置 line properties

```python
line, = plt.plot(x, y, "-")
line.set_antialiased(False)
```

- 使用 setp 设置 line properties

```python
lines = plt.plot(x1, y1, x2, y2)

# keyword arguments
plt.setp(lines, color = "r", linewidth = 2.0)

# MATLAB style
plt.setp(lines, "color", "r", "linewidth", 2.0)

# line properties
plt.setp(lines)
```

## 7.6 多个 figures 和 axes

- MATPLOT 和 pyplot 具有当前窗口 figure、当前坐标区 axes 的概念，所有绘图函数都适用于当前 axes
	- gca 函数返回当前轴, matplotlib.axes.Axes 实例
	- gcf 返回当前图形, matplotlib.figure.Figure 实例

```python
def f(t):
    return np.exp(-t) * np.cos(2 * np.pi * t)

t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)

plt.figure()  # 可选
plt.subplot(221)
plt.plot(
    t1, f(t1), "bo", 
    t2, f(t2), "k",
)

plt.subplot(212)
plt.plot(t2, np.cos(2 * np.pi * t2), "r--")

plt.show()
```



```python
plt.figure(1)
plt.subplot(211)
plt.plot([1, 2, 3])
plt.subplot(212)
plt.plot([4, 5, 6])

plt.figure(2)
plt.plot([4, 5, 6])

plt.figure(1)
plt.subplot(211)
plt.title("Easy as 1, 2, 3")
```

## 7.7 处理文本

- APIs
	- text
	- xlabel
	- ylabel
	- title
	- suptitle

```python
# data
mu, sigma = 100, 15
x = mu + sigma * np.random.randn(10000)

# histogram
n, bins, patches = plt.hist(x, 50, density = 1, facecolor = "g", alpha = 0.75)

plt.xlabel("Smarts")
plt.ylabel("Probability")
plt.title("Histogram of IQ")
plt.text(60, 0.025, r'$\mu=100,\ \sigma=15$')
plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.show()
```

- 文本注释

```python
ax = plt.subplot()

t = np.arange(0.0, 5.0, 0.01)
s = np.cos(2 * np.pi * t)

line = plt.plot(t, s, lw = 2)

plt.annotate(
    "local max", 
    xy = (2, 1), 
    xytext = (3, 1,5),
    arrowprops = dict(facecolor = "black", shrink = 0.5)
)
plt.ylim(-2, 2)
plt.show()
```



## 7.8 对数轴、非线性轴

```python
np.random.seed(19680801)

# data
y = np.random.normal(loc = 0.5, scale = 0.4, size = 1000)
y = y[(y > 0) & (y < 1)]
y.sort()
x = np.arange(len(y))

# plot with various axes scales
plt.figure()

# linear
plt.subplot(221)
plt.plot(x, y)
plt.yscale("linear")
plt.title("linear")
plt.grid(True)

# log
plt.subplot(222)
plt.plot(x, y)
plt.yscale("log")
plt.title("log")
plt.grid(True)

# symmetric log
plt.subplot(223)
plt.plot(x, y - y.mean())
plt.yscale("symlog", linthresh = 0.01)
plt.title("symlog")
plt.grid(True)

# logit
plt.subplot(224)
plt.plot(x, y)
plt.yscale("logit")
plt.title("logit")
plot.grid(True)

plt.subplots_adjust(
	top = 0.92,
    bottom = 0.08,
    left = 0.10,
    right = 0.95,
    hspace = 0.25,
    wspace = 0.35
)
plt.show()
```

# 8.Image

- Matplotlib 依赖 Pillow 库导入图片数据

```python
import matplotlib.image as mpimg
```

## 8.1 将 image 数据转换为 Numpy array

![../../_images/stinkbug.png](https://matplotlib.org/stable/_images/stinkbug.png)

- 图片信息
	- 24-bit RGB PNG image(8 big for each R, G, B)
- 其他格式图片
	- RGBA(允许透明度 transparency)图片
	- 单通道灰度(single-channel grayscale, luminosity)图片
- 图片 array 数据
	- dtype: float32
	- matplotlib 将每个通道的 8 位数据重新缩放为 0.0 和 1.0 之间的浮点数据
	- Pillow 可以使用的唯一数据类型是 uint8
	- matplotlib 绘图可以处理 float32 和 uint8，但对 PNG 以外的任何格式的图像读取、写入仅限于 uint8，大多数显示器只能呈现每个通道 8 位的颜色等级，因为人眼所能看到的只有 8 位

```python
img = mpimg.imread("stinkbug.png")
print(img)
print(img.shape)

[[[0.40784314 0.40784314 0.40784314]
  [0.40784314 0.40784314 0.40784314]
  [0.40784314 0.40784314 0.40784314]
  ...
  [0.42745098 0.42745098 0.42745098]
  [0.42745098 0.42745098 0.42745098]
  [0.42745098 0.42745098 0.42745098]]

 [[0.4117647  0.4117647  0.4117647 ]
  [0.4117647  0.4117647  0.4117647 ]
  [0.4117647  0.4117647  0.4117647 ]
  ...
  [0.42745098 0.42745098 0.42745098]
  [0.42745098 0.42745098 0.42745098]
  [0.42745098 0.42745098 0.42745098]]

 [[0.41960785 0.41960785 0.41960785]
  [0.41568628 0.41568628 0.41568628]
  [0.41568628 0.41568628 0.41568628]
  ...
  [0.43137255 0.43137255 0.43137255]
  [0.43137255 0.43137255 0.43137255]
  [0.43137255 0.43137255 0.43137255]]

 ...

 [[0.4392157  0.4392157  0.4392157 ]
  [0.43529412 0.43529412 0.43529412]
  [0.43137255 0.43137255 0.43137255]
  ...
  [0.45490196 0.45490196 0.45490196]
  [0.4509804  0.4509804  0.4509804 ]
  [0.4509804  0.4509804  0.4509804 ]]

 [[0.44313726 0.44313726 0.44313726]
  [0.44313726 0.44313726 0.44313726]
  [0.4392157  0.4392157  0.4392157 ]
  ...
  [0.4509804  0.4509804  0.4509804 ]
  [0.44705883 0.44705883 0.44705883]
  [0.44705883 0.44705883 0.44705883]]

 [[0.44313726 0.44313726 0.44313726]
  [0.4509804  0.4509804  0.4509804 ]
  [0.4509804  0.4509804  0.4509804 ]
  ...
  [0.44705883 0.44705883 0.44705883]
  [0.44705883 0.44705883 0.44705883]
  [0.44313726 0.44313726 0.44313726]]]

(375, 500, 3)
```

## 8.2 将 Numpy array 绘制成图片

```python
imgplot = plt.imshow(img)
```

![image-20211204001730928](/Users/zfwang/Library/Application Support/typora-user-images/image-20211204001730928.png)

## 8.2 将伪彩色方案应用于图像

- 伪彩色可以使图像增强对比度和更轻松地可视化数据
- 伪彩色仅与单通道、灰度、亮度图像有关

```python	
lum_img = img[:, :, 0]
plt.imshow(lum_img)
```

![image-20211204003701542](/Users/zfwang/Library/Application Support/typora-user-images/image-20211204003701542.png)

```python
lum_img = img[:, : 0]
plt.imshow(lum_img, cmap = "hot")
```

![image-20211204003910446](/Users/zfwang/Library/Application Support/typora-user-images/image-20211204003910446.png)

```python
lum_img = img[:, :, 0]
imgplot = plt.imshow(lum_img)
imgplot.set_cmap("nipy_spectral")
```

![image-20211204004118731](/Users/zfwang/Library/Application Support/typora-user-images/image-20211204004118731.png)

## 8.3 色标参考

```python
lum_img = img[:, : 0]
imgplot = plt.imshow(lum_img)
plt.colorbar()
```

![image-20211204004308940](/Users/zfwang/Library/Application Support/typora-user-images/image-20211204004308940.png)

## 8.4 检查特定数据范围

```python
lum_img = img[:, :, 0]
plt.hist(lum_img.ravel(), bins = 256, range(0.0, 1.0), fc = "k", ec = "k")
```

![image-20211204005900327](/Users/zfwang/Library/Application Support/typora-user-images/image-20211204005900327.png)

```python
# 调增上限，有效放大直方图中的一部分
lum_img = img[:, :, 0]
imgplot = plt.imshow(lum_img, clim = (0.0, 0.7))
```

![image-20211204010057096](/Users/zfwang/Library/Application Support/typora-user-images/image-20211204010057096.png)

```python
lum_img = img[:, :, 0]

fig = plt.figure()

ax = fig.add_subplot(1, 2, 1)
imgplot = plt.imshow(lum_img)
ax.set_title('Before')
plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')

ax = fig.add_subplot(1, 2, 2)
imgplot = plt.imshow(lum_img)
imgplot.set_clim(0.0, 0.7)
ax.set_title('After')
plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
```

![image-20211204011743914](/Users/zfwang/Library/Application Support/typora-user-images/image-20211204011743914.png)

## 8.5 数组插值

```python
from PIL import Image

img = Image.open("stinkbug.png")
img.thumbnail((64, 64), Image.ANTIALIAS) # resizes image in-replace
imgplot = plt.imshow(img)
```

![image-20211204011041795](/Users/zfwang/Library/Application Support/typora-user-images/image-20211204011041795.png)

```python
from PIL import Image

img = Image.open("stinkbug.png")
img.thumbnail((64, 64), Image.ANTIALIAS) # resizes image in-replace
imgplot = plt.imshow(img, interpolation = "nearest")
```

![image-20211204011222797](/Users/zfwang/Library/Application Support/typora-user-images/image-20211204011222797.png)

```python
from PIL import Image

img = Image.open("stinkbug.png")
img.thumbnail((64, 64), Image.ANTIALIAS) # resizes image in-replace
imgplot = plt.imshow(img, interpolation = "bicubic")
```

![image-20211204011514588](/Users/zfwang/Library/Application Support/typora-user-images/image-20211204011514588.png)

# 9.一个 Plot 的生命周期

```python
import numpy as np
import matplotlib.pyplot as plt
print(plt.style.available)
plt.style.use("fivethirtyeight")
plt.rcParams.update({
    "figure.autolayout": True,
})

# data
data = {
    'Barton LLC': 109438.50,
    'Frami, Hills and Schmidt': 103569.59,
    'Fritsch, Russel and Anderson': 112214.71,
    'Jerde-Hilpert': 112591.43,
    'Keeling LLC': 100934.30,
    'Koepp Ltd': 103660.54,
    'Kulas Inc': 137351.96,
    'Trantow-Barrows': 123381.38,
    'White-Trantow': 135841.99,
    'Will LLC': 104437.60
}
group_data = list(data.values())
group_names = list(data.keys())
group_mean = np.mean(group_data)

# plot
def currency(x, pos):
    """
    The two arguments are the value and tick position

    Args:
    x ([type]): [description]
    pos ([type]): [description]
    """
    if x >= 1e6:
        s = "${:1.1f}M".format(x * 1e-6)
    else:
        s = "${:1.0f}K".format(x * 1e-3)
    return s

fig, ax = plt.subplots(figsize = (8, 4))
# bar
ax.barh(group_names, group_data)
labels = ax.get_xticklabels()
plt.setp(labels, rotation = 45, horizontalalignment = "right")
# vertical line
ax.axvline(group_mean, ls = "--", color = "r")
# group text
for group in [3, 5, 8]:
    ax.text(
        145000, 
        group, 
        "New Company", 
        fontsize = 10, 
        verticalalignment = "center"
    )
# 标题设置
ax.title.set(y = 1.05)
# 设置X轴限制、X轴标签、Y轴标签、主标题
ax.set(
    xlim = [-10000, 140000], 
    xlabel = "Total Revenue", 
    ylabel = "Company", 
    title = "Company Revenue"
)
# 设置X轴主刻度标签格式
ax.xaxis.set_major_formatter(currency)
# 设置X轴主刻度标签
ax.set_xticks([0, 25e3, 50e3, 75e3, 100e3, 125e3])
# 微调fig
fig.subplots_adjust(right = 0.1)
# 图片保存
print(fig.canvas.get_supported_filetypes())
fig.savefig("sale.png", transparent = False, dpi = 80, bbox_inches = "tight")
plt.show()
```

# 10.Matplotlib 个性化

- rcParams
- style sheets
- matplollibrc file

> 优先级： rcParams > style sheets > matplotlibrc file

## 10.1 rcParams

- rc: runtime configuration
	- 可以动态改变默认 rc 设置
	- rc 设置保存在类字典的变量中 matplotlib.rcParams
	- rc 设置对于 matplotlib 库是全局的
	- rc 设置可以直接修改

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
mpl.rcParams["lines.linewidth"] = 2
mpl.rcParams["lines.linestyle"] = "--"
mpl.rcParams["axes.porp_cycle"] = cycler(color = ["r", "g", "b", "y"])
mpl.rc("lines", linewidth = 4, linestyle = "-.")
# 临时 rc 设置
with mpl.rc_context({"lines.linewidth": 2, "lines.linestyle": ":"}):
    plt.plot(data)
    
@mpl.rc_context({"lines.linewidth": 3, "lines.linestyle": "-"})
def plotting_function():
    plt.plot(data)
```

## 10.2 style sheets

```python
import matplotlib.pyplot as plt

print(plt.style.available)
plt.style.use("ggplot")
plt.style.use("./images/presentation.mplstyle")
plt.style.use(["dark_background", "presentation"])
# 临时样式
with plt.style.context("dark_background"):
    plt.plot(data)
plt.show()
```

## 10.3 matplotlibrc file

- matplotlibrc 文件的位置
	- 当前目录
	- \$MATPLOTLIBRC 或 \$MATPLOTLIBRC/matplotlibrc
	- .matplotlib/matplotlibrc
	- INSTALL/matplotlib/mpl-data/matplotlibrc

- 查看当前加载的 matplotlibrc 文件的位置

```python
import matplotlib as mpl
mpl.matploblib_fname()
```

