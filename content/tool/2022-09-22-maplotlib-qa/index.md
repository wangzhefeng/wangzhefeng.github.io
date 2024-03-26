---
title: Maplotlib QA
author: 王哲峰
date: '2022-09-22'
slug: maplotlib-qa
categories:
  - data analysis
  - data visual
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

- [颜色搭配](#颜色搭配)
- [图像风格设置](#图像风格设置)
- [解决使用中文乱码问题](#解决使用中文乱码问题)
    - [查看可以指定的中文字体](#查看可以指定的中文字体)
        - [查找系统已安装的中文字体](#查找系统已安装的中文字体)
        - [查找 Matplotlib 可以使用的中文字体](#查找-matplotlib-可以使用的中文字体)
        - [在系统上安装中文字体](#在系统上安装中文字体)
    - [指定中文字体名称](#指定中文字体名称)
        - [Matplotlib rc 设置全局字体](#matplotlib-rc-设置全局字体)
        - [在画图函数中使用字体名称](#在画图函数中使用字体名称)
    - [指定中文字体的具体路径](#指定中文字体的具体路径)
- [解决其他乱码问题](#解决其他乱码问题)
    - [负号的乱码问题](#负号的乱码问题)
    - [支持数学符号](#支持数学符号)
- [双坐标轴](#双坐标轴)
- [控制坐标轴刻度间距和标签](#控制坐标轴刻度间距和标签)
    - [添加主刻度和副刻度](#添加主刻度和副刻度)
    - [为刻度添加辅助网络](#为刻度添加辅助网络)
    - [控制刻度标签](#控制刻度标签)
        - [ticker.Locator 和 ticker.Formatter](#tickerlocator-和-tickerformatter)
        - [plt.xticks](#pltxticks)
    - [高级刻度标签控制](#高级刻度标签控制)
- [参考](#参考)
</p></details><p></p>

# 颜色搭配

# 图像风格设置

# 解决使用中文乱码问题

在使用 Matplotlib 画图的时候，发现一些 Unicode 字符(例如，汉字) 无法正常显示：在生成的图片中，
汉字是乱码的，显示为一个方框。经过大量的查找和阅读，
终于明白了如何在使用 Matplotlib 时，正确渲染 Unicode 字符

之所以中文字符被显示为方框，是因为 Matplotlib 默认使用的字体并不支持中文字符，
并不是 Matplotlib 本身的原因。为了能够在图片上正确显示中文字符，
需要指示 Matplotlib 使用一种支持中文的字体即可。
或者，更直接地，在画图时可以直接给 Matplotlib 提供一个中文字体的路径

## 查看可以指定的中文字体

### 查找系统已安装的中文字体

Matplotlib 提供了 `FontManager` 类来处理字体相关的操作，这个类有一个 `ttflist` 属性，
该属性提供了 Matplotlib 所能够发现到的字体列表。从这个字体列表，可以很容易得到这些字体的名称。
问题是，不清楚这些字体中有哪些字体是支持中文的。这时，需要使用 `fc-list` 命令行工具帮助找到系统上安装的中文字体。
如果系统上没有 `fc-list` 命令，应该先安装 [fontconfig](https://www.freedesktop.org/wiki/Software/fontconfig/)

```bash
$ fc-list :lang=zh
```

* Linux/macOS: 在 Linux/macOS 系统上，`fc_list` 程序通常是自带的，无需安装
* 在 Windows 系统上，可以安装 MiKTeX 或 Tex Live 来使用 `fc-list` 命令

### 查找 Matplotlib 可以使用的中文字体

使用 `fc-list :lang=zh` 可以列出系统上可用的中文字体，值的注意的是，这些中文字体并非都可以被 Matplotlib 使用，
Matplotlib 无法使用其中的 ttc(TrueType Collection) 格式的字体，
所以需要得到 Matplotlib 索引的字体和系统提供的中文字体两个集合的交集

```python
import matplotlib.font_manager import FontManager

fm = FontManager()
mat_fonts = set(f.name for f in fm.ttflist)
print(mat_fonts)
```

### 在系统上安装中文字体

在开始下面步骤之前，确保你的系统上已经安装了中文字体，如果你使用的是中文系统，
这应该不是问题；或者如果你想使用一种新的中文字体，
可以尝试 Google 和 Adobe 发布的 [Source Han Serif](https://source.typekit.com/source-han-serif/)

* macOS 中安装、停用字体
    - https://support.apple.com/zh-cn/HT201749
* macOS 中字体册
    ![img](images/mac_font.png)

## 指定中文字体名称

第一种使用中文的方式是给 Matplotlib 提供一个有效的中文字体名，有两种方式

### Matplotlib rc 设置全局字体

第一种方式是，使用 `rcParams` 设置全局中文字体名。
找到了 Matplotlib 索引的中文字体以后，
可以通过更改 Matplotlib rc 指示 Matplotlib 使用中文字体。
这样设置以后，后续脚本中的画图语句都会使用新指定的中文字体

* Windows/Linux

```python
import matplotlib as mpl

font_name = ["Arial Unicode MS"]
mpl.rcParams["font.family"] = font_name
mpl.rcParams["axes.unicode_minus"] = False

plt.text(0.5, 0.5, s = u"测试")
plt.show()
```

* macOS

```python
import matplotlib as mpl

font_name = ["Arial Unicode MS"]
mpl.rcParams["font.sans-serif"] = font_name
mpl.rcParams["axes.unicode_minus"] = False

plt.text(0.5, 0.5, s = u"测试")
plt.show()
```

### 在画图函数中使用字体名称

第二种方式是，仅想在某个画图命令中使用中文字体，可以在画图命令中指定使用的字体名称

```python
import matplotlib as mpl

font_name = "STKaiti"
mpl.rcParams["axes.unicode_minus"] = False

plt.text(0.5, 0.5, s = u"测试", fontname = font_name)
plt.show()
```

## 指定中文字体的具体路径

为了使用系统中的任何字体，也可以使用第二种方式：直接给 Matplotlib 提供一个字体的路径

Ubuntu:

```python
import matplotlib.pyplot as plt
import matplotlib.font_manager as mfm

font_path = "/usr/share/fonts/custom/simhei.ttf"
prop = mfm.FontProperties(fname = font_path)

plt.text(0.5, 0.5, s = u"测试", fontproperties = prop)
plt.show()
```

Windows:

```python

```

macOS:

```python

```

# 解决其他乱码问题

## 负号的乱码问题

```python
import matplotlib as mpl

mpl.rcParams["axes.unicode_minus"] = False # 解决图像中的 “-” 负号的乱码问题
```

## 支持数学符号

```python
from matplotlib import rc

rc('mathtext', default = 'regular')  # 支持数学符号
```

# 双坐标轴

```python
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc


# style
plt.style.use("classic")
plt.style.use("seaborn-whitegrid")
# font
plt.rcParams['font.sans-serif']= ["Arial Unicode MS"] # 支持中文(macOS)
# minus
mpl.rcParams["axes.unicode_minus"] = False # 解决图像中的 “-” 负号的乱码问题
# math
rc('mathtext', default = 'regular')  # 支持数学符号


def timeseries_plot_two_yaxis(df, 
                              col_xaxis,
                              col_left, col_right, 
                              col_left_ylim, col_right_ylim, 
                              title, imgpath):
    fig = plt.figure(figsize = (20, 7))

    # 左坐标轴画图
    ax = fig.add_subplot(111)
    line1 = ax.plot_date(
        df[col_xaxis], 
        df[col_left], 
        linestyle = "solid", 
        color = "#FF4700", 
        label = col_left
    )
    # 右坐标轴画图
    ax2 = ax.twinx()
    line2 = ax2.plot_date(
        df[col_xaxis], 
        df[col_right], 
        linestyle = "solid", 
        color = "#009AFF", 
        label = col_right
    )
    # 图例
    lines = line1 + line2
    labs = [line.get_label() for line in lines]
    ax.legend(lines, labs, loc = 0)
    
    # 日期设置
    plt.gcf().autofmt_xdate()
    plt.gca().xaxis.set_major_formatter(
        mpl_dates.DateFormatter('%Y-%m-%d')
    )
    
    # x 轴标签
    ax.grid()
    ax.set_xlabel(col_xaxis)

    # 左坐标轴设置
    ax.set_ylabel(col_left)
    ax.set_ylim(col_left_ylim)
    ax.set_xticks()
    ax.set_yticks()
    # 右坐标轴设置
    ax2.set_ylabel(col_right)
    ax2.set_ylim(col_right_ylim)
    ax2.xticks()
    ax2.yticks()
    
    # 标题
    plt.title(title)
    
    # 保存
    if imgpath:
        plt.savefig(os.path.join(os.path.dirname(__file__), imgpath))
```

# 控制坐标轴刻度间距和标签

Matplotlib 默认自动处理刻度在坐标轴上的位置，但有时我们需要覆盖默认的坐标轴刻度配置，
以便更加快速估计图形中点的坐标。

## 添加主刻度和副刻度

举例说明：强制水平刻度每隔 5 个单位步长呈现一次。此外，还添加了副刻度，
副刻度的间隔为 1 个单位步长。

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

x = np.linspace(-20, 20, 1024)
y = np.sinc(x)

ax = plt.axes()
ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
plt.plot(x, y, c = "m")
plt.show()
```

![img](images/ticker1.png)

## 为刻度添加辅助网络

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

x = np.linspace(-20, 20, 1024)
y = np.sinc(x)

ax = plt.axes()
ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
plt.grid(True, which = "both", ls = "dashed")
plt.plot(x, y, c = "m")
plt.show()
```

![img](images/ticker2.png)

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

x = np.linspace(-20, 20, 1024)
y = np.sinc(x)

ax = plt.axes()
ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
plt.grid(True, which = "major", ls = "dashed")
plt.plot(x, y, c = "m")
plt.show()
```

![img](images/ticker3.png)


```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

x = np.linspace(-20, 20, 1024)
y = np.sinc(x)

ax = plt.axes()
ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
plt.grid(True, which = "minor", ls = "dashed")
plt.plot(x, y, c = "m")
plt.show()
```

![img](images/ticker4.png)


## 控制刻度标签

刻度标签是图形空间中的坐标，虽然数字刻度标签对于大多说场景来说是足够的，但是却并不总是能够满足需求。
例如，我们需要显示 100 个公司的营收情况，这时候我们就需要横坐标刻度标签为公司名，而非数字；
同样对于时间序列，我们希望横坐标刻度标签为日期。考虑到此类需求，
我们需要使用 Matplotlib 为此提供了的 API 控制刻度标签。

### ticker.Locator 和 ticker.Formatter

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


name_list = ('Apple', 'Orange', 'Banana', 'Pear', 'Mango')
value_list = np.random.randint(0, 99, size = len(name_list))
pos_list = np.arange(len(name_list))

ax = plt.axes()
ax.xaxis.set_major_locator(
    ticker.FixedLocator((pos_list))
)
ax.xaxis.set_major_formatter(
    ticker.FixedFormatter((name_list))
)

plt.bar(pos_list, value_list, color = 'c', align = 'center')
plt.show()
```

首先使用 `ticker.Locator` 实例生成刻度的位置，然后使用 `ticker.Formatter` 实例为刻度生成标签。

* 用 `ticker.FixedFormatter` 从字符串列表中获取标签
* 用 `set_major_formatter` 设置坐标轴标签
* 用 `ticker.FixedLocator` 来确保每个标签中心都正好与刻度中间对齐
* 用 `ticker.set_major_locator` 设置坐标轴标签位置

### plt.xticks

虽然使用上述方法可以控制刻度标签，但可以看出此方法过于复杂，
如果刻度标签是固定的字符列表，那么可以用以下简单的设置方法。

```python
import numpy as np
import matplotlib.pyplot as plt

name_list = ('Apple', 'Orange', 'Banana', 'Pear', 'Mango')
value_list = np.random.randint(0, 99, size = len(name_list))
pos_list = np.arange(len(name_list))

plt.bar(pos_list, value_list, color = 'c', align = 'center')
plt.xticks(pos_list, name_list)
plt.show()
```

使用 plt.xticks() 函数为一组固定的刻度提供固定标签，此函数接受位置列表和名称列表作为参数值，
可以看出，此方法比第一种方法实现起来更简单。

## 高级刻度标签控制

不仅可以使用固定标签，使用 ticker API 可以使用函数生成的标签：

在此示例中，刻度标签是由自定义函数 `make_label` 生成的。此函数以刻度的坐标作为输入，
并返回一个字符串作为坐标标签，这比给出固定的字符串列表更灵活。为了使用自定义函数，
需要使用 `FuncFormatter` 实例——一个以函数为参数的格式化实例。

这种将生成标签的实际任务指派给其他函数的方法称为委托(delegation)模式，
这是一种漂亮的编程技术。比方说，我们要将每个刻度显示为日期，
这可以使用标准的 Python 时间和日期函数完成。

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def make_label(value, pos):
    return '%0.1f%%' % (100. * value)

ax = plt.axes()
ax.xaxis.set_major_formatter(ticker.FuncFormatter(make_label))

x = np.linspace(0, 1, 256)
plt.plot(x, np.exp(-10 * x), c ='c')
plt.plot(x, np.exp(-5 * x), c = 'c', ls = '--')
plt.show()
```


```python
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

start_date = datetime.datetime(1998, 1, 1)
def make_label(value, pos):
    time = start_date + datetime.timedelta(days = 365 * value)
    return time.strftime('%b %y')

ax = plt.axes()
ax.xaxis.set_major_formatter(ticker.FuncFormatter(make_label))

x = np.linspace(0, 1, 256)
plt.plot(x, np.exp(-10 * x), c ='c')
plt.plot(x, np.exp(-5 * x), c= 'c', ls = '--')

plt.setp(ax.get_xticklabels(), rotation = 30.0)
plt.show()
```

可以利用 `ax.get_xticklabels()` 获取刻度标签实例，然后对标签进行旋转，
以避免长标签之间重叠，旋转使用 `plt.setp()` 函数，其接受刻度标签实例和旋转角度作为参数值。


# 参考

* [Matplotlib控制坐标轴刻度间距和标签](https://developer.aliyun.com/article/840487)
