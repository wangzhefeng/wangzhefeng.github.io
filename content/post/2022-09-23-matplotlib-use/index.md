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
</style>

<details><summary>目录</summary><p>

- [快速开始](#快速开始)
- [一张统计图的结构](#一张统计图的结构)
  - [API](#api)
- [Subplots layout](#subplots-layout)
  - [API](#api-1)
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
</p></details><p></p>

<!-- <img src="images/quick_start.png" width="50%" /> -->

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

<img src="images/figure.png" width="100%" />

# 一张统计图的结构

<img src="images/anatomy.png" width="70%" />

## API

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

