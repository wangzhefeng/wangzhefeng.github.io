---
title: 滤波算法
author: wangzf
date: '2022-11-19'
slug: timeseries-processing-filter
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
img {
    pointer-events: none;
}
</style>

<details><summary>目录</summary><p>

- [限幅滤波](#限幅滤波)
- [中位数滤波](#中位数滤波)
- [算法平均滤波](#算法平均滤波)
- [递推平均滤波(滑动平均滤波)](#递推平均滤波滑动平均滤波)
- [中位数平均滤波(防脉冲干扰平均滤波)](#中位数平均滤波防脉冲干扰平均滤波)
- [限幅平均滤波](#限幅平均滤波)
- [一阶滞后滤波](#一阶滞后滤波)
- [加权递推平均滤波](#加权递推平均滤波)
- [消抖滤波](#消抖滤波)
- [限幅消抖滤波](#限幅消抖滤波)
- [低通滤波](#低通滤波)
- [高通滤波](#高通滤波)
- [带通滤波](#带通滤波)
- [带阻滤波](#带阻滤波)
- [参考](#参考)
</p></details><p></p>

# 限幅滤波

> 限幅滤波也叫程序判断滤波法

* 方法: 
    - 根据经验判断, 确定两次采样允许的最大偏差值, 假设为 `$\delta$`, 每次检测到新的值时判断: 
        - 如果本次值与上次值之差小于等于 `$\delta$`, 则本次值有效
        - 如果本次值与上次值之差大于 `$\delta$`, 则本次值无效, 放弃本次值, 用上一次值代替本次值
* 优点: 
    - 能有效克服因偶然因素引起的脉冲干扰
* 缺点: 
    - 无法抑制周期性的干扰, 平滑度差

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.signal as signal

def limiting_filter(inputs, per):
    """
    限幅滤波(程序判断滤波法)
    Args:
        inputs:
        per:
    """
    pass
```

# 中位数滤波

* 方法: 
    - 连续采样 `$N$` 次(`$N$` 取奇数), 把 `$N$` 次采样值按照大小排列, 取中间值为本次有效值
* 优点: 
    - 能有效克服因偶然因素引起的波动干扰, 对温度、液位的变化缓慢的被测参数有良好的的滤波效果
* 缺点: 
    - 对流量、速度等快速变化的参数不适用

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.signal as signal

def median_filter(inputs, per):
    """
    中位值滤波
    Args:
        inputs:
        per:
    """
    pass
```

# 算法平均滤波

* 方法: 
    - 连续取 `$N$` 个采样值进行算术平均运算
        - `$N$` 值较大时: 信号平滑度较高, 但灵活性较低
        - `$N$` 值较小时: 信号平滑度较低, 但灵敏度较高
        - `$N$` 值的选取: 一般流量: `$N=12$`, 压力: `$N = 4$`
* 优点: 
    - 适用于对一般具有随机干扰的信号进行滤波, 这样的信号的特点是有一个平均值, 信号在某一数值范围附近上下波动
* 缺点: 
    - 对于测量速度较慢或要求数据计算速度较快的实时控制不适用, 比较浪费 RAM

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.signal as signal

def arithmetic_average_filter(inputs, per):
    '''
    算术平均滤波法
    Args:
        inputs:
        per:
    '''
    if np.shape(inputs)[0] % per != 0:
        lengh = np.shape(inputs)[0] / per
        for x in range(int(np.shape(inputs)[0]),int(lengh + 1)*per):
                inputs = np.append(inputs,inputs[np.shape(inputs)[0]-1])
    inputs = inputs.reshape((-1,per))
    mean = []
    for tmp in inputs:
        mean.append(tmp.mean())
    return mean

if __name__ == "__main__":
    num = signal.chirp(np.arange(0, 0.5, 1 / 4410.0), f0 = 10, t1 = 0.5, f1 = 1000.0)
    result = arithmetic_average_filter(num.copy(), 30)
    plt.subplot(2, 1, 1)
    plt.plot(num)
    plt.subplot(2, 1, 2)
    plt.plot(result)
    plt.show()
```

# 递推平均滤波(滑动平均滤波)

- 方法: 
    - 把连续取 `$N$` 个采样值看成一个队列队列的长度固定为 `$N$`，
      每次采样到一个新数据放入队尾, 并扔掉原来队首的一次数据.
      先进先出原则，把队列中的 `$N$` 个数据进行算术平均运算, 就可获得新的滤波结果
    - `$N$` 值的选取: 
        - 流量: `$N=1$`
        - 压力: `$N=4$`
        - 液面: `$N=4~12$`
        - 温度: `$N=1~4$`
- 优点: 
   - 对周期性干扰有良好的抑制作用, 平滑度高适用于高频振荡的系统
- 缺点: 
   - 灵敏度低对偶然出现的脉冲性干扰的抑制作用较差不易消除由于脉冲干扰所引起的采样值偏差不适用于脉冲干扰比较严重的场合比较浪费 RAM

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.signal as signal

def sliding_average_filter(inputs, per):
    '''
    递推平均滤波法(滑动平均滤波)
    Args:
        inputs:
        per: 
    filter = np.ones(200)*(1/200)
    sample_filter_03 = np.convolve(data,filter,'valid')
    sample_filter_012 = np.convolve(data,filter,'valid')
    '''
    if np.shape(inputs)[0] % per != 0:
        lengh = np.shape(inputs)[0] / per
        for x in range(int(np.shape(inputs)[0]),int(lengh + 1)*per):
                inputs = np.append(inputs,inputs[np.shape(inputs)[0]-1])
    inputs = inputs.reshape((-1,per))
    tmpmean = inputs[0].mean()
    mean = []
    for tmp in inputs:
        mean.append((tmpmean+tmp.mean())/2)
        tmpmean = tmp.mean()
    return mean
```

# 中位数平均滤波(防脉冲干扰平均滤波)

- 方法: 
   - 相当于`中位值滤波法 + 算术平均滤波法` 连续采样 `$N$` 个数据, 
      去掉一个最大值和一个最小值然后计算 `$N-2$` 个数据的算术平均值N值的选取: `$3~14$`
- 优点: 
   - 融合了两种滤波法的优点对于偶然出现的脉冲性干扰, 可消除由于脉冲干扰所引起的采样值偏差
- 缺点: 
   - 测量速度较慢, 和算术平均滤波法一样比较浪费 RAM

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.signal as signal

def median_average_filter(inputs, per):
    '''
    中位值平均滤波法(防脉冲干扰平均滤波)
    Args:
        inputs:
        per:
    '''
    if np.shape(inputs)[0] % per != 0:
        lengh = np.shape(inputs)[0] / per
        for x in range(int(np.shape(inputs)[0]),int(lengh + 1)*per):
                inputs = np.append(inputs,inputs[np.shape(inputs)[0]-1])
    inputs = inputs.reshape((-1,per))
    mean = []
    for tmp in inputs:
        tmp = np.delete(tmp,np.where(tmp==tmp.max())[0],axis = 0)
        tmp = np.delete(tmp,np.where(tmp==tmp.min())[0],axis = 0)
        mean.append(tmp.mean())
    return mean
```

# 限幅平均滤波

- 方法: 
     - 相当于 限幅滤波法 + 递推平均滤波法 每次采样到的新数据先进行限幅处理, 再送入队列进行递推平均滤波处理
- 优点: 
    - 融合了两种滤波法的优点对于偶然出现的脉冲性干扰, 可消除由于脉冲干扰所引起的采样值偏差
- 缺点: 
    - 比较浪费 RAM

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.signal as signal

def amplitude_limiting_average_filter(inputs, per, amplitude):
    '''
    限幅平均滤波法
    Args:
        inputs:
        per:
        amplitude: 限制最大振幅
    '''
    if np.shape(inputs)[0] % per != 0:
        lengh = np.shape(inputs)[0] / per
        for x in range(int(np.shape(inputs)[0]),int(lengh + 1)*per):
                inputs = np.append(inputs,inputs[np.shape(inputs)[0]-1])
    inputs = inputs.reshape((-1,per))
    mean = []
    tmpmean = inputs[0].mean()
    tmpnum = inputs[0][0] #上一次限幅后结果
    for tmp in inputs:
        for index,newtmp in enumerate(tmp):
                if np.abs(tmpnum-newtmp) > amplitude:
                tmp[index] = tmpnum
                tmpnum = newtmp
        mean.append((tmpmean+tmp.mean())/2)
        tmpmean = tmp.mean()
    return mean
```


# 一阶滞后滤波

- 方法: 
   - 取 `$a=0~1$` 本次滤波结果 `$=(1-a)$` 本次采样值 + a `$\times$`  上次滤波结果
- 优点: 
   - 对周期性干扰具有良好的抑制作用 适用于波动频率较高的场合
- 缺点: 
   - 相位滞后, 灵敏度低滞后程度取决于a值大小不能消除滤波频率高于采样频率的1/2的干扰信号

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.signal as signal

def first_order_lag_filter(inputs, a):
    '''
    一阶滞后滤波法
    Args:
        inputs:
        a: 滞后程度决定因子, 0~1
    '''
    tmpnum = inputs[0] #上一次滤波结果
    for index,tmp in enumerate(inputs):
        inputs[index] = (1-a)*tmp + a*tmpnum
        tmpnum = tmp
    return inputs
```

# 加权递推平均滤波

- 方法: 
   - 是对递推平均滤波法的改进, 即不同时刻的数据加以不同的权通常是, 越接近现时刻的数据, 权取得越大. 给予新采样值的权系数越大, 则灵敏度越高, 但信号平滑度越低
- 优点:
   - 适用于有较大纯滞后时间常数的对象 和采样周期较短的系统
- 缺点: 
   - 对于纯滞后时间常数较小, 采样周期较长, 变化缓慢的信号不能迅速反应系统当前所受干扰的严重程度, 滤波效果差

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.signal as signal

def weight_backstep_average_filter(inputs, per):
    '''
    加权递推平均滤波法
    Args:
        inputs: 
        per:
    '''
    weight = np.array(range(1,np.shape(inputs)[0]+1)) # 权值列表
    weight = weight/weight.sum()

    for index,tmp in enumerate(inputs):
        inputs[index] = inputs[index]*weight[index]
    return inputs
```

# 消抖滤波

- 方法: 
   - 设置一个滤波计数器将每次采样值与当前有效值比较: 如果采样值 ＝= 当前有效值, 则计数器清零如果采样值 <> 当前有效值, 
      则计数器 + 1, 并判断计数器是否 >= 上限 N(溢出) 如果计数器溢出,则将本次值替换当前有效值,并清计数器
- 优点: 
   - 对于变化缓慢的被测参数有较好的滤波效果, 可避免在临界值附近控制器的反复开/关跳动或显示器上数值抖动
- 缺点: 
   - 对于快速变化的参数不宜如果在计数器溢出的那一次采样到的值恰好是干扰值,则会将干扰值当作有效值导入系统

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.signal as signal

def shake_off_filter(inputs, N):
    '''
    消抖滤波法
    Args:
        inputs:
        N: 消抖上限
    '''
    usenum = inputs[0] #有效值
    i = 0 # 标记计数器
    for index,tmp in enumerate(inputs):
        if tmp != usenum:
                i = i + 1
                if i >= N:
                i = 0
                inputs[index] = usenum
    return inputs
```

# 限幅消抖滤波

- 方法: 
   - 相当于 `限幅滤波法 + 消抖滤波法` 先限幅, 后消抖
- 优点: 
   - 继承了 `限幅` 和 `消抖` 的优点改进了 `消抖滤波法` 中的某些缺陷, 避免将干扰值导入系统
- 缺点: 
   - 对于快速变化的参数不宜

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.signal as signal

def amplitude_limiting_shake_off_filter(inputs, amplitude, N):
    '''
    限幅消抖滤波法
    Args:
        inputs:
        amplitude: 限制最大振幅
        N:         消抖上限
    '''
    tmpnum = inputs[0]
    for index,newtmp in enumerate(inputs):
        if np.abs(tmpnum-newtmp) > amplitude:
                inputs[index] = tmpnum
        tmpnum = newtmp
    usenum = inputs[0]
    i = 0
    for index2,tmp2 in enumerate(inputs):
        if tmp2 != usenum:
                i = i + 1
                if i >= N:
                i = 0
                inputs[index2] = usenum
    return inputs
```

# 低通滤波

- 低通滤波指的是去除高于某一阈值频率的信号
- 假设采样频率为 1000hz, 信号本身最大的频率为 500hz, 要滤除 400hz 以上频率成分, 
 即截至频率为 400hz, 则归一化截止频率 `$W_{n}=\frac{2 \times 400}{1000}=0.8$`

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.signal as signal

def low_pass_filer(data, N, Wn):
    """
    低通滤波:  低通滤波指的是去除高于某一阈值频率的信号

    Args:
        data ([type]): 要过滤的信号
        N ([type]): 滤波器的阶数
        Wn ([type]): 归一化截止频率, Wn = 2 * 截止频率 / 采样频率
                - 根据采样定理, 采样频率要大于两倍的信号本身最大的频率, 才能还原信号. 
                - 截止频率一定小于信号本身最大的频率, 所以 Wn 一定在 0 和 1 之间
                - 当构造带通滤波器或者带阻滤波器时, Wn为长度为2的列表
        b: 滤波器的分子
        a: 滤波器的分母
    Returns:
        [type]: [description]
    """
    b, a = signal.butter(N = N, Wn = Wn, btype = "lowpass")
    filted_data = signal.filtfilt(b, a, data)
    return filted_data
```


# 高通滤波

- 高通滤波去除低于某一频率的信号
- 假设采样频率为 1000hz, 信号本身最大的频率为 500hz, 要滤除 100hz 以下频率成分, 
即截至频率为 100hz, 则归一化截止频率 `$W_{n}=\frac{2 \times 100}{100W_{}=0.2$`

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.signal as signal

def high_pass_filter(data, N, Wn):
   """
   高通滤波: 高通滤波去除低于某一频率的信号

   Args:
      data ([type]): 要过滤的信号
      N ([type]): 滤波器的阶数
      Wn ([type]): 归一化截止频率, Wn = 2 * 截止频率 / 采样频率
            - 根据采样定理, 采样频率要大于两倍的信号本身最大的频率, 才能还原信号. 
            - 截止频率一定小于信号本身最大的频率, 所以 Wn 一定在 0 和 1 之间
            - 当构造带通滤波器或者带阻滤波器时, Wn为长度为2的列表
      b: 滤波器的分子
      a: 滤波器的分母

   Returns:
      [type]: [description]
   """
   b, a = signal.butter(N = N, Wn = Wn, btype = "highpass")
   filted_data = signal.filtfilt(b, a, data)
   return filted_data
```

# 带通滤波

- 带通滤波指的是类似低通高通的结合保留中间频率信号
- 假设采样频率为 1000hz, 信号本身最大的频率为 500hz, 要滤除 100hz 以下, 400hz 以上频率成分, 
   即截至频率为 100hz, 400hz, 则归一化截止频率 $W_{n1}=\frac{2 \times 100}{1000}=0.2`,
   $W_{n2}=\frac{2 \times 400}{1000}=0.8$, 所以:  $W_{n}=[0.02,0.8]$

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.signal as signal

def band_pass_filter(data, N, Wn):
   """
   带通滤波: 带通滤波指的是类似低通高通的结合保留中间频率信号

   Args:
      data ([type]): 要过滤的信号
      N ([type]): 滤波器的阶数
      Wn ([type]): 归一化截止频率, Wn = 2 * 截止频率 / 采样频率
            - 根据采样定理, 采样频率要大于两倍的信号本身最大的频率, 才能还原信号. 
            - 截止频率一定小于信号本身最大的频率, 所以 Wn 一定在 0 和 1 之间
            - 当构造带通滤波器或者带阻滤波器时, Wn为长度为2的列表
      b: 滤波器的分子
      a: 滤波器的分母

   Returns:
      [type]: [description]
   """
   b, a = signal.butter(N = N, Wn = Wn, btype = "bandpass")
   filted_data = signal.filtfilt(b, a, data)
   return filted_data
```

# 带阻滤波

- 带阻滤波也是低通高通的结合只是过滤掉的是中间部分
- 假设采样频率为 1000hz, 信号本身最大的频率为 500hz, 要滤除 100hz 以上, 400hz 以下频率成分, 
 即截至频率为 100hz, 400hz, 则 $W_{n1}=\frac{2 \times 100}{1000}=0.2$,
 $W_{n2}=\frac{2 \times 400}{1000}=0.8$, 所以:  $W_{n}=[0.2,0.8]$
 和带通相似, 但是带通是保留中间, 而带阻是去除

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.signal as signal

def band_stop_filter(data, N, Wn):
    """
    带阻滤波: 带阻滤波也是低通高通的结合只是过滤掉的是中间部分

    Args:
        data ([type]): 要过滤的信号
        N ([type]): 滤波器的阶数
        Wn ([type]): 归一化截止频率, Wn = 2 * 截止频率 / 采样频率
                - 根据采样定理, 采样频率要大于两倍的信号本身最大的频率, 才能还原信号. 
                - 截止频率一定小于信号本身最大的频率, 所以 Wn 一定在 0 和 1 之间
                - 当构造带通滤波器或者带阻滤波器时, Wn为长度为2的列表
        b: 滤波器的分子
        a: 滤波器的分母

    Returns:
        [type]: [description]
    """
    b, a = signal.butter(N = N, Wn = Wn, btype = "bandstop")
    filted_data = signal.filtfilt(b, a, data)
    return filted_data
```

# 参考

* [1](https://blog.csdn.net/u010720661/article/details/63253509)
* [2](http://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/)
* [3](https://www.geek-workshop.com/thread-7694-1-1.html>)
* [4](https://blog.csdn.net/u014033218/article/details/97004609?utm_medium=distribute.pc_relevant.none-task-blog-baidujs_baidulandingword-0&spm=1001.2101.3001.4242)
* [5](https://blog.csdn.net/kengmila9393/article/details/81455165)
* [6](https://blog.csdn.net/phker/article/details/48468591)

