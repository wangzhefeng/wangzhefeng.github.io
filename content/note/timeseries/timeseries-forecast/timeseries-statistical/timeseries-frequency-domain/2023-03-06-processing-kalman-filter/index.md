---
title: 卡尔曼滤波
author: 王哲峰
date: '2023-03-06'
slug: processing-kalman-filter
categories:
  - timeseries
tags:
  - model
---

# 卡尔曼滤波

* 什么是卡尔曼滤波？
    - 你可以在任何含有不确定信息的动态系统中使用卡尔曼滤波, 对系统下一步的走向做出有根据的预测, 
      即使伴随着各种干扰, 卡尔曼滤波总是能指出真实发生的情况
    - 在连续变化的系统中使用卡尔曼滤波是非常理想的, 它具有占内存小的优点(除了前一个状态量外, 不需要保留其它历史数据), 
      而且速度很快, 很适合应用于实时问题和嵌入式系统
* 算法的核心思想:
    - 根据当前的仪器 测量值和上一刻的预测值和误差值, 计算得到当前的最优量, 再预测下一刻的量
        - 核心思想比较突出的观点是把误差纳入计算, 而且分为预测误差和测量误差两种, 统称为噪声
        - 核心思想还有一个非常大的特点是: 误差独立存在, 始终不受测量数据的影响









# Python API

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.signal as signal
```

# 参考

* [卡尔曼滤波](https://mp.weixin.qq.com/s?__biz=MzUyNzA1OTcxNg==&mid=2247486294&idx=1&sn=5c84f404cd77f2742b12d30c1fb5427d&chksm=fa04153dcd739c2b3f9c70a5aa1674a8c523fb885635f6854df4015c9dd2df72bf0017d5a66f&scene=178&cur_album_id=1577157748566310916#rd)

