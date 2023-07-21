---
title: Numpy
author: 王哲峰
date: '2023-04-20'
slug: numpy
categories:
  - python
tags:
  - tool
---


```python    
# -*- coding: utf-8 -*-


import numpy as np
from numpy import pi
import numpy.linalg
from numpy.linalg import *
import numpy.random
from numpy.random import *
np.set_printoptions(threshold = 'nan')


#####################################################################
#           1. Numpy的ndarray: 一种多维数组对象
#####################################################################
# 1.1 创建ndarray
# ----------------------------------
# 数组创建函数
data_list = [1, 2, 3]
data_tuple = (1, 2, 3)
data_string = "python"
arr_list = np.array(data_list)               # np.array(, dtype = "")
arr_tuple = np.array(data_tuple)
arr_string = np.array(data_string)
np.asarray(data_list)                        # np.asarray()
np.arange(10)                                # ndarray
list(range(10))                              # list
np.linspace(start = 2, stop = 5, num = 5)    # 生成序列
np.ones((4, 5))
np.ones_like(arr_tuple)
np.zeros((4, 5))
np.zeros_like(arr_tuple)
np.empty((4, 5))
np.empty_like(arr_tuple)
np.eye(4)
np.identity(4)

# 查看数组维度, 形状, 大小, 类型, 数据存储
print(arr_list.ndim)
print(arr_list.shape)
print(arr_list.size)
print(arr_list.dtype)
print(arr_list.itemsize)
print(arr_list.data)
type(arr_list)

# ----------------------------------
# 1.2 ndarray的数据类型
# ----------------------------------
# np.array(, dtype = 'i1, u1, i2, u2, i4, u4, i8, u8',
# 	               np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64
# 	               'f2, f4, f, f8, d, f16, g',
# 	               np.float16, np.float32, np.float64, np.float128,
# 	               'c8, c16, c32',
# 	               np.complex64, np.complex128, np.complex256,
# 	               '?',
# 	               np.bool,
# 	               'O',
# 	               np.object,
# 	               'S(x)',
# 	               np.string_,
# 	               'U(x)',
# 	               np.unicode_)
data1 = [1, 2, 3, 4]
data2 = [1.0, 2.0, 3.0, 4.0]
arr1 = np.array(data1)
arr1.astype(dtype = np.float64)
arr2 = np.array(data2)
arr2.astype(arr1.dtype)

# ----------------------------------
# 1.3 数组与标量之间的运算(矢量化 vectorization)
# ----------------------------------
# +, -, *, /, **
arr = np.array([[1.0, 2, 3.0], [4.0, 5.0, 6.0]])
print(arr)
arr + arr
arr - arr
arr * arr
1 / arr
arr ** 0.5

# ----------------------------------
# 1.4 基本的索引和切片
# ----------------------------------
# 1d
arr = np.arange(10)
# 索引
print(arr[5])
# 切片(不包含第二个index)
print(arr[1:])
print(arr[:7])
print(arr[1:7])
print(arr[:])

# 2d
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# 索引
print(arr2d[2])
print(arr2d[0, 0])
print(arr2d[0][0])
# 切片
print(arr2d[:])
print(arr2d[:, :])
print(arr2d[:, ])
print(arr2d[, 1])

# 3d
arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# 索引
print(arr3d[0])
print(arr3d[0, 0])
print(arr3d[0][0])
print(arr3d[0, 0, 0])
print(arr3d[0][0][0])
# 切片
print(arr3d[:])
print(arr3d[:, :])
print(arr3d[:, ])
print(arr3d[, :])
print(arr3d[:, :, :])
print(arr3d[:, :, ])
print(arr3d[:, , :])
print(arr3d[, :, :])
print(arr3d[:, , ])
print(arr3d[, :, ])
print(arr3d[, , :])
# ----------------------------------
# 1.5 布尔型索引
# ----------------------------------
# 布尔型数组的长度必须和被索引的轴的长度一致, 此外还可以将布尔型数组跟切片、整数混合使用
# 比较运算符
# >
# >=
# <
# <=
# ==
# "!="
# -（条件非）
# &（条件和）
# |（条件或）
# ----------------------------------
# 1.6 花式索引(Fancy indexing)
# ----------------------------------
arr = np.empty((8, 4))
for i in range(8):
    arr[i] = i
print(arr)
print(arr[[4, 3, 0, 6]])    # 选取多行
print(arr[[-3, -5, -7]])

arr = np.arange(32).reshape((8, 4))
print(arr[[1, 5, 7, 2], [0, 3, 1, 2]])

print(arr[[1, 5, 7, 2]][: [0, 3, 1, 2]])
# ==
print(arr[np.ix_([1, 5, 7, 2], [0, 3, 1, 2])])


# 1.7 数组转置和轴对换(View)
# .T
# .transpose()
# .swapaxes()

# arr.T == arr.transpose()
arr = np.arange(15).reshape((3, 5))
print(arr.transpose())
print(arr.T)

arr = np.random.randn(6, 3)
print(np.dot(arr.T, arr))

arr = np.arange(16).reshape((2, 2, 4))
arr.transpose((1, 0, 2))
arr.swapaxex(1, 2)
############################################################################
#              2 通用函数(ufunc:执行元素级运算的函数)
############################################################################
arr1 = np.random.randn(8)
arr2 = np.random.randn(8)

np.abs(arr1)
np.fabs(arr1)      # 对于非复数更快
np.sqrt(arr1)      # arr ** 0.5
np.square(arr1)    # arr ** 2
np.exp(arr1)
np.log(arr1)       # e为底
np.log10(arr1)     # 10为底
np.log2(arr1)      # 2为底
np.log1p(arr1)     # log(1+x)
np.sign(arr1)
np.ceil(arr1)      # 计算各元素的大于等于该值的最小整数
np.floor(arr1)     # 计算各元素的小于等于该值的最大整数
np.rint(arr1)      # 将各元素四舍五入到最近的整数, 保留dtype
np.modf(arr1)      # 将数组的小数和整数部分以两个独立数组的形式返回
np.isnan(arr1)
np.isfinite(arr1)  # 是否为有穷数
np.isinf(arr1)     # 是否为无穷数
np.cos(arr1)
np.cosh(arr1)
np.sin(arr1)
np.sinh(arr1)
np.tan(arr1)
np.tanh(arr1)
np.arccos(arr1)
np.arccosh(arr1)
np.arcsin(arr1)
np.arcsinh(arr1)
np.arctan(arr1)
np.arctanh(arr1)
np.logical_not(arr1) #  -arr

np.add(arr1, arr2)          #  +
np.subtract(arr1, arr2)     #  -
np.multiply(arr1, arr2)     #  *
np.divide(arr1, arr2)       # /
np.floor_divide(arr1, arr2) # /丢弃余数
np.power(arr1, 2)           # arr ** 2
np.maximum(arr1, arr2)
np.fmax(arr1, arr2)                  # 忽略NaN
np.minimum(arr1, arr2)
np.fmin(arr1, arr2)                  # 忽略NaN
np.mod(arr1, arr2)                   # 余数
np.copysign(arr1, arr2)              # 将第二个数组中的符号复制给第一个数组中的值
np.greater(arr1, arr2)               # >
np.greater_equal(arr1, arr2)         # >=
np.less(arr1, arr2)                  # <
np.less_equal(arr1, arr2)            # <=
np.equal(arr1, arr2)                 # ==
np.not_equal(arr1, arr2)             # !=
np.logical_and(arr1, arr2)           # &
np.logical_or(arr1, arr2)            # |
np.logical_xor(arr1, arr2)            # ^
#####################################################################
#                       3.利用数组进行数据处理
#####################################################################
np.meshgrid() # 接受两个一维数组, 并且产生两个举证,对应于两个数组中所有的(x, y)对
points = np.arange(-5, 5, 0.01)
print(points)
xs, ys = np.meshgrid(points, points)
print(xs)
print(ys)

# ----------------------------------
## 3.1 将条件逻辑表达为数组运算
# ----------------------------------
# np.where()  ### x if condetion else y
print(1 if "wangzhefeng" == "python" else 0)

xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])
result_1 = [(x if c else y) for x, y, c in zip(xarr, yarr, cond)]
reslut_2 = np.where(xarr, yarr, cond)
print(reslut_2)

arr = np.random.randn(4, 4)
np.where(arr > 0, 2, -2)
np.where(arr > 0, 2, arr)
# np.where(cond1 & cond2, 0, np.where(cond1, 1, np.where(cond2, 2, 3)))

# ----------------------------------
## 3.2 数学和统计方法
# ----------------------------------
arr = np.random.randn(5, 4)
arr.sum()
arr.sum(axis = 1)
arr.sum(axis = 0)
np.sum(arr)
np.sum(arr, axis = 1)
np.sum(arr, axis = 0)

np.mean()
.mean()
np.cumsum()
.cumsum()
np.cumprod()
.cumprod()
np.std()
.std()
np.var()
.var()
np.min()
.min()
np.max()
.max()
np.argmin() # index
.argmin()
np.argmax() # index
.argmax()
# ----------------------------------
## 3.3 用于布尔型数组的方法
# ----------------------------------
.sum()   # 布尔数组中的True值计数
.any()   # 测试数组中是否存在一个或多个True
.all()   # 测试数组中所有值是否都是True
arr = np.random.randn(100)
(arr > 0).sum()

bools = np.array([False, False, True, False])
bools.any()
bools.all()
# ----------------------------------
## 3.4 排序
# ----------------------------------
np.sort()             #（副本）
np.argsort()
.sort(axis = 0, 1, 2) #（就地排序）
sorted()
arr = np.random.randn(8)
arr.sort()
arr = np.random.randn(5, 3)
arr.sort(axis = 1)
# ----------------------------------
## 3.5 唯一化及其他的集合逻辑--一维数组
# ----------------------------------
np.unique(x)
sort(set(x))
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
np.unique(names) # sorted(set(names))
np.intersect1d(x, y) # 交集
np.union1d(x, y)     # 并集
np.in1d(x, y)        # 元素包含关系
np.setdiff1d(x, y)   # 差集
np.setxor1d(x, y)    # 异或
#####################################################################
#                   4.用于数组的文件输入输出 (二进制数据和文本数据)
#####################################################################
# 二进制
np.save('.npy', arr)
np.load('.npy')
np.savez('.npz', arr1, arr2)
np.load('.npz')
# 文本
np.loadtxt('.txt', delimiter = ',')
np.genfromtxt()
np.savetxt()
#####################################################################
#                           5.线性代数
#####################################################################
from numpy.linalg import *
import numpy.linalg

np.linalg.diag()
np.linalg.dot(x, y)  # 矩阵点积
x.dot(y)             # 矩阵元素乘积
np.linalg.trace()
np.linalg.det()
np.linalg.eig()

np.linalg.inv()
np.linalg.pinv()

np.linalg.qr()
np.linalg.svd()

np.linalg.solve()
np.linalg.lstsq()

#####################################################################
#                         6.随机数生成 (函数参数)
#####################################################################
np.random.seed(123)                    # 设置随机数
np.random.permutation(np.arange(16))   # 返回一个序列的随机排列或返回一个随机排列的范围
np.random.shuffle(np.arange(5))        # 对一个序列就地随机排列
np.random.rand((5)                     # 生成均匀分布随机数[0, 1)
np.random.uniform(10)                  # 均匀分布(0, 1)
np.random.randint()                    # 从给定范围内随机取整数

# ----------------------------------
# 正态分布
# ----------------------------------
np.random.normal(loc = 0, scale = 1, size = (6))
np.random.normal(size = (5))
# from random import normalvariate
# normalvariate(, )
np.random.randn()                      # 标准正态分布
np.random.normal(loc = 0, scale = 1, size = (6))
# ----------------------------------
# 其他分布
# ----------------------------------
np.random.binomial(5)
np.random.beta(5)
np.random.chisquare(5)
np.random.gamma(5)

#####################################################################
#                               其他
#####################################################################
np.array().reshape()
np.array().ravel()
np.array().flatten()
np.concatenate()
np.vstack()
np.hstack()
np.array().repeat([], axis = 1)
np.array().repeat([], axis = 0)
np.tile(np.array(), ())
```
```python    
# -*- coding: utf-8 -*-


import numpy as np
from numpy import pi
import numpy.linalg
from numpy.linalg import *
import numpy.random
from numpy.random import *
np.set_printoptions(threshold = 'nan')


#####################################################################
#           1. Numpy的ndarray: 一种多维数组对象
#####################################################################
# 1.1 创建ndarray
# ----------------------------------
# 数组创建函数
data_list = [1, 2, 3]
data_tuple = (1, 2, 3)
data_string = "python"
arr_list = np.array(data_list)               # np.array(, dtype = "")
arr_tuple = np.array(data_tuple)
arr_string = np.array(data_string)
np.asarray(data_list)                        # np.asarray()
np.arange(10)                                # ndarray
list(range(10))                              # list
np.linspace(start = 2, stop = 5, num = 5)    # 生成序列
np.ones((4, 5))
np.ones_like(arr_tuple)
np.zeros((4, 5))
np.zeros_like(arr_tuple)
np.empty((4, 5))
np.empty_like(arr_tuple)
np.eye(4)
np.identity(4)

# 查看数组维度, 形状, 大小, 类型, 数据存储
print(arr_list.ndim)
print(arr_list.shape)
print(arr_list.size)
print(arr_list.dtype)
print(arr_list.itemsize)
print(arr_list.data)
type(arr_list)

# ----------------------------------
# 1.2 ndarray的数据类型
# ----------------------------------
# np.array(, dtype = 'i1, u1, i2, u2, i4, u4, i8, u8',
# 	               np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64
# 	               'f2, f4, f, f8, d, f16, g',
# 	               np.float16, np.float32, np.float64, np.float128,
# 	               'c8, c16, c32',
# 	               np.complex64, np.complex128, np.complex256,
# 	               '?',
# 	               np.bool,
# 	               'O',
# 	               np.object,
# 	               'S(x)',
# 	               np.string_,
# 	               'U(x)',
# 	               np.unicode_)
data1 = [1, 2, 3, 4]
data2 = [1.0, 2.0, 3.0, 4.0]
arr1 = np.array(data1)
arr1.astype(dtype = np.float64)
arr2 = np.array(data2)
arr2.astype(arr1.dtype)

# ----------------------------------
# 1.3 数组与标量之间的运算(矢量化 vectorization)
# ----------------------------------
# +, -, *, /, **
arr = np.array([[1.0, 2, 3.0], [4.0, 5.0, 6.0]])
print(arr)
arr + arr
arr - arr
arr * arr
1 / arr
arr ** 0.5

# ----------------------------------
# 1.4 基本的索引和切片
# ----------------------------------
# 1d
arr = np.arange(10)
# 索引
print(arr[5])
# 切片(不包含第二个index)
print(arr[1:])
print(arr[:7])
print(arr[1:7])
print(arr[:])

# 2d
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# 索引
print(arr2d[2])
print(arr2d[0, 0])
print(arr2d[0][0])
# 切片
print(arr2d[:])
print(arr2d[:, :])
print(arr2d[:, ])
print(arr2d[, 1])

# 3d
arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# 索引
print(arr3d[0])
print(arr3d[0, 0])
print(arr3d[0][0])
print(arr3d[0, 0, 0])
print(arr3d[0][0][0])
# 切片
print(arr3d[:])
print(arr3d[:, :])
print(arr3d[:, ])
print(arr3d[, :])
print(arr3d[:, :, :])
print(arr3d[:, :, ])
print(arr3d[:, , :])
print(arr3d[, :, :])
print(arr3d[:, , ])
print(arr3d[, :, ])
print(arr3d[, , :])
# ----------------------------------
# 1.5 布尔型索引
# ----------------------------------
# 布尔型数组的长度必须和被索引的轴的长度一致, 此外还可以将布尔型数组跟切片、整数混合使用
# 比较运算符
# >
# >=
# <
# <=
# ==
# "!="
# -（条件非）
# &（条件和）
# |（条件或）
# ----------------------------------
# 1.6 花式索引(Fancy indexing)
# ----------------------------------
arr = np.empty((8, 4))
for i in range(8):
    arr[i] = i
print(arr)
print(arr[[4, 3, 0, 6]])    # 选取多行
print(arr[[-3, -5, -7]])

arr = np.arange(32).reshape((8, 4))
print(arr[[1, 5, 7, 2], [0, 3, 1, 2]])

print(arr[[1, 5, 7, 2]][: [0, 3, 1, 2]])
# ==
print(arr[np.ix_([1, 5, 7, 2], [0, 3, 1, 2])])


# 1.7 数组转置和轴对换(View)
# .T
# .transpose()
# .swapaxes()

# arr.T == arr.transpose()
arr = np.arange(15).reshape((3, 5))
print(arr.transpose())
print(arr.T)

arr = np.random.randn(6, 3)
print(np.dot(arr.T, arr))

arr = np.arange(16).reshape((2, 2, 4))
arr.transpose((1, 0, 2))
arr.swapaxex(1, 2)
############################################################################
#              2 通用函数(ufunc:执行元素级运算的函数)
############################################################################
arr1 = np.random.randn(8)
arr2 = np.random.randn(8)

np.abs(arr1)
np.fabs(arr1)      # 对于非复数更快
np.sqrt(arr1)      # arr ** 0.5
np.square(arr1)    # arr ** 2
np.exp(arr1)
np.log(arr1)       # e为底
np.log10(arr1)     # 10为底
np.log2(arr1)      # 2为底
np.log1p(arr1)     # log(1+x)
np.sign(arr1)
np.ceil(arr1)      # 计算各元素的大于等于该值的最小整数
np.floor(arr1)     # 计算各元素的小于等于该值的最大整数
np.rint(arr1)      # 将各元素四舍五入到最近的整数, 保留dtype
np.modf(arr1)      # 将数组的小数和整数部分以两个独立数组的形式返回
np.isnan(arr1)
np.isfinite(arr1)  # 是否为有穷数
np.isinf(arr1)     # 是否为无穷数
np.cos(arr1)
np.cosh(arr1)
np.sin(arr1)
np.sinh(arr1)
np.tan(arr1)
np.tanh(arr1)
np.arccos(arr1)
np.arccosh(arr1)
np.arcsin(arr1)
np.arcsinh(arr1)
np.arctan(arr1)
np.arctanh(arr1)
np.logical_not(arr1) #  -arr

np.add(arr1, arr2)          #  +
np.subtract(arr1, arr2)     #  -
np.multiply(arr1, arr2)     #  *
np.divide(arr1, arr2)       # /
np.floor_divide(arr1, arr2) # /丢弃余数
np.power(arr1, 2)           # arr ** 2
np.maximum(arr1, arr2)
np.fmax(arr1, arr2)                  # 忽略NaN
np.minimum(arr1, arr2)
np.fmin(arr1, arr2)                  # 忽略NaN
np.mod(arr1, arr2)                   # 余数
np.copysign(arr1, arr2)              # 将第二个数组中的符号复制给第一个数组中的值
np.greater(arr1, arr2)               # >
np.greater_equal(arr1, arr2)         # >=
np.less(arr1, arr2)                  # <
np.less_equal(arr1, arr2)            # <=
np.equal(arr1, arr2)                 # ==
np.not_equal(arr1, arr2)             # !=
np.logical_and(arr1, arr2)           # &
np.logical_or(arr1, arr2)            # |
np.logical_xor(arr1, arr2)            # ^
#####################################################################
#                       3.利用数组进行数据处理
#####################################################################
np.meshgrid() # 接受两个一维数组, 并且产生两个举证,对应于两个数组中所有的(x, y)对
points = np.arange(-5, 5, 0.01)
print(points)
xs, ys = np.meshgrid(points, points)
print(xs)
print(ys)

# ----------------------------------
## 3.1 将条件逻辑表达为数组运算
# ----------------------------------
# np.where()  ### x if condetion else y
print(1 if "wangzhefeng" == "python" else 0)

xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])
result_1 = [(x if c else y) for x, y, c in zip(xarr, yarr, cond)]
reslut_2 = np.where(xarr, yarr, cond)
print(reslut_2)

arr = np.random.randn(4, 4)
np.where(arr > 0, 2, -2)
np.where(arr > 0, 2, arr)
# np.where(cond1 & cond2, 0, np.where(cond1, 1, np.where(cond2, 2, 3)))

# ----------------------------------
## 3.2 数学和统计方法
# ----------------------------------
arr = np.random.randn(5, 4)
arr.sum()
arr.sum(axis = 1)
arr.sum(axis = 0)
np.sum(arr)
np.sum(arr, axis = 1)
np.sum(arr, axis = 0)

np.mean()
.mean()
np.cumsum()
.cumsum()
np.cumprod()
.cumprod()
np.std()
.std()
np.var()
.var()
np.min()
.min()
np.max()
.max()
np.argmin() # index
.argmin()
np.argmax() # index
.argmax()
# ----------------------------------
## 3.3 用于布尔型数组的方法
# ----------------------------------
.sum()   # 布尔数组中的True值计数
.any()   # 测试数组中是否存在一个或多个True
.all()   # 测试数组中所有值是否都是True
arr = np.random.randn(100)
(arr > 0).sum()

bools = np.array([False, False, True, False])
bools.any()
bools.all()
# ----------------------------------
## 3.4 排序
# ----------------------------------
np.sort()             #（副本）
np.argsort()
.sort(axis = 0, 1, 2) #（就地排序）
sorted()
arr = np.random.randn(8)
arr.sort()
arr = np.random.randn(5, 3)
arr.sort(axis = 1)
# ----------------------------------
## 3.5 唯一化及其他的集合逻辑--一维数组
# ----------------------------------
np.unique(x)
sort(set(x))
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
np.unique(names) # sorted(set(names))
np.intersect1d(x, y) # 交集
np.union1d(x, y)     # 并集
np.in1d(x, y)        # 元素包含关系
np.setdiff1d(x, y)   # 差集
np.setxor1d(x, y)    # 异或
#####################################################################
#                   4.用于数组的文件输入输出 (二进制数据和文本数据)
#####################################################################
# 二进制
np.save('.npy', arr)
np.load('.npy')
np.savez('.npz', arr1, arr2)
np.load('.npz')
# 文本
np.loadtxt('.txt', delimiter = ',')
np.genfromtxt()
np.savetxt()
#####################################################################
#                           5.线性代数
#####################################################################
from numpy.linalg import *
import numpy.linalg

np.linalg.diag()
np.linalg.dot(x, y)  # 矩阵点积
x.dot(y)             # 矩阵元素乘积
np.linalg.trace()
np.linalg.det()
np.linalg.eig()

np.linalg.inv()
np.linalg.pinv()

np.linalg.qr()
np.linalg.svd()

np.linalg.solve()
np.linalg.lstsq()

#####################################################################
#                         6.随机数生成 (函数参数)
#####################################################################
np.random.seed(123)                    # 设置随机数
np.random.permutation(np.arange(16))   # 返回一个序列的随机排列或返回一个随机排列的范围
np.random.shuffle(np.arange(5))        # 对一个序列就地随机排列
np.random.rand((5)                     # 生成均匀分布随机数[0, 1)
np.random.uniform(10)                  # 均匀分布(0, 1)
np.random.randint()                    # 从给定范围内随机取整数

# ----------------------------------
# 正态分布
# ----------------------------------
np.random.normal(loc = 0, scale = 1, size = (6))
np.random.normal(size = (5))
# from random import normalvariate
# normalvariate(, )
np.random.randn()                      # 标准正态分布
np.random.normal(loc = 0, scale = 1, size = (6))
# ----------------------------------
# 其他分布
# ----------------------------------
np.random.binomial(5)
np.random.beta(5)
np.random.chisquare(5)
np.random.gamma(5)

#####################################################################
#                               其他
#####################################################################
np.array().reshape()
np.array().ravel()
np.array().flatten()
np.concatenate()
np.vstack()
np.hstack()
np.array().repeat([], axis = 1)
np.array().repeat([], axis = 0)
np.tile(np.array(), ())
```

