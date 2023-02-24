---
title: PCA
author: 王哲峰
date: '2022-09-13'
slug: ml-pca
categories:
  - machinelearning
tags:
  - model
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

- [数据降维](#数据降维)
- [降维的目的](#降维的目的)
- [常见降维模型](#常见降维模型)
- [主成分分析](#主成分分析)
  - [思想](#思想)
  - [原理](#原理)
  - [算法](#算法)
  - [优缺点](#优缺点)
  - [算法实现](#算法实现)
</p></details><p></p>

# 数据降维

维是对数据高维度特征的一种预处理方法. 降维是将高维度的数据保留下最重要的一些特征, 
去除噪声和不重要的特征, 从而实现提升数据处理速度的目的. 
在实际的生产和应用中, 降维在一定信息损失范围内, 
可以为我们节省大量的时间和成本. 降维也称为了应用非常广泛的数据预处理方法


# 降维的目的 

- 使得数据更容易使用
- 确保变量相互独立
- 降低很多算法的计算开销
- 去除噪音
- 使得结果易懂, 已解释

# 常见降维模型

- 主成分分析(Principal Components Analysis)
- 因子分析(Factor Analysis)
- 独立成分分析(Independ Component Analysis, ICA)

# 主成分分析

## 思想

- 去除平均值
- 计算协方差矩阵
- 计算协方差矩阵的特征值和特征向量
- 将特征值排序
- 保留前 N 个最大的特征值对应的特征向量
- 将数据转换到上面得到的N个特征向量构建的新空间中(实现了特征压缩)

## 原理

1. 找出第一个主成分的方向, 也就是数据方差最大的方向. 
2. 找出第二个主成分的方向, 也就是数据方差次大的方向, 
   并且该方向与第一个主成分方向正交(orthogonal 如果是二维空间就叫垂直). 
3. 通过这种方式计算出所有的主成分方向. 
4. 通过数据集的协方差矩阵及其特征值分析, 我们就可以得到这些主成分的值. 
5. 一旦得到了协方差矩阵的特征值和特征向量, 我们就可以保留最大的 N 个特征. 
   这些特征向量也给出了 N 个最重要特征的真实结构, 
   我们就可以通过将数据乘上这 N 个特征向量从而将它转换到新的空间上. 

## 算法

输入: `$m$` 个 `$n$` 维样本数据 `$D = (x^{(1)}, x^{(2)}, \ldots, x^{(m)})$`

输出: `$m$` 个 `$k$` 维样本数据

1. 对样本集进行标准化；
2. 计算样本的协方差矩阵 `$XX^{T}$`
3. 对协方差矩阵进行特征分解, 得到 `$n$` 个特征向量和其对应的特征值；
4. 取出最大的 `$k$` 个特征值对应的特征向量 `$(\omega_1, \omega_2, \ldots, \omega_k)$`, 
   将所有的特征向量标准化后, 组成特征向量矩阵 `$W$`
5. 对样本集中每一个样本 `$x^{(i)}$`, 转化为新的样本 `$z^{(i)}=W^{T}x^{(i)}$`
6. 得到输出的样本数据 `$D_{pca} = (z^{(1)}, z^{(2)}, \ldots, z^{(m)})$`

## 优缺点

- 优点: 降低数据复杂性, 识别最终要的多个特征
- 缺点: 
    - 可能损失有用信息
    - 只适用于数值型数据

## 算法实现

```python
import numpy as np
import pandas as pd


def loadData(fileName, delim = "\t"):
	data = pd.read_csv(fileName, sep = delim, header = None)

    return np.mat(data)

def PCA(dataMat, topNfeat = 9999999):
	meanVals = np.mean(dataMat, axis = 0)
	meanRemoved = dataMat - meanVals 					# 标准化
	covMat = np.cov(meanRemoved, rowvar = 0)		   	# 计算样本协方差矩阵
	eigVals, eigVects = np.linalg.eig(np.mat(covMat))	# 对样本协方差矩阵进行特征分解, 得到特征向量和对应的特征值
	eigValInd = np.argsort(eigVals)				   		# 对特征值进行排序
	eigValInd = eigValInd[:-(topNfeat + 1):-1]	   		# 取最大的topNfeat个特征向量对应的index序号
	redEigVects = eigVects[:, eigValInd]		   		# 根据取到的特征值对特征向量进行排序
	lowDDataMat = meanRemoved * redEigVects				# 降维之后的数据集
	reconMat = (lowDDataMat * redEigVects.T) + meanVals # 新的数据空间

	return lowDDataMat, reconMat

def show_picture(dataMat, reconMat):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(dataMat[:, 0].flatten().A[0], dataMat[:, 1].flatten().A[0], marker='^', s=90,c='green')
	ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0], marker='o', s=50, c='red')
	plt.show()


def main():
	data = loadData(fileName = "PCA.txt", delim = "\t")
	lowDDataMat, reconMat = PCA(data, 1)
	show_picture(data, reconMat)

if __name__ == "__main__":
    main()
```


* https://mp.weixin.qq.com/s?__biz=MjM5MjAxMDM4MA==&mid=2651890105&idx=1&sn=3c425c538dacd67b1732948c5c015b46&scene=21#wechat_redirect