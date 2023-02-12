---
title: 机器学习
subtitle: Machine Learning
list_pages: true
# order_by: title
---

# 文章

* [机器学习初学者易踩的5个坑](https://mp.weixin.qq.com/s/6r4FX7DEIUHE8pnYkftXSA)

# 机器学习系统的种类

1. 监督、无监督、半监督学习
2. 批量学习、在线学习
3. 基于实例、基于模型的学习

# 机器学习的主要挑战

1. 训练数据的数量不足
2. 训练数据不具有代表性
3. 数据质量差
4. 无关特征
5. 模型在训练数据上过度拟合
    - 如果模型对训练数据过拟合, 改进模型过拟合的方法之一是提供更多的训练数据, 直到模型验证误差接近训练误差；
    - 模型复杂度高, 对模型进行正则化: 
        - 多项式模型: 降低多项式的阶数；
        - 岭回归, Ridge Regression
        - Lasso回归, Lasso Regression
        - 弹性网络, Elastic Net
6. 模型在训练数据上欠拟合
    - 模型复杂度低
        - 如果模型对训练数据欠拟合, 添加更多的训练样本也没用, 此时需要使用更复杂的模型, 或找到更好的特征；
7.偏差/方差权衡
    - 模型的泛华误差可以被表示为三个不同的误差之和: 
        - 偏差: 这部分泛化误差的原因在于错误的假设, 比如假设数据是线性的, 而实际上是二次的. 高偏差模型最有可能对训练数据拟合不足；
        - 方差: 这部分误差是模型对训练数据的微小变化过度敏感导致的. 具有高自由度的模型, 很可能有高方差, 所以很容易对训练数据过拟合；
        - 不可避免的误差: 这部分误差是因为数据本身的噪声所致. 减少这部分误差的唯一方法是清理数据；
    - 增加模型的复杂度通常会显著提升模型的方差, 减少偏差；反过来, 降低模型的复杂度则会提升模型的偏差, 降低方差；

# 搭建 Python Machine Learning 项目环境

## 搭建 virtualenv 虚拟环境

```bash
# 创建项目目录
export ML_PATH="$HOME/project/mlenv"
mkdir -p $ML_PATH

# 更新Python包管理器pip
pip --version
pip install --upgrade pip

# 创建项目的虚拟环境
pip install --user --upgrade virtualenv
cd $ML_PATH
virtualenv env

# 激活虚拟环境
cd $ML_PATH
source env/bin/activate

# 安装模块及依赖项
pip install --upgrade jupyter matplotlib numpy pandas scipy scikit-learn
python -c "import jupyter, matplotlib, numpy, pandas, scipy, sklearn"

# 启动jupyter
jupyter notebook
```

## 搭建 pyenv 虚拟环境

```shell
# 创建项目目录
$ export ML_PATH="$HOME/project/mlenv"
$ mkdir -p $ML_PATH

# 更新Python包管理器pip
$ pip --version
$ pip install --upgrade pip

# 创建项目的虚拟环境
$ pip install pyenv

# 激活虚拟环境
$ cd $ML_PATH
$ pyenv local venv

# 安装模块及依赖项
pip install --upgrade jupyter matplotlib numpy pandas scipy scikit-learn
python -c "import jupyter, matplotlib, numpy, pandas, scipy, sklearn"

# 启动jupyter
jupyter notebook
```

## 搭建 conda 虚拟环境

```shell
# 创建项目目录
export ML_PATH="$HOME/project/mlenv"
mkdir -p $ML_PATH

# 更新Python包管理器pip

# 创建项目的虚拟环境

# 激活虚拟环境
cd $ML_PATH
source env/bin/activate

# 安装模块及依赖项

# 启动jupyter
jupyter notebook
```

# 文档

