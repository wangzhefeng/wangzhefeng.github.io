---
title: LightGBM
author: 王哲峰
date: '2022-07-31'
slug: ml-gbm-lightgbm
categories:
  - machinelearning
tags:
  - ml
  - model
---


1.LightGBM 特点
----------------

   LightGBM is a gradient boosting framework that uses tree based learning algorithms. It is designed to be distributed and efficient with the following advantages:

      - Faster training speed and higher efficiency.
      
      - Lower memory usage.
      
      - Better accuracy.
      
      - Support of parallel and GPU learning.
      
      - Capable of handling large-scale data.

2.LightGBM 资源
----------------

   - `原始算法论文 <https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf>`_ 

   - `GitHub-Python-Package <https://github.com/Microsoft/LightGBM/tree/master/python-package>`_ 

   - `GitHub-R-Package <https://github.com/Microsoft/LightGBM/tree/master/R-package>`_ 

   - `GitHub-Microsoft <https://github.com/Microsoft/LightGBM>`_ 

   - `Doc <https://lightgbm.readthedocs.io/en/latest/>`_ 

   - `Python 示例 <https://github.com/microsoft/LightGBM/tree/master/examples/python-guide>`_ 

3.LightGBM 安装
----------------------------

   -  CLI 版本

      -  Win
      -  Linux
      -  OSX
      -  Docker
      -  Build MPI 版本
      -  Build GPU 版本

   -  Python library

      -  安装依赖库
      -  安装 ``lightgbm``

3.1 CLI(Command Line Interface) 安装
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   MacOS 上安装 LightGBM(CLI) 有以下三种方式:

      - Apple Clang
      
         - Homebrew

         - CMake
         
            - Build from GitHub

      - gcc

         - Build from GitHub

1.Homebrew
''''''''''''''

   安装 LightGBM:

      .. code-block:: shell

         $ brew install lightgbm

2.使用 CMake 从 GitHub 上构建
'''''''''''''''''''''''''''''

   (1)安装 CMake(3.16 or higher)

      .. code-block:: shell

         $ brew install cmake

   (2)安装 OpenMP

      .. code-block:: shell
      
         $ brew install libomp

   (3)构建 LightGBM

      .. code-block:: shell

         $ git clone --recursive https://github.com/microsoft/LightGBM
         $ cd LightGBM
         $ cmake ..
         $ make -j4

3.使用 gcc 从 GitHub 上构建
'''''''''''''''''''''''''''

   (1)安装 CMake(3.2 or higher)

      .. code-block:: shell

         $ brew install cmake

   (2)安装 gcc

      .. code-block:: shell
      
         $ brew install gcc

   (3)构建 LightGBM

      .. code-block:: shell

         $ git clone --recursive https://github.com/microsoft/LightGBM
         $ cd LightGBM
         $ export CXX=g++-7 CC=gcc-7  # replace "7" with version of gcc installed on your machine
         $ mkdir build
         $ cd build
         $ cmake ..
         $ make -j4

3.2 Python Package 安装
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- pip 安装

   - 安装 ``lightgbm``:

      .. code-block:: shell

         # 默认版本
         $ pip install lightgbm

         # MPI 版本
         $ pip install lightgbm --install-option=--mpi

         # GPU 版本
         $ pip install lightgbm --install-option=--gpu

   - 约定成俗的库导入:

      .. code-block:: python

         import lightgbm as lgb

- 从源码构建

   .. code-block:: shell

      git clone --recursive https://github.com/microsoft/LightGBM
      cd LightGBM
      mkdir build
      cd build
      cmake ..

      # 开启MPI 通信机制, 训练更快
      cmake -DUSE_MPI=ON ..

      # GPU 版本, 训练更快
      cmake -DUSE_GPU=1 ..
      make -j4



4.LightGBM 数据接口
---------------------

数据接口:

   -  LibSVM(zero-based), TSV, CSV, TXT 文本文件

   -  Numpy 2 维数组

   -  pandas DataFrame

   -  H2O DataTable’s Frame

   -  SciPy sparse matrix

   -  LightGBM 二进制文件

   .. note:: 

      - 数据保存在 ``Dataset`` 对象中.

4.1 加载 LibSVM(zero-based) 文本文件、LightGBM 二进制文件
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   .. code-block:: python

      import lightgbm as lgb

      # csv
      train_csv_data = lgb.Dataset('train.csv')

      # tsv
      train_tsv_data = lgb.Dataset('train.tsv')

      # libsvm
      train_svm_data = lgb.Dataset('train.svm')

      # lightgbm bin
      train_bin_data = lgb.Dataset('train.bin')


4.2 加载 Numpy 2 维数组
~~~~~~~~~~~~~~~~~~~~~~~~~~~

   .. code-block:: python

      import liggtgbm as lgb

      data = np.random.rand(500, 10)
      label = np.random.randint(2, size = 500)
      train_array = lgb.Dataset(data, label = label)

4.3 加载 scipy.sparse.csr_matrix 数组
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   .. code-block:: python

      import lightgbm as lgb
      import scipy

      csr = scipy.sparse.csr_matirx((dat, (row, col)))
      train_sparse = lgb.Dataset(csr)

4.4 保存数据为 LightGBM 二进制文件
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   .. code-block:: python

      import lightgbm as lgb

      train_data = lgb.Dataset("train.svm.txt")
      train_data.save_binary('train.bin')

   .. note:: 

      - 将数据保存为 LightGBM 二进制文件会使数据加载更快


4.5 创建验证数据
~~~~~~~~~~~~~~~~~~~

   .. code-block:: python
   
      import lightgbm as lgb

      # 训练数据
      train_data = lgb.Dataset("train.csv")
      
      # 验证数据
      validation_data = train_data.create_vaild('validation.svm')
      # or
      validation_data = lgb.Dataset('validation.svm', reference = train_data)

.. note:: 

   - 在 LightGBM 中, 验证数据应该与训练数据一致(格式)

4.6 在数据加载时标识特征名称和类别特征
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   .. code-block:: python

      import numpy as np
      import lightgbm as lgb

      data = np.random.rand(500, 10)
      label = np.random.randint(2, size = 500)
      train_array = lgb.Dataset(data, label = label)
      w = np.random.rand(500, 1)

      train_data = lgb.Dataset(data, 
                              label = label, 
                              feature_name = ['c1', 'c2', 'c3'], 
                              categorical_feature = ['c3'],
                              weight = w,
                              free_raw_data = True)
      # or
      train_data.set_weight(w)
      
      train_data.set_init_score()

      train_data.set_group()


4.6 有效利用内存空间
~~~~~~~~~~~~~~~~~~~~~~~~

   The Dataset object in LightGBM is very memory-efficient, 
   it only needs to save discrete bins. However, Numpy/Array/Pandas object is memory expensive. 
   If you are concerned about your memory consumption, you can save memory by:

      - 1.Set ``free_raw_data=True`` (default is ``True``) when constructing the Dataset

      - 2.Explicitly set ``raw_data=None`` after the Dataset has been constructed

      - Call ``gc``


5.LightGBM 设置参数
-----------------------

   - 参数设置方式: 

      -  命令行参数

      -  参数配置文件

      -  Python 参数字典

   - 参数类型:

      -  核心参数

      -  学习控制参数

      -  IO参数

      -  目标参数

      -  度量参数

      -  网络参数

      -  GPU参数

      -  模型参数

      -  其他参数


5.1 Booster 参数
~~~~~~~~~~~~~~~~~~~~~

   .. code-block:: python

      param = {
         'num_levels': 31,
         'num_trees': 100,
         'objective': 'binary',
         'metirc': ['auc', 'binary_logloss']
      }


6.LightGBM 应用
------------------------------

6.1 训练、保存、加载模型
~~~~~~~~~~~~~~~~~~~~~~~~~~

   .. code-block:: python

      # 训练模型
      
      import lightgbm as lgb

      # 训练数据
      train_data = lgb.Dataset("train.csv")
      
      # 验证数据
      validation_data = train_data.create_vaild('validation.svm')

      # 参数
      param = {
         'num_levels': 31,
         'num_trees': 100,
         'objective': 'binary',
         'metirc': ['auc', 'binary_logloss']
      }
      num_round = 10

      # 模型训练
      bst = lgb.train(param, train_data, num_round, vaild_sets = [validation_data])
      
      # 保存模型
      bst.save_model('model.txt')
      json_model = bst.dump_model()

      # 加载模型
      bst = lgb.Booster(model_file = 'model.txt')

6.2 交叉验证
~~~~~~~~~~~~~~~~~~~~~~

   .. code-block:: python

      num_round = 10
      lgb.cv(param, train_data, num_round, nfold = 5)

6.3 提前停止
~~~~~~~~~~~~~~~~~~~~~~

   .. code-block:: python

      bst = lgb.train(param,
                      train_data,
                      num_round,
                      valid_sets = valid_sets,
                      ealy_stopping_rounds = 10)

6.4 预测
~~~~~~~~~~~~~~~~~~~~~~

   -  用已经训练好的或加载的保存的模型对数据集进行预测

   -  如果在训练过程中启用了提前停止, 可以用 ``bst.best_iteration`` 从最佳迭代中获得预测结果

   .. code-block:: python

      testing = np.random.rand(7, 10)
      y_pred = bst.predict(testing, num_iteration = bst.best_iteration)




7.LightGBM API
-----------------

7.1 Data Structure API
~~~~~~~~~~~~~~~~~~~~~~~~

   -  ``Dataset(data, label, reference, weight, ...)``

   -  ``Booster(params, train_set, model_file, ...)``



7.2 Training API
~~~~~~~~~~~~~~~~~~~~~~~~

   -  ``train(params, train_set, num_boost_round, ...)``

   -  ``cv(params, train_ste, num_boost_round, ...)``



7.3 Scikit-learn API
~~~~~~~~~~~~~~~~~~~~~~~~

   -  ``LGBMModel(boosting\ *type, num*\ leaves, ...)``

   -  ``LGBMClassifier(boosting\ *type, num*\ leaves, ...)``

   -  ``LGBMRegressor(boosting\ *type, num*\ leaves, ...)``

   -  ``LGBMRanker(boosting\ *type, num*\ leaves, ...)``


.. code-block:: python

   lightgbm.LGBMClassifier(boosting_type = "gbdt", # gbdt, dart, goss, rf
                           num_leaves = 31, 
                           max_depth = -1, 
                           learning_rate = 0.1,
                           n_estimators = 100,
                           subsample_for_bin = 200000,
                           objective = None, 
                           class_weight = None,
                           min_split_gain = 0.0,
                           min_child_weight = 0.001, 
                           min_child_samples = 20,
                           subsample = 1.0,
                           subsample_freq = 0,
                           colsample_bytree = 1.0,
                           reg_alpha = 0.0,
                           reg_lambda = 0.0,
                           random_state = None,
                           n_jobs = -1, 
                           silent = True,
                           importance_type = "split",
                           **kwargs)

   lgbc.fit(X, y,
            sample, 
            weight = None, 
            init_score = None,
            eval_set = None,
            eval_names = None, 
            eval_sample_weight = None,
            eval_class_weight = None,
            eval_init_score = None,
            eval_metric = None,
            early_stopping_rounds = None,
            verbose = True,
            feature_name = "auto",
            categorical_feature = "auto",
            callbacks = None)

   lgbc.predict(X, 
                raw_score = False,
                num_iteration = None,
                pred_leaf = False,
                pred_contrib = False,
                **kwargs)

   lgbc.predict_proba(X, 
                      raw_score = False,
                      num_iteration = None,
                      pred_leaf = False,
                      pred_contrib = False,
                      **kwargs)


.. code-block:: python

   lightgbm.LGBMRegressor(boosting_type = "gbdt",
                          num_leaves = 31,
                          max_depth = -1,
                          learning_rate = 0.1,
                          n_estimators = 100,
                          subsample_for_bin = 200000,
                          objective = None,
                          class_weight = None,
                          min_split_gain = 0.0,
                          min_child_weight = 0.001,
                          min_child_samples = 20,
                          subsample = 1.0,
                          subsample_freq = 0,
                          colsample_bytree = 1.0,
                          reg_alpha = 0.0,
                          reg_lambda = 0.0,
                          random_state = None,
                          n_jobs = -1,
                          silent = True,
                          importance_type = "split",
                          **kwargs)

   lgbr.fit(X, y, sample_weight = None,
            init_score = None, 
            eval_set = None,
            eval_names = None,
            eval_sample_weight = None,
            eval_init_score = None,
            eval_metric = None,
            early_stopping_rounds = None,
            verbose = True,
            feature_name = "auto",
            categorical_feature = "auto",
            callbacks = None)

   lgbr.predict(X, 
                raw_score = False, 
                num_iteration = None, 
                pred_leaf = False,
                pred_contrib = False,
                **kwargs)



7.4 Callbacks
~~~~~~~~~~~~~~~~~~~~~~~~

   -  ``early_stopping(stopping_round, ...)``

   -  ``print_evaluation(period, show_stdv)``

   -  ``record_evaluation(eval_result)``

   -  ``reset_parameter(**kwargs)``

.. code-block:: python

   early_stopping(stopping_round, ...)
   print_evaluation(period, show_stdv)
   record_evaluation(eval_result)
   reset_parameter(**kwargs)



7.5 Plotting
~~~~~~~~~~~~~~~~~~~~~~~~

   -  ``plot_importance(booster, ax, height, xlim, ...)``
   -  ``plot_split_value_histogram(booster, feature)``
   -  ``plot_metric(booster, metric, ...)``
   -  ``plot_tree(booster, ax, tree_index, ...)``
   -  ``create_tree_digraph(booster, tree_index, ...)``

   .. code-block:: python

      plot_importance(booster, ax, height, xlim, ...)
      plot_split_value_histogram(booster, feature)
      plot_metric(booster, ax, tree, index, ...)
      plot_tree(booster, ax, tree_index, ...)
      create_tree_digraph(booster, tree_index, ...)

8.LightGBM 示例
-----------------------------------

8.1 示例 1: 常用操作总结
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. note::
   
   - 人工调参

      - 提高速度

         - Use bagging by setting bagging_fraction and bagging_freq
         - Use feature sub-sampling by setting feature_fraction
         - Use small max_bin
         - Use save_binary to speed up data loading in future learning
         - Use parallel learning, refer to Parallel Learning Guide

      - 提高准确率

         - Use large max_bin (may be slower)
         - Use small learning_rate with large num_iterations
         - Use large num_leaves (may cause over-fitting)
         - Use bigger training data
         - Try dart
      
      - 处理过拟合

         - Use small max_bin
         - Use small num_leaves
         - Use min_data_in_leaf and min_sum_hessian_in_leaf
         - Use bagging by set bagging_fraction and bagging_freq
         - Use feature sub-sampling by set feature_fraction
         - Use bigger training data
         - Try lambda_l1, lambda_l2 and min_gain_to_split for regularization
         - Try max_depth to avoid growing deep tree
         - Try extra_trees
         - Try increasing path_smooth





8.2 示例 2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

