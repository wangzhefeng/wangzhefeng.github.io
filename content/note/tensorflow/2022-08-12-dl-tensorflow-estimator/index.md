---
title: TensorFlow Estimator
author: 王哲峰
date: '2022-08-12'
slug: dl-tensorflow-estimator
categories:
  - deeplearning
  - tensorflow
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

- [预创建的 Estimator](#预创建的-estimator)
- [自定义的 Estimator](#自定义的-estimator)
- [从 Keras 模型创建 Estimator](#从-keras-模型创建-estimator)
</p></details><p></p>


- 一种可极大地简化机器学习编程的高阶TensorFlow API; 
- Estimator封装的操作:
   - 训练
   - 评估
   - 预测
   - 导出以使用
- Estimator优势:
   - 可以在本地主机上或分布式多服务器环境中运行基于 Estimator
      的模型, 而无需更改模型。此外, 可以在 CPU、GPU 或 TPU 上运行基于
      Estimator 的模型, 而无需重新编码模型
   - Estimator 简化了在模型开发者之间共享实现的过程
   - 可以使用高级直观代码开发先进的模型。简言之, 采用 Estimator
      创建模型通常比采用低阶 TensorFlow API 更简单
   - Estimator 本身在 tf.layers 之上构建而成, 可以简化自定义过程
   - Estimator 会为您构建图
   - Estimator 提供安全的分布式训练循环, 可以控制如何以及何时:
      - 构建图
      - 初始化变量
      - 开始排队
      - 处理异常
      - 创建检查点文件并从故障中恢复
      - 保存 TensorBoard 的摘要

# 预创建的 Estimator

**预创建的 Estimator 程序的结构**

**依赖预创建的Estimator的TensorFlow程序通常包含下列四个步骤:**

1. 编写一个或多个数据集导入函数
   - 创建一个函数来导入训练集, 并创建另一个函数来导入测试集。每个数据集导入函数都必须返回两个对象:
      - 一个字典, 其中键是特征名称, 值是包含相应特征数据的张量(or Sparse Tensro); 
      - 一个包含一个或多个标签的张量; 
1. 定义特征列
   - 每个 `tf.feature_column` 都标识了特征名称、特征类型和任何输入预处理操作
2. 实例化相关的预创建的Estimator
   - LinearClassifier
4. 调用训练、评估或推理方法
   - 所有Estimator都提供训练模型的 `train` 方法

**上面步骤实现举例:**

```python
def input_fn_train(dataset):
   # manipulate dataset, extracting the feature dict and the label
   
   return feature_dict, label

def input_fn_test(dataset):
   # manipulate dataset, extracting the feature dict and the label
   
   return feature_dict, label


my_training_set = input_fn_train()
my_testing_set = input_fn_test()

population = tf.feature_column.numeric_column('population')
crime_rate = tf.feature_column.numeric_column('crime_rate')
median_education = tf.feature_column.numeric_column('median_education', 
                                                   normalizer_fn = lambda x: x - global_education_mean)

estimator = tf.estimator.LinearClassifier(
   feature_columns = [population, crime_rate, median_education],
)

estimator.train(input_fn = my_training_set, setps = 2000)
```

**预创建的 Estimator 的优势**

- 预创建的 Estimator 会编码最佳做法, 从而具有下列优势:
   - 确定计算图不同部分的运行位置以及在单台机器或多台机器上实现策略的最佳做法。
   - 事件(汇总)编写和普遍有用的汇总的最佳做法。


# 自定义的 Estimator

- 每个
   Estimator(无论是预创建还是自定义)的核心都是其模型函数, 这是一种为训练、评估和预测构建图的方法。如果您使用预创建的
   Estimator, 则有人已经实现了模型函数。如果您使用自定义
   Estimator, 则必须自行编写模型函数。

- 推荐的工作流程:

   - 1.假设存在合适的预创建的Estimator, 使用它构建第一个模型并使用其结果确定基准; 
   - 2.使用此预创建的Estimator构建和测试整体管道, 包括数据的完整性和可靠性; 
   - 3.如果存在其他合适的预创建的Estimator, 则运行试验来确定哪个预创建的Estimator效果好; 
   - 4.可以通过构建自定义的Estimator进一步改进模型; 


# 从 Keras 模型创建 Estimator


- 可以将现有的Keras的模型转换为Estimator, 这样Keras模型就可以利用Estimator的优势, 比如进行分布式训练; 

```python
keras_inception_v3 = tf.keras.applications.keras_inception_v3.InceptionV3(weights = None)

keras_inception_v3.compile(optimizer = tf.keras.optimizers.SGD(lr = 0.0001, momentum = 0.9),
                           loss = 'categorical_crossentropy',
                           metric = 'accuracy')

est_inception_v3 = tf.keras.estimator.model_to_estimator(keras_model = keras_inception_v3)

keras_inception_v3.input_names

train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
      x = {'input_1': train_data},
      y = train_labels,
      num_epochs = 1,
      shuffle = False
)

est_inception_v3.train(input_fn = train_input_fn, steps = 2000)
```

**API:**

从一个给定的Keras模型中构造一个Estimator实例

```python
tf.keras.estimator.model_to_estimator(
      keras_model = None,
      keras_model_path = None,
      custom_objects = None,
      model_dir = None,
      config = None
)
```


