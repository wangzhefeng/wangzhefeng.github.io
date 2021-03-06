---
title: 优化算法
author: 王哲峰
date: '2022-07-15'
slug: dl-model-optimization
categories:
  - deeplearning
tags:
  - model
---

优化算法
========

1.神经网络模型学习算法优化
---------------------------------------------

   深度学习算法在很多种情况下都涉及优化。在深度学习涉及的诸多优化问题中, 最难的是神经网络训练, 
   这里的学习算法优化就特指神经网络训练中的优化问题:寻找神经网络上的一组参数 `$\theta`, 
   它能显著地降低代价函数 `$J(\theta) = E_{(x, y) \backsim \hat{p}_{data}}L(f(x;\theta), y)$`, 
   该代价函数通常包括整个训练集上的性能评估和额外的正则化项.

1.1 经验风险最小化
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




1.2 代理损失函数和提前终止
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




1.3 批量算法和小批量算法
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   1. 梯度下降算法(Gradient Descent, GD) 
   2. 小批量梯度下降算法(mini-batch Gradient Descent)

      - 在实际的数据应用场景, 直接对大批量数据执行梯度下降法训练模型时, 机器处理数据的速度会非常缓慢, 
         这时将训练数据集分割成小一点的子集进行多次训练非常重要。这个被分割成的小的训练数据子集就做 mini-batch, 
         意为小批量。对每个小批量数据集同时执行梯度下降法会大大提高训练效率。在实际利用代码实现的时候, 
         小批量梯度下降算法通常包括两个步骤:充分打乱数据(shuffle)和分组组合数据【分区】(partition)

   3. 随机梯度下降算法(Stochastic Gradient Descent, SGD)
   4. 带动量的梯度下降算法(Gradient Descent with Momentum)
   5. 自适应矩估计(Adaptive Moment Estimation, Adam)
   6. 加速梯度下降算法(RMSprop)

2.神经网络优化中的挑战
-------------------------------------

2.1 病态
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

2.2 局部最小值
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


2.3 高原、鞍点和其他平坦区域
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


2.4 悬崖和梯度爆炸
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

2.5 长期依赖
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

2.6 非精确梯度
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

2.7 局部和全局结构间的弱对应
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

2.8 优化的理论限制
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

3.梯度算法
-------------------------------------

机器学习和神经网络的学习都是从训练数据集中学习时寻找最优参数(权重和偏置), 这里的最优参数就是损失函数取最小值时的参数。
一般而言, 损失函数很复杂, 参数空间庞大, 通过使用梯度来寻找损失函数最小值(或尽可能小的值)的方法就是 **梯度法**。

   - 梯度表示的是一个函数在各点处的函数值减小最多的方向, 因此, 无法保证梯度所指的方向就是函数的最小值或者真正应该前进的方向。实际上在复杂的函数中, 梯度指示的方向基本上都不是函数值最小处

      - 函数的极小值, 最小值, 鞍点(saddle point)的地方, 梯度为0

         - 最小值是指全局最小值

         - 极小值是指局部最小值, 也就是限定在某个范围内的最小值

         - 鞍点是从某个方向上看是极大值, 从另一个方向上看则是极小值的点; 

   - 虽然梯度的方向并不一定指向最小值, 但沿着它的方向能够最大限度地减小函数的值。因此, 在寻找函数的最小值(或者尽可能小的值)的位置的任务中, 要以梯度的信息为线索, 决定前进的方向


3.1 梯度算法
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在梯度算法中, 函数的取值从当前位置沿着梯度方向前进一段距离, 然后在新的地方重新求梯度, 再沿着新梯度方向前进, 如此反复, 不断地沿着梯度方向前进, 逐渐减小函数的值, 梯度算法可以分为两种, 但常用的是梯度下降算法:

- 梯度下降算法(gradient descent method)

- 梯度上升算法(gradient ascent method)

**梯度下降法算法的数学表示:**

`$\omega_i^{(t)} = \omega_i^{(t)} - \eta^{(t)} \frac{\partial l}{\partial \omega_i^{(t)}}$`

其中:

- `$\omega_i^{(t)}$` :在第 `$t$` 轮迭代时的第 `$i`
   个参数; 

- `$l$` :损失函数; 

- `$\eta^{(t)}$` :第 `$t$` 轮迭代时的学习率 (learning
   rate), 决定在一次学习中, 应该学习多少, 以及在多大程度上更新参数; 实验表明, 设定一个合适的
   learning rate 是一个很重要的问题:

   - 学习率过大, 会发散成一个很大的值; 

   - 学习率过小, 基本上没怎么更新就结束了; 


3.2 神经网络学习算法
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**神经网络的学习步骤:**

   - 前提:
      - 神经网络存在合适的权重和偏置, 调整权重和偏置以便你和训练数据的过程称为“学习”
   - 步骤1:mini-batch
      - 从训练数据中随机选出一部分数据, 这部分数据称为mini-batch。目标是减小 mini-batch 的损失函数的值; 
   - 步骤2:计算梯度
      - 为了减小 mini-batch 的损失函数的值, 需要求出各个权重参数的梯度; 梯度表示损失函数的值减小最多的方向; 
   - 步骤3:更新参数
      - 将权重参数沿梯度方向进行微小更新; 
   - 步骤4:(重复)
      - 重复步骤 1, 步骤 2, 步骤 3

**梯度算法(Gradient Descent, GD)改进:**

   在深度学习实际的算法调优中, 原始的梯度下降法一般不大好用。通常来说, 工业环境下深度学习所处理的数据量都是相当大的。
   这时若直接使用原始版本的梯度下降, 可能训练速度和运算效率会非常低。这时候就需要采取一些策略对原始的梯度下降法进行调整来加速训练过程。

   - 改进1:

      - `随机梯度下降:从 GD 到 mini-batch GD, SGD`
      
      - `mini-batch GD`:将训练数据划分为小批量(mini-batch)进行训练
         - 将训练集划分为一个个子集的小批量数据, 相较于原始的整体进行梯度下降的方法, 整个神经网络的训练效率会大大提高

      - `SGD`:如果批量足够小, 小到一批只有一个样本, 这时算法就是随机梯度下降(SGD)

         - 使用随机梯度下降算法, 模型训练起来会很灵活, 数据中的噪声也会得到减小, 但是随机梯度下降会有一个劣势就是失去了向量化运算带来的训练加速度, 
            算法也较难收敛, 因为一次只处理一个样本, 虽然足够灵活但效率过于低下

      - 在深度学习模型的实际处理中, 选择一个合适的 `batch-size` 是一个比较重要的问题

         - 一般而言需要视训练的数据量来定, 也需要不断的试验

            - 通常而言, batch-size 过小会使得算法偏向 SGD 一点, 失去向量化带来的加速效果, 算法也不容易收敛
            - 但若是盲目增大 batch-size, 一方面会比较吃内存, 另一方面是梯度下降的方向很难再有变化, 进而影响训练精度
            - 所以一个合适的 batch-size 对于深度学习的训练来说就非常重要, 合适的 batch-size 会提高内存的利用率, 
               向量化运算带来的并行效率提高, 跑完一次 epoch 所需要的迭代次数也会减少, 训练速度会加快。这便是小批量 (mini-batch) 梯度下降 batch-size 的作用

      - 无论是梯度下降法(GD)、小批量 (mini-batch) 梯度下降法还是随机梯度下降法(SGD), 它们的本质都是基于梯度下降的算法策略, 三者的区别即在于执行一次运算所需要的样本量

   - 改进2:

      - `动量梯度下降`:从 Momentum 到 Adam
      - `Momentum GD`:基于移动加权的思想, 给梯度下降加上历史梯度的成分来进一步加快下降速度, 这种基于历史梯度和当前梯度进行加权计算的梯度下降法便是动量梯度下降法(Momentum GD)
         - 动量梯度下降算法公式

3.3 基本算法
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

3.3.1 SGD
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- SGD的数学表示:
 
`$$W \leftarrow W - \eta \frac{\partial L}{\partial W}$$`

- 其中:
   - `$W$`: 需要更新的权重参数
   - `$\frac{\partial L}{\partial W}$`: 损失函数关于权重参数 `$W$` 的梯度
   - `$\eta$`: 学习率(learning rate), 事先决定好的值, 比如: 0.001, 0.01

- SGD的Python实现:

.. code:: python

   class SGD:
      def __init__(self, lr = 0.01):
         self.lr = lr

      def update(self, params, grads):
         for key in params.keys():
            params[key] -= self.lr * grads[key]

- SGD的缺点:

   - 低效

      - 如果损失函数的形状非均向(anisotropic), 比如呈延伸状, 搜索的路径就会非常低效
      - SGD低效的根本原因是:梯度的方向并没有指向最小值的方向


3.3.2 Momentum SGD(动量随机梯度下降法)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Momentum SGD的数学表示:


`$$\upsilon \leftarrow \alpha \upsilon - \eta \frac{\partial L}{\partial W}$$`
`$$W \leftarrow W + \upsilon$$`

其中:

   - `$W$`: 需要更新的权重参数
   - `$\frac{\partial L}{\partial W}$`: 损失函数关于权重参数 `$W` 的梯度
   - `$\eta$`: 学习率(learning rate), 事先决定好的值, 比如: 0.001, 0.0
   - `$\upsilon$`: 对应物理上的速度, `$\upsilon$` 的更新表示了物体在梯度方向上受力, 在这个力的作用下, 物体的速度增加
   - `$\alpha \upsilon$`: 对应了物理上的地面摩擦或空气阻力, 表示在物体不受任何力时, 该项承担使物体逐渐减速的任务(`$\alpha$` 一般设定为0.9)

- Momentum SGD的Python实现:

.. code:: python

   class Momentum:
      def __init__(self, lr = 0.01, momentum = 0.9):
         self.lr = lr
         self.momentum = momentum
         self.v = None

      def update(self, params, grads):
         if self.v is None:
            self.v = {}
            for key, val in params.items():
               self.v[key] = np.zeros_like(val)

         for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * self.grads[key]
            params[key] += self.v[key]

- Momentum SGD的优缺点:

   - 为什么会好？


3.3.3 Nesterov 动量
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

3.4 自适应学习率算法
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

3.4.1 AdaGrad
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   - 在神经网络的学习中, 学习率(learning rate, lr or `$\eta$`)的值很重要。

   - 学习率过小, 会导致学习花费过多的时间; 

   - 学习率过大, 会导致学习发散而不能正确进行; 

   - `学习率衰减(learning rate decay)$` :随着学习的进行, 使学习率逐渐减小; 

   - 逐渐减小学习率的想法, 相当于将“全体”参数的学习率一起降低; 

   - AdaGrad发展了学习率衰减的想法, 针对一个一个的参数, 赋予其“定制”的值; 

   - AdaGrad会为参数的每个元素适当地调整学习率, 与此同时进行学习; 

- AdaGrad SGD的数学表示:

`$$h \leftarrow h + \frac{\partial L}{\partial W} \odot \frac{\partial L}{\partial W}$$`
`$$W \leftarrow W - \eta \frac{1}{\sqrt{h}} \frac{\partial L}{\partial W}$$`

   其中:

      - `$W$`: 需要更新的权重参数; 
      - `$\frac{\partial L}{\partial W}$`: 损失函数关于权重参数 `$W$` 的梯度; 
      - `$\eta$`: 学习率(learning rate), 事先决定好的值, 比如:0.001, 0.01; 
      - `$h$`: 保存了以前所有梯度值的平方和, 然后, 在更新参数时, 通过乘以 `$\frac{1}{\sqrt{h}}$` 就可以调整学习的尺度
         - 参数的元素中变动较大(被大幅更新)的元素的学习率将变小, 也就是说, 可以按照参数的元素进行学习率衰减, 使变动的参数的学习率逐渐减小; 

- AdaGrad SGD的Python实现:

   .. code:: python

      class AdaGrad:
         def __init__(self, lr = 0.01):
            self.lr = lr
            self.h = None

         def update(self, params, grads):
            if self.h is None:
               self.h = {}
               for key, val in params.items():
                  self.h[key] = np.zeros_like(val)

            for key in params.keys():
               self.h[key] += grads[key] * grads[key]
               param[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)

- AdaGrad SGD的优缺点:

   - AdaGrad会记录过去所有梯度的平方和。因此, 学习越深入, 更新的幅度就越小。实际上, 如果无止境地学习, 更新量就会变为0, 完全不再更新。
   - 为了改善这个问题, 可以使用RMSProp方法。RMSProp方法并不是将过去所有的梯度一视同仁地相加, 而是逐渐地遗忘过去的梯度, 
      在做加法运算时将新的梯度的信息更多地反映出来。这种操作从专业上讲, 称为“指数移动平均”, 呈指数函数式地减小过去的梯度的尺度。


:

3.4.2 RMSProp
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- RMSProp的数学表示:

- RMSProp的Python实现:

- RMSProp的优缺点:


3.4.3 Adam
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   Adam 融合了 Momentum 和 AdaGrad 的方法, 通过组合两个方法的优点, 有希望实现参数空间的高效搜索。并且 Adam 会进行超参数的“偏置校正”.

- Adam的数学表示:

- Adam的Python实现:

- Adam的优缺点