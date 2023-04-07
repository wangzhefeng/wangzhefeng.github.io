---
title: ReinForcement Learning
author: 王哲峰
date: '2022-07-15'
slug: dl-reinforcementlearning
categories:
  - deeplearning
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

- [深度强化学习](#深度强化学习)
- [强化学习示例](#强化学习示例)
  - [开发环境](#开发环境)
  - [CartPole 示例](#cartpole-示例)
</p></details><p></p>

# 深度强化学习

强化学习(Reinforcement learning, RL)强调如何基于环境而行动, 以取得最大化的预期利益。
结合了深度学习技术后的强化学习更是如虎添翼。近两年广为人知的 AlphaGo 既是深度强化学习(DRL)的典型应用。

# 强化学习示例

以下示例使用深度强化学习玩 CartPole(倒立摆)游戏, 倒立摆是控制论中的经典问题, 
在这个游戏中, 一根杆的底部与一个小车通过轴相连, 而杆的重心在轴之上, 因此是一个不稳定的系统。
在重力的作用下, 杆很容易倒下。需要控制小车在水平的轨道上进行左右运动, 使得杆一直保持竖直平衡状态。

## 开发环境

- OpenAI 推出的 Gym 环境库中的 CartPole 游戏环境
    - 和 Gym 的交互过程很像一个回合制游戏：
        - 首先, 获得游戏的初始状态(比如杆的角度、小车位置)
        - 然后, 在每个回合, 都需要在当前可行的动作中选择一个并交由 Gym 执行
            - 比如：向左或者右推动小车, 每个回合中二者只能选择一
        - Gym 在执行动作后, 会返回动作执行后的下一个状态和当前回合所获得的奖励值
            - 比如：选择向左推动小车并执行后, 小车位置更加偏左, 而杆的角度更加偏右, Gym 将新的角度和位置返回, 
              而如果在这一回合杆没有倒下, Gym 同时返回一个小的正奖励
        - 上述过程可以一直迭代下去, 直到游戏结束
            - 比如：杆倒下了

- Gym 环境库中的 CartPole 游戏环境库安装

```bash
$ pip install gym
```

- Gym 的 Python 基本调用方法

```python
import gym

env = gym.make("CartPole-v1")                         # 实例化一个游戏环境, 参数为游戏名称
state = env.reset()                                   # 初始化环境, 获得初始状态
while True:
    env.render()                                      # 对当前帧进行渲染, 绘图到屏幕
    action = model.predict(state)                     # 假设我们有一个训练好的模型, 能够通过当前状态预测出这时应该进行的动作
    next_state, reward, done, info = env.step(action) # 让环境执行动作, 获得执行完动作的下一个状态, 动作的奖励, 游戏是否一结束以及额外信息
    if done:                                          # 如果游戏结束, 则退出循环
        break
```

## CartPole 示例

- 任务

    - 训练出一个模型, 能够根据当前的状态预测出应该进行的一个好的动作。
      粗略地说, 一个好的动作应当能够最大化整个游戏过程中获得的奖励之和, 
      这也是强化学习的目标。
    - CartPole 游戏中的目标是, 希望做出的合适的动作使得杆一直不倒, 
      即游戏交互的回合数尽可能多, 而每进行一回合, 都会获得一个小的正奖励, 
      回合数越多则积累的奖励值也越高。因此, 最大化游戏过程中的奖励之和与最终目标是一致的。
    - 使用深度强化学习中的 Deep Q-Learning 方法来训练模型

1. 首先, 引入常用库, 定义一些模型超参数

```python
import tensorflow as tf
```

2.使用 `tf.keras.Model` 建立一个 Q 函数网络, 用于拟合 Q-Learning 中的 Q 函数

    - 使用较简单的多层全连接神经网络进行拟合, 该网络输入当前状态, 输入各个动作下的 Q-Value(CartPole 下为二维, 即向左和向右推动小车)

```python
class QNetwork(tf.keras.Model):
    pass
```
