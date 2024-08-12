---
title: Model Distilling
subtitle: 模型蒸馏
author: 王哲峰
date: '2023-03-17'
slug: model-distilling
categories:
  - deeplearning
tags:
  - model
---

模型蒸馏的思想就是利用一个已经训练好的、大型的、效果比较好的 Teacher 模型，去指导一个轻量型、
参数少的 Student 模型去训练——在减小模型的大小和计算资源的同时，
尽量把 Student 模型的准确率保证在 Teacher 模型附近。这种思想和方法在 Hinton 等的论文 Distilling the Knowledge in a Neural Network 中做了详细的介绍和说明。





# 参考

* [模型蒸馏](https://blog.csdn.net/HUSTHY/article/details/115174978)
* [Distilling the Knowledge in a Neural Network](https://arxiv.org/pdf/1503.02531.pdf)
