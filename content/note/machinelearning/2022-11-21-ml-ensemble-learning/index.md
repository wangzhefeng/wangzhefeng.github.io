---
title: 集成学习 Ensemble Learning
author: 王哲峰
date: '2022-11-21'
slug: ml-ensemble-learning
categories:
  - machinelearning
tags:
  - model
---

所谓集成学习(ensemble learning), 是指利用训练数据集构建多个分类器(弱分类器), 并分别对测试数据集进行预测, 
然后用某种策略将多个分类器预测的结果集成起来, 作为最终预测结果. 通俗比喻就是"三个臭皮匠赛过诸葛亮”, 
或一个公司董事会上的各董事投票决策, 它要求每个弱分类器具备一定的"准确性”, 分类器之间具备"差异性”

集成学习根据各个弱分类器之间有无依赖关系, 分为 Boosting 和 Bagging 两大流派: 

* Boosting 流派, 各分类器之间有依赖关系, 必须串行, 
  比如: Adaboost、GBDT(Gradient Boosting Decision Tree)、Xgboost、LightGBM、CatBoost等
* Bagging 流派, 各分类器之间没有依赖关系, 可各自并行, 比如: Bagging、随机森林(Random Forest)

