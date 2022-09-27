---
title: 模型融合
author: 王哲峰
date: '2022-09-13'
slug: model-fusion
categories:
  - machinelearning
tags:
  - ml
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

- [Voting](#voting)
- [Averaging](#averaging)
- [Bagging](#bagging)
- [Stacking](#stacking)
- [工具](#工具)
  - [Python](#python)
  - [caretEnsemnble 包的 caretEnsemble.caretStack() 方法](#caretensemnble-包的-caretensemblecaretstack-方法)
  - [h2o 包的 h2o.stack() 方法](#h2o-包的-h2ostack-方法)
</p></details><p></p>


# Voting

模型融合其实也没有想象的那么高大上, 从最简单的 Voting 说起, 这也可以说是一种模型融合. 
    
- 假设对于一个二分类问题, 有 3 个基础模型, 那么就采取投票制的方法, 投票多者确定为最终的分类. 

# Averaging

对于回归问题, 一个简单直接的思路是取平均. 稍稍改进的方法是进行加权平均. 

- 权值可以用排序的方法确定, 举个例子, 比如 A、B、C 三种基本模型, 模型效果进行排名, 
  假设排名分别是 1, 2, 3, 那么给这三个模型赋予的权值分别是 3/6、2/6、1/6
  这两种方法看似简单, 其实后面的高级算法也可以说是基于此而产生的, Bagging 或者 
  Boosting都是一种把许多弱分类器这样融合成强分类器的思想. 

# Bagging

Bagging 就是采用有放回的方式进行抽样, 用抽样的样本建立子模型, 对子模型进行训练, 
这个过程重复, 最后进行融合. 大概分为这样两步: 

- 1.重复 k 次
    - 有放回地重复抽样建模
    - 训练子模型
- 2.模型融合
    - 分类问题: voting
    - 回归问题: average

Bagging 算法不用我们自己实现, 随机森林就是基于 Bagging 算法的一个典型例子, 采用的基分类器是决策树. 


Bagging算法可以并行处理, 而Boosting的思想是一种迭代的方法, 每一次训练的时候都更加关心分类错误的样例, 
给这些分类错误的样例增加更大的权重, 下一次迭代的目标就是能够更容易辨别出上一轮分类错误的样例. 
最终将这些弱分类器进行加权相加. 


同样地, 基于Boosting思想的有AdaBoost、GBDT等, 在R和python也都是集成好了直接调用. 
PS: 理解了这两点, 面试的时候关于Bagging、Boosting的区别就可以说上来一些, 问Randomfroest
和AdaBoost的区别也可以从这方面入手回答. 也算是留一个小问题, 
随机森林、Adaboost、GBDT、XGBoost的区别是什么？


# Stacking

# 工具

## Python

```python
def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    
    # NFOLDS 行, ntest 列的二维 array
    oof_test_skf = np.empty((NFOLDS, ntest))
    
    # 循环 NFOLDS 次
    for i, (train_index, test_index) in enumerate(kf):
        y_tr = y_train[train_index]
        x_tr = x_train[train_index]
        x_te = x_train[test_index]
        clf.fit(x_tr, y_tr)
        oof_train[test_index] = clf.predict(x_te)

        # 固定行填充, 循环一次, 填充一行
        oof_test_skf[i, :] = clf.predict(x_test)
    
    # axis = 0,按列求平均, 最后保留一行
    oof_test[:] = oof_test_skf.mean(axis=0)
    
    # 转置, 从一行变为一列
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
```

## caretEnsemnble 包的 caretEnsemble.caretStack() 方法

```r
algorithmList <- c('lda', 'rpart', 'glm', 'knn', 'svmRadial')
stackControl <- trainControl(method="repeatedcv", number=10, repeats=3, savePredictions=TRUE, classProbs=TRUE)
stack.glm <- caretStack(models, method="glm", metric="Accuracy", trControl=stackControl)
```

## h2o 包的 h2o.stack() 方法

```r
nfolds <- 5  
glm1 <- h2o.glm(x = x, y = y, family = family,
            training_frame = train,
            nfolds = nfolds,
            fold_assignment = "Modulo",
            keep_cross_validation_predictions = TRUE)
gbm1 <- h2o.gbm(x = x, y = y, distribution = "bernoulli",
            training_frame = train,
            seed = 1,
            nfolds = nfolds,
            fold_assignment = "Modulo",
            keep_cross_validation_predictions = TRUE)
rf1 <- h2o.randomForest(x = x, y = y, # distribution not used for RF
                    training_frame = train,
                    seed = 1,
                    nfolds = nfolds,
                    fold_assignment = "Modulo",
                    keep_cross_validation_predictions = TRUE)
dl1 <- h2o.deeplearning(x = x, y = y, distribution = "bernoulli",
                    training_frame = train,
                    nfolds = nfolds,
                    fold_assignment = "Modulo",
                    keep_cross_validation_predictions = TRUE)
models <- list(glm1, gbm1, rf1, dl1)
metalearner <- "h2o.glm.wrapper"
stack <- h2o.stack(models = models,
                response_frame = train[,y],
                metalearner = metalearner,
                seed = 1,
                keep_levelone_data = TRUE)
# Compute test set performance:
perf <- h2o.ensemble_performance(stack, newdata = test)
```

