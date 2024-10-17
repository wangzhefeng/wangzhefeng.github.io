---
title: LightGBM baseline
author: wangzf
date: '2023-03-07'
slug: ml-gbm-lightgbm-baseline
categories:
    - machinelearning
tags:
    - model
---

```python
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")


# data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sample_submit = pd.read_csv("sample_submit.csv")

# training test data
all_cols = [f for f in train.columns if f not in ["customer_id", "loan_default"]]
x_train = train[all_cols]
x_test = test[all_cols]
y_train = train["loan_default"]

# cv
def cv_model(clf, train_x, train_y, test_x, clf_name = "lgb"):
    # random seed
    seed = 2023

    # result init
    train = np.zeros(train_x.shape[0])
    test = np.zeros(test_x.shape[0])

    # k-fold
    folds = 5
    kf = KFold(n_splits = folds, shuffle = True, random_state = seed)
    cv_scores = []
    for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
        print(f"**************** {str(i + 1)} *****************")
        # train and test fold data
        trn_x, trn_y = train_x.iloc[train_index], train_y[train_index]
        val_x, val_y = train_x.iloc[valid_index], train_y[valid_index]
        train_matrix = clf.Dataset(trn_x, label = trn_y)
        valid_matrix = clf.Dataset(val_x, label = val_y)
        # model params    
        params = {
            "boosting_type": "gbdt",
            "objective": "binary",
            "metric": "auc",
            "min_child_weight": 5,
            "num_leaves": 2 ** 7,
            "lambda_l2": 10,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 4,
            "learning_rate": 0.01,
            "seed": 2023,
            "nthread": 28,
            "n_jobs": -1,
            "silent": True,
            "verbose": -1,
        }
        # model
        model = clf.train(
            params,
            train_matrix,
            50000,
            valid_sets = [train_matrix, valid_matrix],
            verbose_eval = 500,
            early_stopping_rounds = 200,
        )
        # model predict
        val_pred = model.predict(val_x, num_iteration = model.best_iteration)
        test_pred = model.predict(test_x, num_iteration = model.best_iteration)

        train[valid_index] = val_pred
        test += test_pred / kf.n_splits
        cv_scores.append(roc_auc_score(val_y, val_pred))
        print(cv_scores)
    
    print(f"{clf_name}_scotrainre_list: {cv_scores}")
    print(f"{clf_name}_score_mean: {np.mean(cv_scores)}")
    print(f"{clf_name}_socre_std: {np.std(cv_scores)}")

    return train, test


# model training
lgb_train, lgb_test = cv_model(lgb, x_train, y_train, x_test)

# result
sample_submit["loan_default"] = lgb_test
sample_submit["loan_default"] = sample_submit["loan_default"].apply(
    lambda x: 1 if x > 0.25 else 0
).values
sample_submit.to_csv("baseline_result.csv", index = False)
```

# 参考

* [百行代码入手数据挖掘赛](https://mp.weixin.qq.com/s/k-0qYiuzzzUZ5nay4LHQBw?forceh5=1)

