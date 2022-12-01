---
title: 模型保存和加载
author: 王哲峰
date: '2022-10-24'
slug: model-save-load
categories:
  - model
tags:
  - note
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

- [参考](#参考)
</p></details><p></p>

```python
# -*- coding: utf-8 -*-


# *********************************************
# * Author      : zhefeng wang
# * Email       : wangzhefengr@163.com
# * Date        : 2021.11.03
# * Version     : 1.0.0
# * Description : 模型部署方法
# * Link        : https://zhuanlan.zhihu.com/p/92691256
# **********************************************


# python libraries
import os
import sys
from typing import List
import logging
import pickle
from sklearn.externals import joblib
from sklearn2pmml import sklearn2pmml, PMMLPipeline
from sklearn_pandas import DataFrameMapper
from pypmml import Model


# global variable
GLOBAL_VARIABLE = None


def func():
    pass


# TODO
class ModelDeploy:
    """
    模型部署类
    """
    def __init__(self, save_file_path: str):
        self.save_file_path = save_file_path
    
    def ModelSave(self):
        """
        模型保存
        """
        pass

    def ModelLoad(self):
        """
        模型载入
        """
        pass


class ModelDeployPkl(ModelDeploy):
    """
    模型离线部署类
    """
    def __init__(self, save_file_path: str):
        self.save_file_path = save_file_path  # 模型保存的目标路径
    
    def ModelSave(self, model):
        """
        模型保存

        Args:
            model (instance): 模型实例

        Raises:
            Exception: [description]
        """
        if not self.save_file_path.endswith(".pkl"):
            raise Exception("参数 save_file_path 后缀必须为 'pkl', 请检查.")
        
        with open(self.save_file_path, "wb") as f:
            pickle.dump(model, f, protocol = 2)
        # TODO
        f.close()
        logging.info(f"模型文件已保存至{self.save_file_path}")

    def ModelLoad(self):
        """
        模型载入

        Raises:
            Exception: [description]
        """
        if not os.path.exists(self.save_file_path):
            raise Exception("参数 save_file_path 指向的文件路径不存在，请检查.")
        
        self.model = joblib.load(self.save_file_path)



class ModelDeployPmml(ModelDeploy):
    """
    模型在线部署类
    """
    def __init__(self, save_file_path: str):
        self.save_file_path = save_file_path  # 模型保存的目标路径
    
    def ModelSave(self, model, features_list: List):
        """
        模型保存

        Args:
            model (instance): 模型实例
            features_list (list): 模型特征名称列表

        Raises:
            Exception: [description]
        """
        if not self.save_file_path.endswith(".pmml"):
            raise Exception("参数 save_file_path 后缀必须为 'pmml', 请检查.")

        mapper = DataFrameMapper([([i], None) for i in features_list])
        pipeline = PMMLPipeline([
            ("mapper", mapper),
            ("classifier", model),
        ])
        sklearn2pmml(pipeline, pmml = self.save_file_path)
        logging.info(f"模型文件已保存至{self.save_file_path}")

    def ModelLoad(self):
        """
        模型载入

        Raises:
            Exception: [description]
        """
        if not os.path.exists(self.save_file_path):
            raise Exception("参数 save_file_path 指向的文件路径不存在，请检查.")

        self.model = Model.fromFile(self.save_file_path)




# 测试代码 main 函数
def main():
    save_file_path = None
    model_deploy_pkl = ModelDeployPkl(save_file_path)
    model_deploy_pmml = ModelDeployPmml(save_file_path)

if __name__ == "__main__":
    main()
```



# 参考

- https://scikit-learn.org/stable/model_persistence.html

