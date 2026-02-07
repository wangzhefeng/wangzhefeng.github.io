---
title: 基于非线性树模型的时间序列预测框架：全生命周期深度研究报告
author: wangzf
date: '2026-02-07'
slug: tsf-report
categories:
  - timeseries
tags:
  - article
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
img {
    pointer-events: none;
}
</style>

<details><summary>目录</summary><p>

- [第一章 时间序列数据的特征本质与采样原理](#第一章-时间序列数据的特征本质与采样原理)
  - [1.1 核心统计属性：平稳性与自相关性](#11-核心统计属性平稳性与自相关性)
  - [1.2 数据清洗中的数学插值](#12-数据清洗中的数学插值)
- [第二章 预处理深层机理：诱导平稳性与外推补偿](#第二章-预处理深层机理诱导平稳性与外推补偿)
  - [2.1 树模型的外推困境（Extrapolation Problem）](#21-树模型的外推困境extrapolation-problem)
  - [2.2 目标转换的数学诱导](#22-目标转换的数学诱导)
- [第三章 特征工程：构建高维时序依赖特征](#第三章-特征工程构建高维时序依赖特征)
  - [3.1 滞后特征（Lag Features）与自回归模拟](#31-滞后特征lag-features与自回归模拟)
  - [3.2 周期性编码的三角映射](#32-周期性编码的三角映射)
  - [3.3 时序目标编码（Target Encoding）防泄露](#33-时序目标编码target-encoding防泄露)
- [第四章 算法选型：GBDT 族群的底层对比](#第四章-算法选型gbdt-族群的底层对比)
- [第五章 深度解析：多步预测策略（Multi-step Strategies）](#第五章-深度解析多步预测策略multi-step-strategies)
  - [5.1 递归预测 (Recursive / Iterated Strategy)](#51-递归预测-recursive--iterated-strategy)
  - [5.2 直接预测 (Direct Strategy)](#52-直接预测-direct-strategy)
  - [5.3 多输入多输出 (MIMO Strategy)](#53-多输入多输出-mimo-strategy)
  - [5.4 混合策略 (DirRec \& Rectify)](#54-混合策略-dirrec--rectify)
- [第六章 模型验证与诊断：超越简单 MSE](#第六章-模型验证与诊断超越简单-mse)
  - [6.1 时间序列交叉验证 (TimeSeriesSplit)](#61-时间序列交叉验证-timeseriessplit)
  - [6.2 残差分析（Residual Diagnostics）](#62-残差分析residual-diagnostics)
- [第七章 MLOps 实践：实验追踪与模型固化](#第七章-mlops-实践实验追踪与模型固化)
  - [7.1 MLflow 实验管理](#71-mlflow-实验管理)
  - [7.2 序列化选型](#72-序列化选型)
- [第八章 工业级部署：高并发推理架构](#第八章-工业级部署高并发推理架构)
  - [8.1 异步预测流水线 (FastAPI + Redis)](#81-异步预测流水线-fastapi--redis)
  - [8.2 监控与自愈](#82-监控与自愈)
- [结论](#结论)
- [引用的著作](#引用的著作)
</p></details><p></p>


# 第一章 时间序列数据的特征本质与采样原理

时间序列（Time Series）不同于独立同分布（I.I.D.）的表格数据，其本质是随时间演变的随机过程（Stochastic Process）的离散实现。

## 1.1 核心统计属性：平稳性与自相关性

* **平稳性（Stationarity）**：指序列的统计特性（均值、方差、自协方差）不随时间平移而改变。
  强平稳要求联合分布不变，而工业应用中通常关注 **弱平稳（Wide-Sense Stationarity）**。
  非平稳序列（如带有趋势或异方差）会导致树模型在分裂节点时捕捉到的是随时间漂移的临时相关性，而非稳定模式。  
* **自相关（Autocorrelation）**：描述了序列自身在不同时间间隔下的相关程度。
  通过 ACF（自相关函数）可以识别周期性，通过 **PACF（偏自相关函数）** 则可以剔除中间步长的干扰，
  识别出对当前时刻有直接影响的显著滞后阶数。

## 1.2 数据清洗中的数学插值

对于非均匀采样或存在缺失值的序列，简单的填充会引入噪声：

* **线性与样条插值**：样条插值能产生更平滑的导数，适合物理传感器数据。  
* **季节性调节插值**：先进行 STL 分解，在残差项上插值后再还原，能最大限度保留季节特征。

# 第二章 预处理深层机理：诱导平稳性与外推补偿

非线性树模型（GBDT）在数学上被定义为**局部逼近器**。其预测值由落入特定叶子节点的训练样本均值决定。

## 2.1 树模型的外推困境（Extrapolation Problem）

当未来的时间戳或特征值超出训练集范围（例如持续增长的销售额），树模型无法像线性回归那样沿斜率延伸，
而会陷入“水平平台”现象 3。

## 2.2 目标转换的数学诱导

为解决外推问题，必须通过转换使模型预测“变化”而非“绝对值”：

* **一阶差分（`$\Delta y_{t}=y_{t}-y_{t-1}$`）**：消除线性趋势，使均值趋于常数。  
* **窗口均值差分（Window-Difference）**：定义 `$z_{t}=y_{t}-\text{mean}(y_{t-1,\cdots,t-w})$`。相比单点差分，它对离群点更鲁棒，
  能生成更平滑的预测轨迹 3。  
* **对数/Box-Cox 变换**：针对指数增长和**异方差性（Heteroscedasticity）**，通过压缩长尾分布使残差符合正态假设。

# 第三章 特征工程：构建高维时序依赖特征

树模型不理解“顺序”，特征工程的任务是将时间拓扑结构映射到欧几里得空间。

## 3.1 滞后特征（Lag Features）与自回归模拟

滞后项本质上是在模拟 AR 模型。选择滞后阶数时，应参考 PACF 截尾阶数，避免引入冗余特征导致树的深度过大（维度灾难）。

## 3.2 周期性编码的三角映射

将日期特征（如 1-12 月）视为连续数值会导致模型认为 12 月与 1 月距离极远。

* **原理**：利用 `$\sin(2\pi \cdot \text{month}/12)$` 和 `$\cos(2\pi \cdot \text{month}/12)$` 将时间投影到单位圆。这确保了时间循环的连续性，显著增强模型对日内或季节性波动的捕获能力 4。

## 3.3 时序目标编码（Target Encoding）防泄露

在处理高基数类别（如数万个 SKU ID）时，目标编码极具威力，但极易引发 **回看偏差（Look-ahead Bias）** 6。

* **正确做法**：使用 **扩展窗口（Expanding Window）** 编码，
  即 `$t$` 时刻的特征只能使用 `$1, \cdots, t-1$` 时刻的均值，
  严禁使用当前时刻及未来的标签信息 6。

# 第四章 算法选型：GBDT 族群的底层对比

| 维度 | XGBoost | LightGBM | CatBoost |
| :---- | :---- | :---- | :---- |
| **生长原理** | Level-wise（按层）：保持树的平衡，防过拟合 9 | Leaf-wise（按叶子）：优先分裂增益最大的点，收敛快 | Symmetric Tree（对称）：每一层分裂条件相同，推理极快 9 |
| **缺失值** | 自动学习默认分支 9 | 视为零或独立分支 | 需预处理 |
| **类别特征** | 需 One-hot 或外部编码 11 | 统计分箱优化 | **Ordered TS**（排序目标统计）：原生防泄露编码 9 |

# 第五章 深度解析：多步预测策略（Multi-step Strategies）

这是时序任务中最关键的工程决策，涉及 **偏差（Bias）与方差（Variance）** 的深刻博弈。

## 5.1 递归预测 (Recursive / Iterated Strategy)

* **数学形式**：`$\hat{y}_{t+1}=f(y_{t}, y_{t-1}); \hat{y}_{t+2}=f(\hat{y}_{t+1}, y_{t}), \cdots$`
* **深层原理**：训练一个单一模型最小化一步预测误差。在推理阶段，将预测值作为“伪真实值”反馈给输入 12。  
* **优劣分析**：  
  * **优点**：参数量少，计算开销最低，能捕获细粒度的自相关性。  
  * **致命缺陷：误差累积（Error Propagation）**。第一步的微小偏差 `$\epsilon$` 会通过 Jacobian 放大因子在后续步骤中呈指数级扩散，导致长远期预测完全失真。

## 5.2 直接预测 (Direct Strategy)

* **数学形式**：为每个步长训练独立模型 `$f_{h}$`，使得 `$\hat{y}_{t+h}=f_{h}(y_{t}, y_{t-1},\cdots)$`。
* **深层原理**：每个模型针对特定的视界进行特征筛选。例如，`$f_{7}$` 可能完全忽略 Lag 1，而专注于 Lag 7（周偏好）。  
* **优劣分析**：  
  * **优点**：**无误差累积**；在模型设定不正确（Misspecified）时比递归更鲁棒。  
  * **缺点**：训练 `$H$` 个模型开销巨大；独立预测忽略了时间步之间的随机依赖关系，预测曲线可能出现不自然的跳变。

## 5.3 多输入多输出 (MIMO Strategy)

* **数学形式**：`$[\hat{y}_{t+1},\cdots,\hat{y}_{t+H}]=F(y_{t}, y_{t-1}\cdots)$`。  
* **深层原理**：训练一个单一模型输出一个向量。它学习的是未来序列片段的**联合分布映射**。  
* **优劣分析**：**利用了步长间的相关性**，推理效率高，且无累积误差。但在树模型中需要特殊的包装器（如 MultiOutputRegressor）。

## 5.4 混合策略 (DirRec & Rectify)

* **DirRec**：在 Direct 的基础上，模型 `$f_{h}$` 的输入包含前序模型 `$f_{1}, \cdots, f_{h-1}$` 的预测值，
  试图兼顾两者优点。  
* **Rectify/Stratify**：先用递归得到有偏预测，再训练 Direct 模型预测残差进行“纠偏”，是目前学术界前沿的策略。

# 第六章 模型验证与诊断：超越简单 MSE

## 6.1 时间序列交叉验证 (TimeSeriesSplit)

* **原理**：采用“前推验证”。验证集始终在训练集时间线之后，模拟“历史预测未来”的真实时序约束 14。

## 6.2 残差分析（Residual Diagnostics）

* **数学准则**：优秀的模型残差应近似为**白噪声（White Noise）**。  
* **Ljung-Box 检验**：统计学测试残差是否存在显著自相关。如果 `$p<0.05$`，
  说明特征工程中遗漏了重要的时序信息（如隐藏的季节性）。

# 第七章 MLOps 实践：实验追踪与模型固化

## 7.1 MLflow 实验管理

* **父子运行（Parent-Child Runs）**：父运行记录超参数搜索范围，
  子运行记录每个交叉验证 Fold 的 RMSE、MAE 及对应的 **SHAP 特征贡献图**。  
* **Artifacts 保存**：除了二进制模型，还应保存残差分布图（Q-Q Plot）以评估预测区间的可靠性。

## 7.2 序列化选型

* 推荐使用原生格式（如 .json 或 .ubj），因为 Pickle 在不同 Python 或库版本间存在严重的兼容性风险，
  不适合长期生产环境。

# 第八章 工业级部署：高并发推理架构

## 8.1 异步预测流水线 (FastAPI + Redis)

1. **模型热加载**：利用 FastAPI 的 lifespan 事件在进程启动时加载模型至内存。  
2. **特征检索延迟**：由于预测需要 Lag 特征，API 不能仅依赖客户端传参。
   生产中常将历史观测值存入 **Redis**，推理时亚毫秒级检索历史窗口。

## 8.2 监控与自愈

* **漂移检测**：使用 **K-S 检验** 监控输入特征分布（Data Drift）
  当 `$p$` 值显著下降时，自动触发重训练流水线（Continuous Training）。

**核心实现：生产级多策略预测框架**

```python
import numpy as np  
import pandas as pd  
import lightgbm as lgb  
import mlflow  
import joblib  
from sklearn.multioutput import MultiOutputRegressor  
from sklearn.model_selection import TimeSeriesSplit  
from sklearn.metrics import mean_squared_error  
from datetime import timedelta

class ProductionTSForecaster:  
    """  
    支持递归(Recursive)与多输出(MIMO)策略的工业级时序框架  
    """  
    def __init__(self, strategy='recursive', horizon=7):  
        self.strategy = strategy  
        self.horizon = horizon  
        self.model = None  
        self.feature_cols =

    def engineer_features(self, df, target_col, lags=[1, 2, 3]):  
        """  
        原理：构建滞后特征。注意在训练集中必须删除包含未来信息的行。  
        """  
        data = df.copy()  
        # 1. 滞后特征 (Memory)  
        for lag in lags:  
            data[f'lag_{lag}'] = data[target_col].shift(lag)  
          
        # 2. 滚动统计 (Trend/Volatility)  
        data['rolling_mean_7'] = data[target_col].shift(1).rolling(7).mean()  
        data['rolling_std_7'] = data[target_col].shift(1).rolling(7).std()  
          
        # 3. 周期性编码 (Cycles)  
        data['dow_sin'] = np.sin(2 * np.pi * data.index.dayofweek / 7)  
        data['dow_cos'] = np.cos(2 * np.pi * data.index.dayofweek / 7)

        if self.strategy == 'mimo':  
            # 为MIMO准备多目标Label  
            for h in range(1, self.horizon + 1):  
                data[f'target_h{h}'] = data[target_col].shift(-h)  
          
        self.feature_cols = [c for c in data.columns if 'lag' in c or 'rolling' in c or 'dow' in c]  
        return data.dropna()

    def train_pipeline(self, df, target_col):  
        """  
        集成 MLflow 的训练流水线  
        """  
        data = self.engineer_features(df, target_col)  
        X = data[self.feature_cols]  
          
        mlflow.set_experiment("TS_Production_Project")  
        with mlflow.start_run():  
            tscv = TimeSeriesSplit(n_splits=3)  
            fold_errors =

            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):  
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]  
                  
                if self.strategy == 'recursive':  
                    y_train, y_val = data[target_col].iloc[train_idx], data[target_col].iloc[val_idx]  
                    self.model = lgb.LGBMRegressor(n_estimators=200, importance_type='gain')  
                    self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(20)])  
                else:  
                    # MIMO 模式：训练多输出回归器  
                    target_list = [f'target_h{h}' for h in range(1, self.horizon+1)]  
                    y_train = data[target_list].iloc[train_idx]  
                    self.model = MultiOutputRegressor(lgb.LGBMRegressor(n_estimators=200))  
                    self.model.fit(X_train, y_train)

            # 保存模型与特征列表（生产部署必备）  
            artifacts = {'model': self.model, 'features': self.feature_cols}  
            joblib.dump(artifacts, "ts_model_pack.pkl")  
            mlflow.log_artifact("ts_model_pack.pkl")

    def predict(self, history_df):  
        """  
        推理逻辑：区分递归与直接输出  
        """  
        if self.strategy == 'mimo':  
            # 原理：一步输出整个向量，无累积误差  
            X_latest = self.engineer_features(history_df, 'sales').tail(1)[self.feature_cols]  
            return self.model.predict(X_latest).flatten()  
          
        else:  
            # 递归预测：原理是将t+1的预测值作为t+2的特征输入  
            current_data = history_df.copy()  
            preds =  
            for _ in range(self.horizon):  
                # 重新构建最后一行的特征  
                feat_df = self.engineer_features(current_data, 'sales')  
                X_input = feat_df.tail(1)[self.feature_cols]  
                p = self.model.predict(X_input)  
                preds.append(p)  
                  
                # 更新历史序列以进行下一步预测  
                new_date = current_data.index.max() + timedelta(days=1)  
                current_data.loc[new_date, 'sales'] = p  
            return np.array(preds)

# ==========================================  
# 模拟运行与测试  
# ==========================================  
if __name__ == "__main__":  
    # 生成带趋势和周季节性的数据  
    dates = pd.date_range('2024-01-01', periods=200)  
    y = np.linspace(0, 10, 200) + 5 * np.sin(np.arange(200) * (2*np.pi/7)) + np.random.normal(0, 1, 200)  
    df = pd.DataFrame({'sales': y}, index=dates)

    # 1. 运行递归策略  
    forecaster = ProductionTSForecaster(strategy='recursive', horizon=7)  
    forecaster.train_pipeline(df, 'sales')  
    res = forecaster.predict(df.tail(30))  
    print(f"未来7天递归预测结果: {res.round(2)}")
```

# 结论

本报告确立了非线性树模型在时序预测中的**全栈方法论**。核心见解在于：**特征工程解决了树模型对时间的盲视，而多步策略的选择决定了模型在长视界下的精度上限**。对于高噪声、长周期任务，推荐使用 **MIMO** 策略以平衡计算成本与预测稳定性；对于实时性要求极高的小规模任务，**递归策略**结合 Redis 特征缓存是性价比最高的部署方案。未来的演进方向应关注**概率预测（Probabilistic Forecasting）**，通过输出分位数区间来量化误差传播的不确定性。

# 引用的著作

1. Time-Series Forecasting: Comparing Transform Techniques for Tree ..., 访问时间为 二月 3, 2026， [https://www.snowflake.com/en/engineering-blog/time-series-forecasting-comparing-transform-techniques-tree-based-models/](https://www.snowflake.com/en/engineering-blog/time-series-forecasting-comparing-transform-techniques-tree-based-models/)  
2. Feature Engineering for Time-Series Data: A Deep Yet Intuitive Guide - Synogize, 访问时间为 二月 3, 2026， [https://www.synogize.io/feature-engineering-for-time-series-data-a-deep-yet-intuitive-guide](https://www.synogize.io/feature-engineering-for-time-series-data-a-deep-yet-intuitive-guide)  
3. Feature engineering for time-series data - Statsig, 访问时间为 二月 3, 2026， [https://www.statsig.com/perspectives/feature-engineering-timeseries](https://www.statsig.com/perspectives/feature-engineering-timeseries)  
4. How to Do Target Encoding Without Data Leakage (The Right Way) | by Prathik C | Medium, 访问时间为 二月 3, 2026， [https://medium.com/@prathik.codes/how-to-do-target-encoding-without-data-leakage-the-right-way-280bd24fbc81](https://medium.com/@prathik.codes/how-to-do-target-encoding-without-data-leakage-the-right-way-280bd24fbc81)  
5. What is Data Leakage in Machine Learning? - IBM, 访问时间为 二月 3, 2026， [https://www.ibm.com/think/topics/data-leakage-machine-learning](https://www.ibm.com/think/topics/data-leakage-machine-learning)  
6. Avoiding Data Leakage in Timeseries 101 - Towards Data Science, 访问时间为 二月 3, 2026， [https://towardsdatascience.com/avoiding-data-leakage-in-timeseries-101-25ea13fcb15f/](https://towardsdatascience.com/avoiding-data-leakage-in-timeseries-101-25ea13fcb15f/)  
7. When to Choose CatBoost Over XGBoost or LightGBM - Neptune.ai, 访问时间为 二月 3, 2026， [https://neptune.ai/blog/when-to-choose-catboost-over-xgboost-or-lightgbm](https://neptune.ai/blog/when-to-choose-catboost-over-xgboost-or-lightgbm)  
8. XGBoost vs. LightGBM vs. CatBoost - ApX Machine Learning, 访问时间为 二月 3, 2026， [https://apxml.com/posts/xgboost-vs-lightgbm-vs-catboost](https://apxml.com/posts/xgboost-vs-lightgbm-vs-catboost)  
9. XGBoost vs. CatBoost vs. LightGBM: A Guide to Boosting Algorithms | by Kishan A - Medium, 访问时间为 二月 3, 2026， [https://kishanakbari.medium.com/xgboost-vs-catboost-vs-lightgbm-a-guide-to-boosting-algorithms-47d40d944dab](https://kishanakbari.medium.com/xgboost-vs-catboost-vs-lightgbm-a-guide-to-boosting-algorithms-47d40d944dab)  
10. Learn Recursive Forecasting | Multi-Step Forecasting Strategies - Codefinity, 访问时间为 二月 3, 2026， [https://codefinity.com/courses/v2/df30ac7b-a08c-4606-b8ce-fb927b2f2df3/a63f4780-5986-4875-ac0e-23472c951c0f/1de6f1f2-e07e-493d-bd5f-39158aa7ffd8](https://codefinity.com/courses/v2/df30ac7b-a08c-4606-b8ce-fb927b2f2df3/a63f4780-5986-4875-ac0e-23472c951c0f/1de6f1f2-e07e-493d-bd5f-39158aa7ffd8)  
11. Recursive MultiStep Time Series Forecasting - Kaggle, 访问时间为 二月 3, 2026， [https://www.kaggle.com/code/ahmedabdulhamid/recursive-multistep-time-series-forecasting](https://www.kaggle.com/code/ahmedabdulhamid/recursive-multistep-time-series-forecasting)  
12. How to Perform Cross-Validation in Time Series - Statology, 访问时间为 二月 3, 2026， [https://www.statology.org/how-to-perform-cross-validation-in-time-series/](https://www.statology.org/how-to-perform-cross-validation-in-time-series/)  
13. Evaluating Time Series Forecasts: A Clear Guide to Metrics and ..., 访问时间为 二月 3, 2026， [https://medium.com/@sumeyyesahinsavaskan/evaluating-time-series-forecasts-a-clear-guide-to-metrics-and-cross-validation-468949d4c995](https://medium.com/@sumeyyesahinsavaskan/evaluating-time-series-forecasts-a-clear-guide-to-metrics-and-cross-validation-468949d4c995)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAIEAAAAYCAYAAADdyZ7bAAACz0lEQVR4Xu2ZS6hOURTHl0ceocizRBiJRFLKQHck5ZVHpgZEMWDARJlcKaEMkCKvTLiFkAh5jMTAq0wlYSRkIspj/e/e27fuOk+3s/c5OftX/85Z/7XPd/73nv3tc77vI4pEIpFIpJDvrOHajLSH3azfrPe6EWkPmABOQ1Qv0gJ2sPay9pCZBK/7tiNtABde7kODhBf5z9nIOijq/WQmwSvhacaxLrJm6gYzTRs1gpXtkDaZ2dqogUZlk6uAw60GaUxi3WYNpOSYdyleXXxgzWD9ZD1UPWScoryQNCrbetYxbTJHyIR5rBtkgoMNlLzgqHuUV4aTrPMZOsc6yzrNOmXHLus9KpvprK12H5meiN5269VF47LlnTBrNcDFBz9Yv4TvVoa5whsq9kPSZbdjyGSa3GnRZ+tJRqg6jTmsBSU10h6TRpfdVpktDRz3RZuapWTeZVmcIRPorm5Y0Nsm6p3Wk+g6NPcomQH1VVEvZz0XdRZLWCtLarw9Jo8qs2mwui+k5OsnKBxAZkzauNWU9L8qDzNdrhR5dLMO/INWmMMKQZ6XosYnHnjzhfeMzN8TGt/ZBlDyGvVhEeuSNlO4TOaF9Njj1pegvib2peoC594i6l3WA7PsvtMFNygQvrMVTgJ9kcpIsk55blLMEx4eILEa1AkyYSID98yi/xZdh8J3ttxJMJWSF7iMDuNgwS3Ru2K3El3XwUTqZHxhtz2ij/t3f+65VdCfbN8KJMmdBFWAJ2AJTvZW1KOp/POAL/BPGCZq3AKRU/5K+pS1RtShCJHN6yTAt4t48Qm2XmVrnNSBBz3cIsBN4YfEvctk/UjUzgO4B4+SDc+EyOZ1EpxgXbf7i8mcaGyn3ctgMp8WHig/JMjlfg19w7rTaf1lH+s+a61ueMZ3NnyH84n1kcx3D/hxsHI2sW6wNutGg8C75yiZSYvbU5NocrZIJBKJtIM/nX/+55VJrYgAAAAASUVORK5CYII=>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAANYAAAAYCAYAAACGAQPqAAAFiUlEQVR4Xu2aV4gsRRSGj1mvOSEoPigmEAUDYkC8BvTBiCKKmAPifRDjg+HhKibMigFMqyiKAQRRTCiKCVEUwZzumnPO2fPdqto5c7p6tifQw7L1wc90/VXd011T8fSIFAqFQqFQKMwU9vFGYVo28kahYHlLtZo3C9PyomoTbxYKcL7qTG8WGvOfallvFmY3i0poGIXB2VX1izcLs5tHVFd4s9A3DE5rebMwe6FBLO7NQt+w13ram55tVLdLWCYAn/M62WNlU9UdqpWdT+NYyXltcrjqPNU9Mb2x6nrVnqmAsqHqOtW5xvOcovpUdbXPUFZRXaX6UPWQatXu7IURquMllNk+evtL+C2pN8+60nsZeLHk915beaMgB0rvupQjJBRI2l31R1eJ8bGT6hLVDlJ9CNLvOa9NzpFOnU2qllctEtMfqZ6UcN/Asb//paJ3SExvG9NLTJUIaToJrBjT86dyw/Xvi/6xqqeiv3706GSWC6Of42/pfMdZxl8veoUqPevlXZf+S6qzQ1Nu7aFbVDerblLdKGF0p3H1It34hDlOkB73rPqxhPtY2ngnRu9I4wGenUV+leoGmMb9lUlzzqkmfXf0PHjeJ+2vn+vgcKiEGRfIpwMm7o1eoUrjemHdmNuQLeeNlpgbP3kA3rskNo8eM0Riuk4KS6q2bKgt4jm9mJRq5R6X8QBvrksz2Kxh9HD062CplsvHm8h4vuwHUZ7D4ierFX8O6W+cNyhrqv7x5ggYV/ukbuygmoV1/ArelLA8+NebLTJHwgOkERXuj57Fp3Pw7mGvhrJ7pTqY7f33Hp3xAI+lLaTl1W2qXTJKsEejIf6mOk3CTF937Wsyni+7QMJgUMfPUj2HNLNwgsF3kH9sMAseI9XrD8s42yfPwhaglu9Vi3kzcpk0D89Sef0o15E9fLf/MUh/Z9IsXcdRue9I9d6OynhgO1ZK07Hq2FtCGT4TBBbqru1/o1zHIgDSq54of6VJE7TAS0Et8Nfshx1luPNz9NM+R03PZ2Gt70knpB8Hfd7JbpU3pPoApE8yx1ZtMinV7+w1Y+3s0rll0WT8zD3P5cazeRwTGbTkzs8NUhbymCUTD0QP7orHSYP8Z27UHcveT7/t83XVjyZ9kWptCZHYpvdYW47NNxenwGeqV+NxilRB7cktQYOx9/B8TNtRlAY6aMBlGAg0+Po5PXo2usceAO8A47GnwrvAeLupzojHL0v12gSWkkegI4F3p0knz5+/QcazkEf4HniVQfrbTvbCJTL3NSij7lgwyPXS70C7SXske50m19xPasrReQi1A42UL6HgvlMlQuX2Wjq0xWvSaSi5BufTbcCGnoGJ0PoXEvZGP6k+iR4DFf9yeFPCeyo88uwKgWDK+9J5NruXgbSXRI9Gj4Hlawnv8XgVwUjNtbkXlvSrS+jwybNRRuBadlCypJA/SmH8eSb/Janur9j/1cnPIqPuWLn2uUCq92Flsfdij383x3U8q3rOm025VLrX3OOACKCFCiDsnMhVbqGeLyX/wnqOdEdZCZb4TuDT/TLqjjVM++QVwzPxeA/VKybvBHNcB89hVyV9wSzGkoVK39rltcEL0v1DsFH1P8zZqmvj8YM2o5BlGanWIeCxMrBpu0xNHkx0uc3JdSyir+x3PLx6ONmbyg/SeSU0TPtk4LghHtM5maGBpfx0bKf605v9wEMTXp3v/LZ4W0IFQIq2+eglSyL2iU84v1DP4xLet1lSg6d+WTKxmfewvKQ9NInmergm+zWWsSxZ02sFlp+5AZG99UHelBAHSCHuYdsn98KsRafknrjWZl0l8tChaXczGqZlRhMbdi4MD/tC+2J1HQn/jCFyOPASZxbwmDTrfIVZzMHeKEwLM2yhUCgUCjOc/wHdUJ4t6IK7nAAAAABJRU5ErkJggg==>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAJ4AAAAYCAYAAAALbES+AAAFjklEQVR4Xu2aV4glRRSGjzlhwoQYBjMm1AfjgxgezIqKKKi4oqAiiAEjyq5ZFMODCVS8uoY14YMihgfXFwMKKqIoirvqmjNmxXA+q2r63HOrenru3HtnYPqDw3T9Vbe6wunqU9Uj0tLS0tIyPNZXW9qLLdPGMl4YADt6YbpZXW2JF1v6Znm17bw4CR6T4S0C/3qhH6jkLC/2Qakx90vI+03tCpfXkudXCWNWGtMm1P32NS8YNlZbLOH3b6mt1pUb2ETtey9OhsOlcoqp8IHa2V6UUPfa8XqHmP67yp71LKV2mxcjD0q989SxldrFTntFKmcu1Xuk2ksmvUhC2cOMlvhF7RAvToZNvdAHuY6cr/aGdC/350ooe4PRZjNMaMnx7pP8uDZhsRcMLBCletGPymi58ltIXh8Zt6j97EXlZQkNu9nppY7MRn6SsuN1pP9x+sMLhokcz+d9HbWtnQ7oK3sxsaraQ2ofqp0n4UlKnKh2tdqjRjtW7VIJSz1spHa92iXjJbrh5rnYjQb5AHcFyXduRbV91PZTO0DtwGhco/FKmiqlfh0/XiKEBazG6Msa3cLrhbGkb9u7vKMlvOLmx/QaanPVrhkvUfGAhHF4Sqr+Wu6SapzG1O5QmzeeW4Y2EEKVqHO809SOcRpOTPlcrId+rRdhQ+mO39aU7pveFNP/GA0HS87xnFSv4q+i5kFj2W3ChRLKn260Z6JWZ7Z8v9T16y8JD+CpUcPxKOcd/k/pfgA/ku6Y6AK13yX8FmfbN+rcz47d3lI5wIvx2sfIOFpq62ZRK82Bhb7UUed4OdKY5XhWCqvrIxICf8uPLk2l1vEgxRfbGG2nqPnJQCutDh7KfmfSZ0poY+LL+Pc4CZMzaOr6da/RAO0Mk14kveMElGNVTpwcNVYPC9ouGa30qk2Ol2urnwOLn2/PZByPMIqybFZy3C6FuhgQMvDKyyWseB7y/YB2om7hHY/Ga9Hiy5V4X+0bp9mJYJLTZL0qzc+gaHvTNnSkt2zq17pOR5vn0jeadALd1jnHpRNo+2e0iRzPUpqDxFVqm3vR0dTx1pJQzj8sFjaQxbquk2pwsHe6s//XvOPdGXULHUJbyem+XA7O8t72osPW06TOROpXE+r65YNktMviNWdbpK+sssfx92e19vcANGJWr7Fq5Mg5XmkOEr58jiaOx4pKGeLgOtIpRQ98TUiwgjwsoeBeRiftHS/XaeIMtNwE1X2aYSUjFrD4uhlIq3HNRmTQTLZfyfFS+h6TTqDbOtnE+HsAmt9AoLGJAOJkuyLeLb31lNoKjOHjXszQxPHIX8WkT5IqzrSwWmfrel5tN6cRIF9k0vzQO14n6pb0tNkGAVquUbCnVDs8i1/9SL9p0tTJBA6ajpT75ScTjfDEpnOn9ej2wZoTNQ/aQRktxZYcsB9q8nLneKU5gCel2cM6keOxOfIxu5+vxNMSvrL0sFB6v0pw03Vc2jveE1G37BG1MaejzXUabCAhL2fsIC1oOKlN+03QIKjrF5+BLGg2/koHptsa7daoWc6JGt9aEzg12glGA7Rv4/UC6Z7wuraOOR182RIcf1DW+kDiY6nmyFsOdE5GeliotrNUZzFY2uLTSTr9abQfos7fJWqfqH0h4annKfgsavy1Xt6R6rcWNhO+8cl2N+V2jZrldemNRadK037xOiYfjfL2AWCleVeqfhAzWjhIZyz57edqL0hYyblGI49PTYl0vIXZs76mbU1sKROf8fGmo65UL385HJ5vyvh5spYDPbdhHQlMRqlhLaOBA+1RwxnxtM87T+ApXmwZGdlD3CFDvHuEF0fNcjIDvH+WwoYl998jw2Q9Ca/tGQHB+XtebBk6uX/QGDYzbpHh00rd55yWwXOwF4aMPX1oaWlpmeX8B7gizFNjLjPgAAAAAElFTkSuQmCC>

[image4]: <data:image/png;base64,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>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAcAAAAYCAYAAAA20uedAAAAcklEQVR4XmNgGOTgGxCfQheEgf9AXIAuCAL6DBBJJmRBGyD2AuLdUElfKB8MioC4BCrxFsoHYRQAksxFFwQBXQaIJCO6BAisYYBIYgUgiXfogjAAkgQ5CgaOILHBkipQ9k9kCRDoYYAo+AHELGhywwEAAMS4F/hUVNxNAAAAAElFTkSuQmCC>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAE8AAAAYCAYAAAC7v6DJAAABV0lEQVR4Xu2XvUpDQRCFR9E0tnax0zYE1HR5goDia6goKpbiEwhWeQYLfYO0VuYF0quIiBYKojZxhtmL6+Em2W12c8l8cMi9ZzabwwncHyLDMIzZ5pS1i+aMs8D6QbPginQ4dNr7P87OJ+sOzQQM6K8T0USmsTzJdIRmQt6pouU1STPN4yAhlSuvzeqweqSZttx5DipX3gnpzUvyvLpzUQ6iyttHMyOS5wDNxESVlztsQYM0zxwOSqixNgO14b4TSlR5h2hm4oYCQzNLrO1AyfUzhqjycj4W+EiWNzQzEFXeMZoltFjraALnaAA7rDqaHpJFbhoFt95xSoLKWyZddIGDEmTduA37pPNLHHhM2kNma+742x8kRt5wRua8Zr2wHlj37vOZxrzPMV2nUayQ7rOIAw/53TM0PeRPlNBfpO+XqflgPZF2InokvYys+osMwzAMwzCmkF+sTV/ecgdQRwAAAABJRU5ErkJggg==>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAI4AAAAYCAYAAAAswsVWAAAEcElEQVR4Xu2aa4hVVRiGv7KbRTdMC6KYMlAwyrKaiojoR3TX1F9CjmWkGN6ioKKIbhDd6EpQVBNR4vwwStToBiqhzI/KwET84VCRSPfoQve+96y1Zn/7Peuss/cMM2fmnP3Ay+zv/fZl3fbea+0zIhUVFRUVY5N5bLQxc9moiLNKNYdNwxrVRWy2MbhJXmYzxYGq/1S7OdHG/KU6S/Wb6nXKgStV77LZAfSrFrAZY4K4QXO0arnqy3y6LblUXJ2n+r/X5tM14Hcqheq+iOJu1WHktRtfS7px7lJ9ymYHMaB6hs0KN2i2s2lA/nw2O4irJX1jyWOqu9lUzmOjTbhCskZ5Sdw8ZkZuD0eq0e4R125M7DxlOU61VjWdE0oXGyNMwzb4W9y8BjvcZ/zTvNeOYBV1v7j63a66VXVubg83mBrVH6+4U1X/qDZTDsecRF4ZThA3GQ8LFctXEW+kwfUuYHOhZHcIdnjE5N70Xqt4LaFXVb3ilox4YryoOrR2VHEwYFL1w00Uy5+iWua3kcfqI7DSe8MBgxH0SP25EPeRN9LgmjewicKBqyReyO/IO4jiohyh+pHNFvO51NfZgsEZy1/i/x4rLn9ilpIfvGcp22ahT/5U/Wv88AQ603jDAYMfT9Vm4JoPsxn4ReorjHi1ifE+f9LElti7OPCcuBUan7/VoDz72TS8Iukyfyj1ecRvmTjVZs3AuW4x8W3es3BcBHzIXCLu2KID53E2A0g+bWJMiuFhlAcw+ieZ2NJs5XGAlK8kXptldJQ7rDAoz0NsGu6QdJmR+8zE4TvY2cZLtVmK66T+2j9HPI7LUGbgrGAzgOQ0E2/wHsAjDdtBS8NOhgvZIIYycEaS0Mld5FtinWdBDnduwM6ZmrXZYtVk8izPS/21Eb9ttq2GAo4rOnAa9i+SYVQd4+Pvs3TNs+9bpuGJPWNt4FwvxcqT2ge5dX47zD/s/o3aDHMU3peZL/l8GEgzjfexaraJy4LzYW7bjFQ5ax0fKoNRjb+4awJPSP5djaXo70Z/UAxZxtrA2SLFyoN9ZrHpOV6yNtvh//aZPLeZpV/1DZvEO5KdP7bC5RjzTO4Dq+5s1xo4/hrymMuk/jqDHC6uYwN3Sv3OWCKm3tXj7YmDsmxiM8JG1UdsiquP/Tkm3HgTjdeszfD9rBHnUIxzfxHxhgOOj/02Z9kl7nfLKDjBTop5+RUK2ZNzM8bDwLlRXBnQ4dzJKWLlhmd9bG8zcfBAozbbw4bnUXHHTvExXkeI7c2NH2fRqWCv8cuAc6b+lQTE6j5ISGLCiEcaCs58IO4L6emc8KQGDr5HYL70rbjvHA/k06MGXieoH+Y32C7Kg+I+/1vQZof47QHVe1lqkFSb7ZP8QLC8oFrvty8Wd63Ykwvt+gmbBUBf/STuGx30az49yPuqy9m04Etor+op1cH5VGHw/yzjgXsl/22qKFtVZ5j4SNWz4joZk+CynMwGcZO4le3NnBgl8ER7g82KocFL6nbGLowqKioqWsj/nKsuCYa8vrAAAAAASUVORK5CYII=>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAKgAAAAYCAYAAABugbbBAAAFCElEQVR4Xu2aacgWVRTHj2WLRRt+UIzEFihs0YwWWqAFglayviRRan1IDW2BsKQFpSAqoo0+FJUVUfkhKSipLJeQJFIrMBNCjMgIWggrtc3O/7339Jz5z8x95vWFmfd5mh8cnrn/M8885965c+fcex+RlpaWlpaWOriKhT7mQhZamuUWtStYdLyidjaLfcy+ar+wOBzYS2232mZ29DF/qp2s9rvay+QDF6u9yyLxt9qvLPY4U9XWsdgke0vonIeozVX7JuvuS86XUOej4+flWfcA0FPAf7zaOWq7yNfroG5jWGyKGVQ+XW1/0vqNbZLugAvUPmXRgZH3QFfGzTzKlXudG9R2sNhSH+ica1l0wH8Gi/8zUg9wbTysdheLymks9AkXqV0qofGflZBn4jXNpG7O3RLajSm6TlPMUnuKRWUcCwnQBmifxvhLQt6JQBY6/Zio9SOYtS+SUL/b1W5TOzVzRrgpZfVHaoBXOSZHq8iH7xxBWhO8JyEvXiH53BgxXktaGX+ovcNiXVwnnSceQT/ofEuj1hQvJewFtcVqz0kYAZ9R22/gW9VBx0zVDw9rkf9ItTnxGP6Pne/mqDXNCLUl8RjxYBAyJkWtKl+qfc1iXUyPn5dIPmiUfyRtJJWrcLjaTgnXQ9I9XPhC8nX24CEo8p8bPw+T4Ef9jJ+j5vGTqMGAJb9NLFZkgnTuFeKZ1nHJm1HzpGJcJvnzawdreBwEyre6MvKtR13ZcxwLjifcMa4525WbBLF8z6Ljecm3iecDyftRfsOVkeemVgHOYiGCN8MFkr/+YLlH8tdAGQ+S0S3GtyR/jdpBAI+7MiZH0PAUG/+ojXZlT9lMd7xkK4dcpmplkW4Mxg4OX6sM4rifRccdko4Vvs9d2daRpzhtg4QF7zIuY4FI/X4VcM9gHlwTObfRLUaM4j+xWDcI+lhX9k8N8i0cm2FmyJzJQgnb1T5hsQGsM00g3YObluog8N3oyj6nnRiPzV61k4ihdNBREnLeFPg+tmkNTAShYeCpGiPSs247aVgvTjFDwkS8DGytzmPRgwDthENj2T810PhJ9FTpoJjEpBq8TjCDrRJL6hz4Xo/HtkXM53OZGUoHtd9LLQViG9ffR9xDviaXGfhT/1OwvBsDWRG2QpT6HfPzSsp/oIPZSZZE+x98RLL5J5ZYsMNghmUMXy7afcA1MbscDqyWdIMZOOcUFiPYNbI2+yx+LnF+bJtybsdtxO02uXPqAKkYr1T7VsJkrgy8KfAbuA7ybfs0imJkUjGAE9W2sEhgBE5NkLEujVSjkAMk23HulHxQWO8ryz9BtxHUJ+VfueOmQP2WsVjA22prWJTQXn4b2B5wvHaN9RI6UYqhjKDgILX5LDr8PUOOjuud5LRuMV6t9h2LdYOgN1L5AVc2DUzPqB1SHXST2sxoN0lYQG6C6yXUAx2LO1OKok4Czes4/siVTQPI9dCRihhqB02Nnhzjb5L/C123GOEfy2LdWJB4HeA185DzGe9L2DE5gR2Rsg6KGa01lBn+KdUEeA2jfsg/cVyV+9ReIw31QGIPtkrxQ4cVghWS/qNzWQdFnHjr/CAhh0TbMxgdl7PoQIy2Rov4il7DqRjPU1vJYhNgZ2Sx2mNq+2RdlcG/enqBeyW7tluVDyX7asRo86Ta0xImkHsK/u63p1TZOcOb8EUJ93gwYNKHEbelhyhaXutXrmGhpaWlpaWX+BeNVUj5NjzxfwAAAABJRU5ErkJggg==>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAYCAYAAADH2bwQAAAAZklEQVR4XmNgGAW4ABsQewCxO7qEKhD/B+LjQFwCxLHIkhFQSXFkQWQAkrwJxCxIGA5MoQpARu9AwnBQDFXAhCyIDAwYIArY0SWQwR8gPorEB5kmg8QHA5C9IJNA+CAQM6JKD2sAAPoREsiqMI75AAAAAElFTkSuQmCC>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABIAAAAYCAYAAAD3Va0xAAAA8klEQVR4Xu2Tvw7BUBSHz4QXMJhNImwSu0WEWASDdyCewyoGm5mYbQaDhURsPIPFYPPnd3pbuT11RHXkS760/Z1zT5vbluhnScEhjMlCGIrwBjvwLmqh4MUD9xh5UEaGYWlRxKdIwwpckxlUg2Vfx4fUYZ/MEN5oPu/5OkLCg6Yy/AYeVBVZHh7hXuQqBdI3egcbMtQYkT5Iy19yIX0B50s4gQd/KQg3L2RI5o1erWvtZk+4oSRDsCXzoXq8HZQlvcHOm2QGB+CmMxzDk6h52IN4H3NwZWUO3NR2jwlRY3jR3Lruwg1MWpkD/+UzGJeFP/QA3yk1IyXpxioAAAAASUVORK5CYII=>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAALgAAAAYCAYAAABJLzcpAAAFT0lEQVR4Xu2aWcwlQxTHj30JxhIzCMlnFzuxxYMHHsS+kxBmeBJiGUFsydgJYn1AhIz1xQMhiAcJQiKx72tMZBLbjC323fnfquOe/nd33e77ffq77le/5KS7/qer7umq6uqq6iuSyWQymUxGZAsWMv8py6utzWKmnjPVDmWxIX+zkOmEb9VWZzHFshIa6312jDm/q+2k9pPa/eQbxBdq67KY6YzGg8tyEi6epXaa2uKie2zZW8J9bxqPBxfdSU5Q+5LFTKdcqfYSi1XMo/TuaiuTNo58Ki1GAQL51mMx0zloh2VYzARQOS+w2IANZPgHIzO1/Kl2OYue69QuYlHZjYUxYj+1AyV00jvV9lfbpnBFmnvUfmMxghX+fWr7skPZkYUhQNwLJfyOBwuuFUmbCdym9geLBhyYd6OhL3H6ZlEbV7BrcqmEezxH7Sy1XQtXpMGoUbUgxVpmUTxH2Ss5Hzr9ZOt0gdrceOSy7GGdaewp5brogUWSjVq44Brneyhq08m9Cbtbwih2l4RGvUPCQ9kGdOxh7xH5LmNR+SgeJyRcs0bf1Utj12UyfBaPH0sxdtsB29ppMwWsgyrbESMBOEDKFyD9FWn8SmzC9mofqr3JjhHgHSnfN7hVqnUP/CeyqFwQjy9KuQykzyCtDVhIbRzPUdbtzoc3EP/eapRuw2MsjDh87wV+kPIFSM93aczTb3Rpz1YsEK+rHcniCIB7rBtRuT4Y+Oex6ID/QZfeJWoYaY1Bv1HHXhLyYjpkfBc1A9POv1y6KWjnQ2T42KaLZLxw3uzSWFxyY6Cy1nFpzx4sEMkfT4ApUxtrs0gEiOsKFiV80XyDRQJ5/ZrFg3qC39fX41HzcLopGDA4L9KPuPQNaje5dFu4/FFmjgyIF84tXRqvJ8twSjw3O9kucmCSnwL5npaw8/BB0TVt2IetCdLBa2rnSphm4HPwtkV3D+TFWqCKY6Rc4Uh/4869eTANqdrR8iCPzcWB3cvOMe3L/twuagnH5WkS41ESOl4dGDyxBkqBvtfkWwwG2FS8Pefp8XzNmP667+5pqdddqoPjdYcdByMZSIccL/WxQEfcYB+1l53PeEDtFxYj60ux7ONiGjs3xivS/w0P1j24Fg9YHW9JsXzL4+F0W1L5B8WIBwD+VBnYvfP1zGwug8swbpEB19k2CwyvORzx9BjXS3H+vYnaz85+pTTMQENiRDOSgXTIs1Ifi9exm+TXIsaE1OcHmPJZndqi1X9tq8u7g4QO9Ak7iCUSysBDhv/Q+PKqBiS8bbiNqtrLqIsPNInxYbWzWXSgz73HIvG82uEsVoD/El3ForGqFCv+fCnfHEbguvk3SI3gvqyjJXT4UQBxPcGihI8o2F0xuC488K3FopS36tAZ/VsMpMoFVbEZvB2Ksk51aQxIfk01DIPiA6kYuyQZK5xvU/pqlzYNzC2ofZp2cIw026k957QuOUlCPJjX4bhK0d0Dn+0xfwQrSD/+w+LRgy2/xaTZliseZmDTldn/XhH+4PVuPF/kdAPbjxuyGMHb0tcppk/cwHiY5kgYvAZtANTBZTKpGLvkQhmwIWA3goUKXlXXOp/xlNozUr3YAnUdHJ0Zr3gDHQINMl1/L8XuA+4R82+cV8EN+72k/62GuaT/T/KxEj7CAEzlUF7V53l85n+VxUhqYYjy8IAA7JT86HzGQRJivpj0JjwpYTG8VMI6DK//KlIxdgm3Vwl8OFgoYUsJI9Yw4L/U/xcWSPWcejJwJR+h9qjaeaRPFdjBwB77sKPzuICdpFksZqYerGGw6s90B6aXG7GYyWQymXHgH9ckV6ylZDl4AAAAAElFTkSuQmCC>

[image12]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABEAAAAYCAYAAAAcYhYyAAAA7UlEQVR4XuWTsQ4BQRRFHwVqNGqVqCV6jQhRKCh8hOj9gU5EIcQHkCg0Who/oyAhEoL7MrPJzNshu9lyT3Ky3DdzxewuUawowClMyUFQqvAN+/AjZoHhjWN9jVRSkmEYuhTh14uwAU+kSlqwbq0IQBsOSRXwofLngbUiBFyylmFYuKTpyLZwBZdwAZPmApMKuQ/1QuqsjvAAn/bYZkb+kgTMGt8nOvvJnfwlJnnYkaGEC/YyNHjJwAWX1GSoycGHDCVl+v9X5nAnQw/eeCW16CxmJvwA8q11wiU9fc2ImckNjmTowW/rBqblIIZ8AVkMMXkPm0fxAAAAAElFTkSuQmCC>

[image13]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABIAAAAYCAYAAAD3Va0xAAAAzklEQVR4Xu2SMQ5BQRCGf1EhcSCJAyChcwnlSxyAA7iB1iEUao3eERC8oKAQZjKz4k12F9Hul/zN+/682cwukPiVKeVEebxlo65E2Rp3oLTVe3FFHwuIq1th4clcXFmhxIYUGECKXSsUdjf70QfvJDSxCXEjK3y4o/coHcgyW5qlusqrHYGLa0pmMlQXOm0Bt5/QlX69nx3CExsQN7bCR+zoc4irWmEpQ4p/v58JpNi3AnJLH380o5wpR8qeklPu6mqUi35jx50r5GkkEo4nVItCIl5coUMAAAAASUVORK5CYII=>

[image14]: <data:image/png;base64,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>

[image15]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFIAAAAYCAYAAABp76qRAAACJElEQVR4Xu2XzSsGURTGj49QEmFjYUVK1qLYsZCPZEWxYCUrsrVQlCwQCgslG9mwsLCwtFD8BXaKkuQzilI+zuPOm5lj7szc4p0r86unmTnnzPs+98zMnTtECQkJ/5Ay1hIrRyYsZ4A1KINxUc96Y/Wx3kXOZs5Yw6xT1rnIxQKaN+Ns/0oj8+nLK7ZzrlxswEi1DFrOJll20bvJMkMRgedXGYyDClYr65CUqQ5Wi6fCTuCxnZTnI1Ybq9FTkWY6WaOkDOFFg/0RT4Wd4OWS8r3i7Hd5KmIChrZk0HJqycLpCIbwqEjwVryXQUtYJv9GLpKKl8qEIZmsYxkMQndlsTCvI/+cDTyS3psuHpU1VhMZ/g7mGN0JGaTPxQ18HcggU0xqvv8JjMb+RPoTTBo5JgOCSlazDArGZSAA+OqXQWaetU7qi+eE1evJmhF17J+geE8GHaI2Em961F3KhAvkg35rg1R+VyZ8KCdVmyUTpO7GWWc/m4L/Mwyjc1GM+cCPqI0EuPpBX0ZTpJqlo4DUtzIezTAmSO/LHccyacd1/Bwiie4/vlFDwcUmjUwnV+S/migkr1989ZS4jk0JHTsKHlirrBuRc2NTIzHHwkuDs8WKQjJNarWRIuV9wRUzIXTsKOhxtnkil+KFdcu6Zt2xJr3ptINmwG8VqZvAD9wURa7jCwqet3XgMceYMXb0YN+b/gJz2TYrVyYsZ4jUXJuQkJDwG3wAsJmFf4ltQTUAAAAASUVORK5CYII=>

[image16]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEYAAAAYCAYAAABHqosDAAAChUlEQVR4Xu2XS+hNURjFP28G5FWKKCZGRiIZScIAyYCBTBiQmYEBMTJRZvhnwICZIiGFmCFS5BWiFAMjeT8jrHX33v7fXefs+zhyU/f8anXPt9a++5y7z95nn2tWU1NT05/sht5Dn6FNkrXjCPQLegHNkIy8hfZA06Hh0ALoVlOL/5SH0CVXP4CuuboVP6GlruYALXd18lTbmlr0mEVqlDDOwoUq9MarKXAW6HdXlHisd0DHoM2S9ZQL0GtokgYl3LHiDyH0uERawTacbQp9LpvED3fcktnQIRu8I3Oh09C8Py26ZwT0GHoGjZKsFWlqKznfw/yymhb8/a7+7o6zDIHuQYstdPAcWhMzdnA0HnfKBOgVdN1C392SG4Cc72F+Vk0L/nlXf4MGoHfQ8ZgvdHmDE9BQaKWFBrNctjd6nTAT+gqd0qBLcgOQ8xP8DczLzk//qau52y1z9XwLbSY6z3bFzxtWPPHJEk/hsuNOcFCDiuQGIOd7mPMRoNC/oqbANhywAgwelXjtLoZbI9vs1KAiuXPmfA/zi2pa8A+7ms8/Jds/zXUl3kvxcqSZc0CDLvlg5RdYduMUtsntSum3bYj19sG4QenApLvuWR+90eK3g2+anywswyqsteK1EHocfI/OUg6Kfjc9PxIbYz3NeYTeE/EaOwgDvgyR9CDjRVZlrIXZdlWDDuC5t7h6X/Q8b6K31Xl8eOrN5AzU133t61yJ1yCN1v14/BGa2tSiOsOg2xb6HylZjjEWruMmdNfCbqdb/xwL70jKEgvfPWNh8HjTlckW2nC75ucXK/bfgCGXzr/mb2Zgz1llmWnU7/AveFqrUyTra7gmuSuttsw6q6mpqanIb6jHtgj7606WAAAAAElFTkSuQmCC>

[image17]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAXCAYAAAAyet74AAAAp0lEQVR4XmNgGAXUBOpAPB2IBaB8YyDeAMSmcBVAwAjEl4DYCYj/A/FDIA6Cyv0G4gVQNsNqIGYCYl8GiEIlmAQQdEDFwKAGSp9AFoSCNVjEwAIgd6KLoSjkhgpIIomxQ8XykcQY2qGCyOAREH9DEwP7DqTwAwPE9I1A/ApFBRSAFM0CYmYgDgNiflRpCAAJghTKokugg0kMmO7DCmBBAMLmaHKDAQAA1WwkZEfq36MAAAAASUVORK5CYII=>
