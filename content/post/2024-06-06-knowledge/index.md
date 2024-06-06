---
title: 知识体系
author: 王哲峰
date: '2024-06-06'
slug: knowledge
categories:
  - blog
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

- [概率与统计](#概率与统计)
    - [Basic](#basic)
        - [Basic Term](#basic-term)
        - [Probability Distribution](#probability-distribution)
    - [Information Theory](#information-theory)
    - [Sampling](#sampling)
    - [Model](#model)
    - [Baysian](#baysian)
        - [Likelihood](#likelihood)
        - [Inference](#inference)
- [机器学习](#机器学习)
    - [Linear Regression](#linear-regression)
    - [Logistic Regression](#logistic-regression)
    - [Non-Probabilistic](#non-probabilistic)
    - [Ensemble](#ensemble)
    - [Clustering](#clustering)
    - [Dimensionality Reduction](#dimensionality-reduction)
    - [Training](#training)
    - [Regularization](#regularization)
- [文本挖掘](#文本挖掘)
    - [Basic Procedure](#basic-procedure)
    - [NLP Basic HyPothesis](#nlp-basic-hypothesis)
    - [Sequential Labeling](#sequential-labeling)
    - [Word Embedding](#word-embedding)
    - [Graph](#graph)
    - [Document](#document)
        - [String Distance](#string-distance)
        - [Classification](#classification)
        - [Clustering](#clustering-1)
        - [Bag of Words model](#bag-of-words-model)
        - [Embedding](#embedding)
- [自然语言处理](#自然语言处理)
    - [Task](#task)
    - [Basic](#basic-1)
        - [Recurrent Model](#recurrent-model)
        - [Convolutional Model](#convolutional-model)
        - [Recursive Model](#recursive-model)
    - [Language Model](#language-model)
        - [Word Representation to Contextual Representation](#word-representation-to-contextual-representation)
        - [Encoder-Decoder Model](#encoder-decoder-model)
    - [Distributed Representation](#distributed-representation)
</p></details><p></p>

# 概率与统计

![img](images/prob_stat.png)

## Basic

* Sample Space：样本空间
* Random Variable：随机变量
* Independent：独立性
* Conjucate：共轭

### Basic Term

* Expecation：期望
* Mean：均值
    - Median：中位数
* Variance：方差
    - Bias-trade off：偏差权衡
* Covariance：协方差
    - Correlation Coefficient：相关系数
* Distance：距离
    - Manhattan
    - Euclidean
    - Mahalanobis
    - Correlation

### Probability Distribution

* PDF：概率分布函数
* CDF：累积分布函数
* Uniform：均匀分布
* GMM - EM ALgorithm
    - Gaussian：高斯分布（正态分布）
    - Categorical：类别分布
* Bernoulli：伯努利分布
* Binomial：二项分布
* MultiNormial：多元正态分布
* Dirichlet：狄利克雷分布
* Beta：贝塔分布
* Gamma：伽马分布
* Exponential：指数分布
* Chi-squared：卡方分布
* Student-t：t 分布
    - t-SNE
* Central Limit Theorem：中心极限定理
* degree of freedom：自由度

## Information Theory

* KL Divergence：KL 散度
* Cross Entropy：交叉熵
    - Uncertainty
* Shannon Entropy：信息熵

## Sampling

* Candidate Sampling：候选人抽样
* Negative Sampling：负采样
* MCMC：马尔科夫蒙特卡洛模拟
    - Monte Carlo Method：蒙特卡洛方法
    - Markov Chain：马尔科夫链
* Gibbs：吉布斯采样

## Model

* Discriminative：判别性
    - Linear Regression
* Gerenative
    - Unsupervisor
        - VAE
        - GAN
    - Supervisor
        - Navie Bayes
        - Linear Discriminant Analysis
        - Logistic Regression

## Baysian

* Joint Probability
* Posterior
* Evidence

### Likelihood

* Marginal Probability
* Class-Conditional Density

### Inference

- Estimator
    - Statistical Estimation
    - Parameter Estimator
        - Bayesian Estimate
        - MAP
            - Weight Decay Term
        - MLE
            - Minimize Cross-Entropy
            - Overfitting
        - Latent Variable
        - EM Algorithm
- Statistical Hypothesis
    - p-value
    - Significance Level
    - T-test
    - ANOVA
        - Bonferroni
        - FDR
- Baysian Inference
    - Variational
        - Reparameterization Trick
        - Normalizing Flow
        - Mean Field Approximate

# 机器学习

![img](images/ml.png)

## Linear Regression

* MSE
* Regularized
    - Ridge(L2)
    - LASSO(L1)
    - Elastic Net(L1 + L2)
        - Forbenius Norm
        - Norm

## Logistic Regression

* From Gerneative to Discriminative Proof
    - logits
    - Activation Function
        - Sigmoid
        - tanh
        - ReLU
        - Leakly ReLU
        - PReLU
        - ELU
        - Maxout
        - Representation Learning
* Maximize Likelihood is Minimize Cross Entropy Proof
    - Gradient Descent
        - Taylor Series Proof
        - Convex Function
            - Jensen's inequality
        - SGD(b = 1)
            - Optimization
                - Momentum
                - NAG
                - Adagrad
                - RMSProp 
                - AdaDelta
                - Adam
                - Nadam
        - Learning Rate
        - Backpropagation Proof
            - Gradient Vanishing & Exploding Proof
        - Finite Difference Method

## Non-Probabilistic

* SVM
    - Lagrangian approach
    - KKT Condition
        - Duality
            - Primal Problem
            - Dual Problem
    - C-SVM
    - Kernel SVM
        - Mapping Function
        - Kernel Trick

## Ensemble

* Bagging
* AdaBoost
* Random Forest

## Clustering

* Hierarchical
* Pational
    - K-means
* SOM
* Spectral
* KNN Algorithm
* Instance-based Learning

## Dimensionality Reduction

* Eigen Decomposition
    - Eigen Value
    - Eigen Vector
    - PCA
        - Kernel PCA
    - SVD
    - Matrix Property
        - Idempotent
        - Square Matrix
        - Symmetric
        - Orthogonal
        - Singular

## Training

* Data Splitting
    - Training
    - Validation
    - Test
    - k-fold cross validation
* Score
    - Diagnosing Model
        - Variance-Bias
    - Confusion Matrix
        - Accuracy
        - AUC-ROC 
        - Presicion
        - Recall
            - Trade Off
            - Type 1, 2 Error
        - F1 Score

## Regularization

* **Early Stopping**
* **Weight Decay**
* **Dropout**
* **Normalization**
    - **Layer**
    - **Batch**
    - **Weight**

# 文本挖掘

![img](images/textmining.png)

## Basic Procedure

- Morphological & Lexical
    - Analysis Procedure
        - Sentence splitting
        - Morphological Analysis
        - Part-Of-Speech Tagging
    - Task
- Syntax
    - Parsing
        - CKY Algorithm
        - Graph Parsing

## NLP Basic HyPothesis

## Sequential Labeling

- HMM
    - Markov Assumption
    - Emission Probablity
    - Transition Probability
- Maximum Entropy Model
- Maximum Entropy Markov Model
- Condition Random Field
- RNN

## Word Embedding

* Topic Modeling
    - LDA(LDI)
        - SVD
        - Probabilistic LDA(pLDI)
            - Multinomial
    - Sparse Coding
    - NMF -> Matrix Factorization
    - Laten Dirichlet Allocation
        - Dirichlet
        - Gibbs Sampling -> Perplexity
* One-hot Representation
    - Curse of Dimension
* Distributed Representation
    - Dimensionality Reduction
    - Co-Occurrence(n-gram)
        - Word2Vec
            - CBOW
            - Skip-gram
        - GloVe
        - FastText
        - Co-Occurrence + Shifted PPMI
    - Implicit Matrix Factorization
        - PMI
        - PPMI
        - Positive PMI
        - Shifted PPMI
* Character Embedding

## Graph

* Ranking Algorithm
    - Text Rank
    - Word Rank
    - PageRank
    - HITS
* Similarity
    - RWR
    - SimRank

## Document

### String Distance

- Levenshitein
- Jaccard
- Cosine

### Classification

- Logistic Regression
- LASSO Regression
- Naive Bayes Classifer
- Support Vector Machine
- Decision Tree

### Clustering

- K-Means
    - Lloyd
    - Euclidean
    - Spherical
    - Cosine
    - Voronoi Diagram
- GMM
- Bayesian GMM
- Hierarchical Clustering
- DBSCAN

### Bag of Words model

- Term Vector Representation
    - TF-IDF
- Distance & Similarity
    - Euclidean
    - Cosine

### Embedding

* Doc2Vec

# 自然语言处理

![img](images/nlp.png)

## Task

* POS Tagging：词性标注
* Parsing：解析
* Named Entity Recognition：命令实体识别
* Coreference Resolution：指代消解
* Sentiment Analysis：情感分析
* Machine Translation：机器翻译
    - BLEU Score
* Question Answering：问答系统
* Reading Comprehension：阅读理解
* Text Generation：文本生成
* Summarization：文本摘要
    - ROUGE Score
* Dialogue Systems：对话系统
* Language Modeling：语言模型
    - Preplexity

## Basic

### Recurrent Model

* Sequential Model
    - n-gram
    - Katz-Backoff Model
    - RNN
        - Bi-directional RNN
        - Deep RNN
        - BPTT 
        - Vanishing Gradient
    - LSTM
        - Cell State
    - GRU
    - Teacher Forcing
    - Non-Teacher Forcing
* `$P(w_{i}|w_{1:i-1})$`
    - Greedy Search
    - Beam Search
### Convolutional Model

* Kernel Size = n-gram
* Channel = number of Perception
* TextCNN
* DCNN

### Recursive Model

* Syntatically-Untied RNN
* RNTN
* Matrix-Vector RNN

## Language Model

### Word Representation to Contextual Representation

* CoVe
* ELMo
    - 2 Layers LSTM
    - **Pretrained for Embedding**
* State of the Art Model
    - Models 1：
        - **Pretraining-Finetuning**
        - Transformer Based Model
    - Models 2：
        - **Transformer**
        - **OpenAI-GPT**
            - `$P(w_{i}|w_{1:i-1})$`
            - Auxiliary Task Head
        - **BERT**
            - Masking Language Model
            - Next Sentence Predict(NSP)
                - CLS
                    - SEP
                - MASK
        - Universal Transformer
            - Inductive Bias of RNN
        - **OpenAI-GPT2**
            - Zero-Shot Learning
            - Not FineTuning
        - **OpenAI-GPT3**
        - RoBERTTa
            - Skip NSP
            - Large Batch
        - DistilBERT
            - lighter BERT
    - Models 3：
        - Relative Positional Encoding
        - Transformer XL
            - Segment-Level Recurrence with State Reuse
        - XLNet
            - Two-Stream Self-Attention for Target-Aware
                - Content Stream
                - Query Stream
            - Permutation Language Model
            - Segment Recurrence Mechanism


### Encoder-Decoder Model

* Source-Target
    - OOV Problem Slove
        - Subword Tokenizing based NMT
            - Byte Pair Encoding
                - Based Frequency
            - Word Piece Model
                - Based Likelihood
            - Subword Regularization
                - EM Algorithm
                - Google Sentence Piece
        - Copying Mechanism
    - Task
        - Machine Translation
        - Summarization
* Model
    - Seq2Seq
    - Attention Mechanism
        - Context Vector
        - Alignment Model
        - Query
        - Key Value
    - Seq2Seq based on
    - Skip-Thought
    - Transformer
        - Self-Attention
            - Scale-Dot Product Attention
        - Positional Encoding
            - Sinusoid Function
        - Poswise FeedForwardNet
        - Multi-Head Attention
        - Masked Multi-Head Attention
    - Unsupervised Machine Translation

## Distributed Representation

* Word Representation
* Co-Occurrence Matrix
    - Co-Occurrences + Shifted PPMI
* NNLM
    - Embedding
        - **Word Embedding**
            - **Word2Vec**
                - CBOW
                - Skip-gram
            - CostMethod
                - Hierarchical Softmax
                - Negative Sampling
            - **GloVe**
            - **FastText**
        - **Character Embedding**
