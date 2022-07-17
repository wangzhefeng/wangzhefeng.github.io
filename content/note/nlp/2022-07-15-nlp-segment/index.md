---
title: NLP--分词
author: 王哲峰
date: '2022-04-05'
slug: nlp-utils-spacy
categories:
  - NLP
tags:
  - tool
---

NLP--分词
===========================================

1.中文分词
-------------------------------------------

   - 在语言理解中, 词是最小的能够独立活动的有意义的语言成分. 将词确定下来是理解自然语言的第一步, 
     只有跨越了这一步, 中文才能像英文那样过渡到短语划分、概念抽取以及主题分析, 以致自然语言理解, 
     最终达到智能计算的最高境界.

   - **词** 的概念一直是汉语言语言学界纠缠不清而又绕不开的问题. 主要难点在于汉语结构与印欧体系语种差异甚大, 
     对词的构成边界方面很难进行界定.
      
      - 在英语中, 单词本身就是 **词** 的表达, 一篇英文文章就是 **单词** 加分隔符(空格)来表示的.
      - 在汉语中, 词以字为基本单位的, 但是一篇文章的语义表达却仍然是以词来划分的. 因此, 在处理中文文本时, 
        需要进行分词处理, 将句子转化为词的表示. 这个切词处理过程就是 **中文分词**, 它通过计算机自动识别出句子的词, 
        在词间加入边界标记符, 分隔出各个词汇.

         - 整个过程看似简单, 然而实践起来却很复杂, 主要的困难在于 **分词歧义**
         - 其他影响分词的因素是: **未登录词**、**分词粒度粗细**

   - 自中文自动分词被提出以来, 历经将近 30 年的探索, 提出了很多方法, 可主要归纳为:

      - 规则分词
      - 统计分词
      - 混合分词(规则+统计)

1.1 规则分词
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   基于规则的分词是一种机械的分词方法, 主要通过维护词典, 在切分语句时, 
   将语句的每个字符串与词汇表中的词逐一进行匹配, 找到则切分, 否则不予切分.

   按照匹配切分的方式, 主要有:
      
      - **正向最大匹配法**
      - **逆向最大匹配法**
      - **双向最大匹配法**

   .. note:: 

      基于规则的分词, 一般都较为简单高效, 但是词典的维护是一个很庞大的工程。
      在网络发达的今天, 网络新词层出不穷, 很难通过词典覆盖到所有词。


1.1.1 正向最大匹配法
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   - 正向最大匹配法思想:

      - 正向最大匹配(Maximum Match Method, MM 法)的基本思想为:假定分词词典中的最长词有 `$i` 个汉字字符, 
        则用被处理文档的当前字符串中的前 `$i` 个字符作为匹配字段, 查找字典。

         - 若字典中存在这样的一个 `$i` 字词, 则匹配成功, 匹配字段被作为一个词切分出来;
         - 若字典中找不到这样一个 `$i` 字词, 则匹配失败, 将匹配字段中的最后一个字去掉, 
           对剩下的字符串重新进行匹配处理;

      - 如此进行下去, 直到匹配成功, 即切分出词或剩余字符串的长度为零为止。这样就完成了一轮匹配, 
        然后取下一个 `$i` 字字符串进行匹配处理, 直到文档被扫描完为止。

   - 正向最大匹配算法描述如下:

      - (1)从左向右取待切分汉语句的 `$m$` 个字符作为匹配字段, `$m$` 为机器词典中最长词条的字符数
      - (2)查找机器词典并进行匹配。
         
         - 若匹配成功, 则将这个匹配字段作为一个词切分出来
         - 若匹配不成功, 则将这个匹配字段的最后一个字去掉, 剩下的字符串作为新的匹配字段, 进行再次匹配, 
           重复以上过程, 直到切分出所有词为止

   - 正向最大匹配法示例:

      ```python
         
         # -*- coding: utf-8 -*-
         
         class MM(object):
            """
            正向最大匹配
            """
            def __init__(self):
               self.window_size = 3

            def cut(self, text):
               result = []
               index = 0
               text_length = len(text)
               dic = ["研究", "研究生", "生命", "命", "的", "起源"]
               while text_length > index:
                     for size in range(self.window_size + index, index, -1):
                        piece = text[index:size]
                        if piece in dic:
                           index = size - 1
                           break
                     index = index + 1
                     result.append(piece + "----")
               print(result)

         if __name__ == "__main__":
            text = "研究生命的起源"
            tokenizer = MM()
            print(tokenizer.cut(text))

1.1.2 逆向最大匹配法
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   - 逆向最大匹配法思想:

      - 逆向最大匹配法(Reverse Maximum Match Method, RMM)的基本原理与正向最大匹配法相同, 
        不同的是分词切分的方向与正向最大匹配法相反。

      - 逆向最大匹配法从被处理文档的末端开始匹配扫描, 每次取最末端的 `$i` 个字符(`$i` 为词典中最长词数)
        作为匹配字段, 若匹配失败, 则去掉匹配字段最前面的一个字, 继续匹配。相应地, 它使用的分词词典是逆序词典, 
        其中的每个词条都按逆序存放。
      
      - 在实际处理时, 先将文档进行倒排处理, 生成逆序文档。然后, 根据逆序词典, 对逆序文档正向最大匹配法处理即可。
        由于汉语中偏正结构较多, 若从后向前匹配, 可以适当提高精确度。所以, 逆向最大匹配法比正向最大匹配法的误差
        要小。统计结果表明, 单纯使用正向最大匹配的错误率为 1/169, 单纯使用逆向最大匹配的错误率为 1/245.

   - 逆向最大匹配法示例:

      ```python

         # -*- coding: utf-8 -*-

         class RMM(object):
            """
            逆向最大匹配法
            """
            def __init__(self):
               self.window_size = 3
            
            def cut(self, text):
               result = []
               index = len(text)
               dic = ["研究", "研究生", "生命", "命", "的", "起源"]
               while index > 0:
                     for size in range(index - self.window_size, index):
                        piece = text[size:index]
                        if piece in dic:
                           index = size + 1
                           break
                     index = index - 1
                     result.append(piece + "----")
               result.reverse()
               print(result)

         if __name__ == "__main__":
            text = "研究生命的起源"
            RMM_tokenizer = RMM()
            print(RMM_tokenizer.cut(text))

1.1.3 双向最大匹配法
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   - 双向最大匹配法思想:

      - 双向最大匹配法(Bi-direction Matching Method)是将正向最大匹配法得到的分词结果和逆向最大匹配法得到的结果进行比较, 
        然后按照最大匹配原则, 选取词数切分最少的作为结果。

   - 双向最大匹配的规则是:

      - (1)如果正、反向分词结果词数不同, 则取分词数量较少的那个
      - (2)如果分词结果词数相同:

         - a.分词结果相同, 就说明没有歧义, 可返回任意一个
         - b.分词结果不同, 返回其中单字较少的那个

   - 双向最大匹配法示例:

      ```python
         
         # -*- coding: utf-8 -*-

         #TODO
         class BMM(object):
            """
            双向最大匹配法
            """
            def __init__(self):
               pass

            def cut(self, text):
               pass
         
         if __init__ == "__main__":
            text = "研究生命的起源"
            BMM_tokenizer = BMM()
            print(BMM_tokenizer.cut(text))

1.2 统计分词
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   随着大规模语料库的建立, 统计机器学习方法的研究和发展, 基于统计的中文分词算法逐渐成为主流。

   - 统计分词的主要思想是:

      - 把每个词看做是由词的最小单位的各个字组成的, 如果相连的字在不同的文本中出现的次数越多, 
        就证明这相连的字很可能就是一个词. 因此我们就可以利用 **字与字相邻出现的频率** 来反应 **成词的可靠度**, 
        统计语料中相邻共现的各个字的组合的频度, 当组合频度高于某一个临界值时,便可以认为此字组成会构成一个词语.

   - 基于统计的分词, 一般要做如下两步操作:

      - (1)建立统计语言模型
      - (2)对句子进行单词划分, 然后对划分结果进行概率计算, 获得概率最大的分词方式。这里就用到了统计学习算法, 
        如隐式马尔科夫(HMM)、条件随机场(CRF)等

1.2.1 语言模型
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   语言模型在信息检索、机器翻译、语音识别中承担着重要的任务。用概率论的专业术语描述语言模型就是:
   
      - 为长度为 `$m$` 的字符串确定其概率分布 `$P(\omega_{1}, \omega_{2}, \cdot, \omega_{m})$`, 
        其中 `$\omega_{1}$` 到 `$\omega_{m}$` 依次表示文本中的各个词语。一般采用链式法计算其概率值:

            `$$P(\omega_{1}, \omega_{2}, \cdots, \omega_{m})=

            P(\omega_{1})P(\omega_{2}|\omega_{1})P(\omega_{3}|\omega_{1}, \omega_{2}) \cdots P(\omega_{i}|\omega_{1}, \omega_{2}, \cdots, \omega_{i-1}) \cdots P(\omega_{m}|\omega_{1}, \omega_{2}, \cdots, \omega_{m-1})$$`

      - `$n` 元模型(n-gram model)

         - 当文本过长时, 公式右部从第三项起的每一项计算难度都很大。为了解决该问题, 有人提出了 `$n` 元模型(n-gram model) 降低该计算难度。
           所谓 `$n` 元模型就是在估算条件概率时, 忽略距离大与等于 `$n` 的上下文词的影响, 因此:

               `$$P(\omega_{i}|\omega_{1}, \omega_{2}, \cdots, \omega_{i-1}) = P(\omega_{i}|\omega_{i-(n-1)}, \omega_{i-(n-2)}, \cdots, \omega_{i-1})$$`

         - 当 `$n=1` 时, 称为一元模型(unigram model), 此时整个句子的概率可以表示为: 

               `$$P(\omega_{1}, \omega_{2}, \cdots, \omega_{m}) = P(\omega_{1})P(\omega_{2}) \cdots P(\omega_{m})$$`

            - 在一元模型中, 整个句子的概率等于各个词语概率的乘积, 即各个词之间都是相互独立的, 
              这无疑是完全损失了句中的词序信息, 所以一元模型的效果并不理想.

         - 当 `$n=2` 时, 称为二元模型(bigram model), 概率的计算变为:

              `$$P(\omega_{i}|\omega_{1}, \omega_{2}, \cdots, \omega_{i-1}) = P(\omega_{i}|\omega_{i-1})$$`

         - 当 `$n=3` 时, 称为三元模型(trigram model), 概率的计算变为:

               `$$P(\omega_{i}|\omega_{1}, \omega_{2}, \cdots, \omega_{i-1}) = P(\omega_{i}|\omega_{i-2},\omega_{i-1})$$`

         - 当 `$n \geq 2` 时, 该模型是可以保留一定的词序信息的, 而且 `$n` 越大, 保留的词序信息越丰富, 但计算成本也呈指数级增长。
           一般使用频率计数的比例来计算 `$n` 元条件概率:

               `$$P(\omega_{i}|\omega_{i-(n-1)}, \omega_{i-(n-2)}, \cdots, \omega_{i-1}) = \frac{count(\omega_{i-(n-1)}, \cdots, \omega_{i-1},\omega_{i})}{count(\omega_{i-(n-1)}, \cdots, \omega_{i-1})}$$`

            - 其中,  `$count(\omega_{i-(n-1)}, \cdots, \omega_{i-1})$` 表示词语 `$\omega_{i-(n-1)}, \cdots, \omega_{i-1}$` 在语料库中出现的总次数

         - 综上, 当 `$n` 越大时, 模型包含的词序信息越丰富, 同时计算量随之增大。与此同时, 长度越长的文本序列出现的次数也会越少, 这样, 按照上式估计 `$n` 元条件概率时, 
           就会出现分子、分母为零的情况。因此, 一般在 `$n` 元模型中需要配合相应的平滑算法解决该问题, 如拉普拉斯平滑算法等。

1.2.2 HMM 模型
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   隐马尔科夫模型(HMM)是将分词作为字在字符串中的序列标注任务来实现的。

   - 隐马尔科夫模型的基本思路是:

      - 每个字在构造一个特定的词语时都占据着一个确定的构词位置(即词位), 现规定每个字最多只有四个构词位置:

         - B(词首)
         - M(词中)
         - E(词尾)
         - S(单独成词)
   
      - 用数学抽象表示如下:

         - 用 `$\lambda = \lambda_{1}\lambda_{2}\lambda_{n}$` 代表输入的句子, `$n` 为句子长度,  
           `$\lambda_{i}$` 表示字,  `$o=o_{1}o_{2} \cdots o_{n}$` 代表输出的标签, 那么理想的输出即为:

               `$$max = max P(o_{1}o_{2} \cdots o_{n}|\lambda_{1}\lambda_{2} \cdots \lambda_{n})$$`

         - 在分词任务上,  `$o` 即为 B、M、E、S 这四种标记,  `$\lambda` 为诸如 “中”、“文” 等句子中的每个字(包括标点等非中文字符).
         - 需要注意的是,  `$P(o|\lambda)$` 是关于 2n 个变量的条件概率, 且 n 不固定。因此, 几乎无法对 `$P(o|\lambda)$` 进行精确计算。
           这里引入观测独立性假设, 即每个字的输出仅仅与当前字有关, 于是就能得到下式:

               `$$P(o_{1}o_{2} \cdots o_{n}|\lambda_{1}\lambda_{2} \cdots \lambda_{n}) = p(o_{1}|\lambda_{1})p(o_{2}|\lambda_{2}) \cdots p(o_{n}|\lambda_{n})$$`

   - 示例:

      - 下面句子(1)的分词结果就可以直接表示成如(2)所示的逐字标注形式:

         (1)中文 / 分词 / 是 /. 文本处理 / 不可或缺 / 的 / 一步！
         (2)中/B 文/E 分/B 词/E 是/S 文/B 本/M 处/M 理/E 不/B 可/M 或/M 缺/E 的/S 一/B 步/E！/S

1.2.3 其他统计分词算法
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   - 条件随机场(CRF)也是一种基于马尔科夫思想的统计模型。

      - 在隐马尔科夫模型中, 有个很经典的假设, 就是每个状态只与它前面的状态有关。这样的假设显然是有偏差的, 
        于是, 学者们提出了条件随机场算法, 使得每个状态不止与它前面的状态有关, 还与它后面的状态有关。
   
   - 神经网络分词算法是深度学习方法在 NLP 上的应用。

      - 通常采用 CNN、LSTM 等深度学习网络自动发现一些模式和特征, 然后结合 CRF、softmax 等分类算法进行分词预测。

   - 对比于机械分词法, 这些统计分词方法不需要耗费人力维护词典, 能较好地处理歧义和未登录词, 是目前分词中非常主流的方法。
     但其分词的效果很依赖训练预料的质量, 且计算量相较于机械分词要大得多。


1.3 混合分词
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   事实上, 目前不管是基于规则的算法、基于 HMM、CRF 或者 deep learning 等的方法, 
   其分词效果在具体任务中, 其实差距并没有那么明显。

   在实际工程应用中, 多是基于一种分词算法, 然后用其他分词算法加以辅助。最常用的方式就是先基于词典的方式进行分词, 
   然后再用统计方法进行辅助。如此, 能在保证词典分词准确率的基础上, 对未登录词和歧义词有较好的识别。
   
   `jieba` 分词工具就是基于这种方法的实现。


2.外文分词
-------------------------------------------




3.jieba 分词
-------------------------------------------

3.1 安装
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   .. code-block:: shell

      $ pip install paddlepaddle-tiny=1.6.1 # Python3.7
      $ pip install jieba

3.2 特点、算法
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   - 特点:

      - 支持四种分词模式:

            - **精确模式**:试图将句子最精确地切开, 适合文本分析

            - **全模式**:把句子中所有的可以成词的词语都扫描出来, 速度非常快, 但是不能解决歧义

            - **搜索引擎模式**:在精确模式的基础上, 对长词再次切分, 提高召回率, 适合用于搜索引擎分词

            - **paddle 模式**:利用 [PaddlePaddle](https://www.paddlepaddle.org.cn/)  深度学习框架, 训练序列标注(双向GRU)网络模型实现分词。同时支持词性标注

               - paddle 模式使用需安装 `paddlepaddle-tiny`

                  .. code-block:: shell

                     $ pip install paddlepaddle-tiny=1.6.1

               - 目前 paddle 模式支持 jieba v0.40 及以上版本, jieba v0.40以下版本, 请升级 jieba

                  .. code-block:: shell

                     $ pip install jieba --upgrade

      - 支持繁体分词
      - 支持自定义词典
      - MIT 授权协议

   - 算法:

      - 基于前缀词典实现高效的词图扫描, 生成句子中汉字所有可能成词情况所构成的有向无环图(DAG)
      - 采用了动态规划查找最大概率路径, 找出基于词频的最大切分组合
      - 对于未登录词, 采用了基于汉字成词能力的 HMM 模型, 使用了 Viterbi 算法


3.3 分词
~~~~~~~~~~~~~~~~~

   - API

      - `jieba.enable_paddle()$`
      - `jieba.cut(sentence = "", cut_all = False, HMM = True, use_paddle = False)$`
      - `jieba.lcut(sentence = "", cut_all = False, HMM = True, use_paddle = False)$`
      - `jieba.cut_for_search(sentence = "", HMM = True)$`
      - `jieba.lcut_for_search(sentence = "", HMM = True)$`
      - `jieba.Tokenizer(dictionary = DEFAULT_DICT)$`

         - 新建自定义分词器, 可用于同时使用不同词典, jieba.dt 为默认分词器, 所有全局分词相关函数都是该分词器的映射

3.4 添加自定义词典
~~~~~~~~~~~~~~~~~~~

开发者可以指定自己自定义的词典, 以便包含 jieba 词库里没有的词。虽然 jieba 有新词识别能力, 但是自行添加新词可以保证更高的正确率。

   - 用法

      - 1.创建新的词典
         
         - 词典格式:

            - 文件名: `dict.txt`
            - 文件格式: 若为路径或二进制方式打开的文件, 则文件必须为 UTF-8 编码
            - 一个词占一行
            - 每一行分三部分:词语、词频(可省略)、词性(可省略), 用空格分隔开顺序不可颠倒

      - 2.使用 jieba.load_userdict(file_name) 载入自定义词典
      - 3.更改分词器(默认为 jieba.dt)的 `tmp_dir` 和 `cache_file` 属性, 可分别指定缓存文件所在的文件夹及其文件名, 用于受限的文件系统

   - API

      - `jieba.load_userdict(file_name)$`
         
         - 载入自定义词典

      - `jieba.dt.tmp_dir`
      - `jieba.dt.cache_file`

   - 示例

      ```python

         import sys
         import jieba
         import jieba.posseg as pseg
         sys.path.append("./util_data")
         jieba.load_userdict("./util_data/userdict.txt")
         
         jieba.add_word("石墨烯")
         jieba.add_word("凱特琳")
         jieba.add_word("自定义词")
         
         test_sent = (
            "李小福是创新办主任也是云计算方面的专家; 什么是八一双鹿\n"
            "例如我输入一个带“韩玉赏鉴”的标题, 在自定义词库中也增加了此词为N类\n"
            "「台中」正確應該不會被切開。mac上可分出「石墨烯」; 此時又可以分出來凱特琳了。"
         )
         words = jieba.cut(test_sent)
         print(" ".join(words))

3.5 调整词典
~~~~~~~~~~~~~~~~~~~~

   - API

      - `add_word(word, freq = None, tag = None)$`
      - `del_word(word)$`
      - `suggest_freq(segment, tune - True)$`

   - 示例

      ```python

         string = "如果放到旧字典中将出错。"
         seg_list = jieba.cut(string, HMM = False)
         print(" ".join(seg_list))
         jieba.suggest_freq(segment = ("中", "将"), tune = True)
         seg_list_tuned = jieba.cut(string, HMM = False)
         print(" ".join(seg_list_tuned))

3.5 关键词提取
~~~~~~~~~~~~~~~~~~

3.5.1 基于 TF-IDF 算法的关键词提取
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   - API

      - `jieba.analyse.extract_tags(sentence, topK = 20, withWeight = False, allowPOS = ())$`
      - `jieba.analyse.TFIDF(idf_path = None`

         - 新建 TF-IDF 实例, idf_path 为 IDF 频率文件

   - 用法

      - 关键词提取所使用的逆向文档频率(IDF)文本语料库可以切换成自定义语料库的路径

         - 

      - 关键词提取所使用停止词(Stop Words)文本语料库可以切换成自定义语料库的路径



3.5.2 基于 TextRank 算法的关键词提取
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   - API

      - `jieba.analyse.textrank(sentence, topK = 20, withWeight = False, allowPOS = ("ns", "n", "vn", "v"))$`
      - `jieba.analyse.TextRank()$`

   - 算法论文

      - [TextRank: Bringing Order into Texts](http://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf) 

   - 基本思想:

      - 1.将待抽取关键词的文本进行分词
      - 2.以固定窗口大小(默认为5, 通过span属性调整), 词之间的共现关系, 构建图
      - 3.计算图中节点的PageRank, 注意是无向带权图

   - 使用示例:

      - test

3.6 词性标注
~~~~~~~~~~~~~


3.7 并行分词
~~~~~~~~~~~~~~


3.8 Tokenize:返回词语在原文的起止位置
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


3.9 ChineseAnalyzer for Whoosh 搜索引擎
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   - API

      - from jieba.analyse import ChineseAnalyzer

   - 示例

      ```python

         

3.10 命令行分词
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   - 语法

      .. code-block:: shell

         $ python -m jieba [option] filename

   - 示例

      .. code-block:: shell

         $ python -m jieba news.txt > cut_result.txt

   - 命令行选项

      - filename
      - `python -m jieba -h`, --help
      - `-d [DELIM]`, --delimiter [DELIM]
      - `-p [DELIM]`, --pos [DELIM]
      - `-D DICT`
      - `-a`, --cut-all
      - `-n`, --no-hmm
      - `-q`, --quiet
      - `-V`, --version



4.其他分词
-------------------------------------------

   - 常用分词库

      - StanfordNLP

      - 哈工大语言云

      - 庖丁解牛分词

      - 盘古分词 (ICTCLAS, 中科院汉语词法分析系统)

      - IKAnalyzer (Luence项目下, 基于java) 

      - FudanNLP (复旦大学) 

      - 中文分词工具

      - `Ansj`

      - 盘古分词

      - `jieba`
