---
title: NLP--命名实体识别
author: 王哲峰
date: '2022-04-05'
slug: nlp-utils-spacy
categories:
  - NLP
tags:
  - tool
---

NLP--命名实体识别
=================================================

1.命名实体识别介绍
-------------------------------------------------

   - 命名实体识别(NER, Named Entity Recognition)

      - 与自动分词、词性标注一样, 命名实体识别也是自然语言处理的一个基础任务, 是信息抽取、信息检索、机器翻译、
        问答系统等多种自然语言处理技术必不可少的组成部分。

   - 命名实体识别的目的
      
      - 识别语料中人名、地名、组织机构名等命名实体。由于这些命名实体数量不断增加, 通常不可能在词典中穷尽列出, 
        且其构成方法具有各自的规律性, 因此, 通常把对这些词的识别在词汇形态处理(如汉语切分)任务中独立处理, 
        称为命名实体识别(Named Entities Recognition, NER)。

   - 命名实体识别研究的命名实体一般分为:

      - 3 大类:

         - 实体类
         - 时间类
         - 数字类

      - 7 小类:

         - 人名
         - 地名
         - 组织机构名
         - 时间
         - 日期
         - 货币
         - 百分比

      .. note:: 

         - 由于数量、时间、日期、货币等实体识别通常可以采用 **模式匹配** 的方式获得较好的识别效果, 
           相比之下人名、地名、机构名较复杂, 因此近年来的研究主要以这几种实体为主。

   - 命名实体识别效果的评判主要看:

      - 实体的边界是否划分正确
      - 实体的类型是否标注正确

   - 中文命名实体识别:

      - 中文的命名实体识别相比英文挑战更大, 在汉语中, 相较于实体类别标注子任务, 实体边界的识别更加困难, 主要难点如下:

         - 各类命名实体的数量众多
         - 命名实体的构成规律复杂
         - 嵌套情况复杂
         - 长度不确定

   - 命名实体识别也划分为三种方法:

      - 基于规则

         - 规则
         - 词典

      - 基于统计

         - 隐马尔科夫模型
         - 最大熵模型
         - 条件随机场

      - 规则和统计混合

         - NLP 并不完全是一个随机过程, 单独使用基于统计的方法是状态搜索空间非常庞大, 必须借助规则知识提前进行过滤修建处理

   - 命名实体识别中目前的主流方法
   
      - 序列标注方式

2.基于条件随机场的命名实体识别
-------------------------------------------------

- HMM 的局限性

   在 HMM 中, 将分词作为字标注问题来解决, 其中有两条非常经典的独立性假设:

      - 输出观察值之间严格独立
      - 状态的转移过程中当前状态只与前一状态有关(一阶马尔科夫模型)

   通过这两条假设, 使得 HMM 的计算成为可能, 模型的计算也简单许多。

   但多数场景下, 尤其在大量真实语料中, 观察序列更多的是以一种多重的交互特征形式表现出来, 
   观察元素之间广泛存在的长程相关性, 这样 HMM 的效果就受到了制约。

- CRF

   - 基于 HMM, 20001 年, Lafferty 等学者提出了条件随机场, 其主要思想来源于 HMM, 
     也是一种用来标记和切分序列化数据的统计模型。不同于 HMM 的是, 条件随机场是在给定观察的标记序列下, 
     计算整个标记序列的联合概率, 而 HMM 是在给定当前状态下, 定义下一个状态的分布。


   - 条件随机场的定义


3.示例
-------------------------------------------------

3.1 日期识别
~~~~~~~~~~~~~~~~~~~~~~~~

当针对结构化数据时, 日期设置一般有良好的规范, 在数据入库时予以类型约束, 在需要时能够通过解析还原读取到对应的日期。
然而在一些非结构化的数据应用场景下, 日期和文本混杂在一起, 此时日期的识别就变得艰难许多。非结构化数据下的日期识别多是与具体需求有关。

   - 任务及背景:

      - 现有一个基于语音问答的酒店预定系统, 其根据用户的每句语音进行解析, 识别出用户的酒店预定需求, 如房间型号、入住时间等; 
        用户的语音在发送给后台进行请求时已经转换成中文文本, 然而由于语音转换工具的识别问题, 许多日期类的数据并不是严格的数字, 
        会出现诸如 "六月 12" "2016年八月" "20160812" "后天下午"等形式。

      - 不关注问答系统的具体实现过程, 主要目的是识别出每个请求文本中可能的日期信息, 并将其转换成统一的格式进行输出。
        例如:“我要今天住到明天”(假设今天为2017年10月1号)那么通过日期解析后, 应该输出为 "2017-10-01" 和 "2017-10-02"。

   - 任务实现技术:

      - 正则表达式
      - Jieba 分词

(1)通过 Jieba 分词将带有时间信息的词进行切分, 然后记录连续时间信息的词

   - Jieba 词性标注提取文本中
   
      - "m": 数字
      - "t": 时间

   ```python

      import re
      from datetime import datetime, timedelta
      from dateutil.parser import parse
      import jieba.posseg as psg

      def time_extract(text):
         time_res = []
         word = ""
         keyDate = {
            "今天": 0,
            "明天": 1,
            "后天": 2,
         }

         for key, value in psg.cut(text):
            if key in keyDate:
                  if word != "":
                     time_res.append(word)
                  word = (datetime.today() + timedelta(days = keyDate.get(key, 0))).strftime("%Y年%m月%d日")
            elif word != "":
                  if value in ["m", "t"]:
                     word = word + key
                  else:
                     time_res.append(word)
                     word = ""
            elif value in ["m", "t"]:
                  word = key
         if word != "":
            time_res.append(word)
         result = list(filter(lambda x: x is not None, [check_time_valid(w) for w in time_res]))
         final_res = [parse_datetime(w) for w in result]

         return [x for x in final_res if x is not None]


      def check_time_valid(word):
         m = re.match("\d+$", word)
         if m:
            if len(word) <= 6:
                  return None
         word1 = re.sub("[号|日]\d+$", "日", word)
         if word1 != word:
            return check_time_valid(word1)
         else:
            return word1


      def parse_datetime(msg):
         if msg is None or len(msg) == 0:
            return None

         try:
            dt = parse(msg, fuzzy=True)
            return dt.strftime('%Y-%m-%d %H:%M:%S')
         except Exception as e:
            m = re.match(
                  r"([0-9零一二两三四五六七八九十]+年)?([0-9一二两三四五六七八九十]+月)?([0-9一二两三四五六七八九十]+[号日])?([上中下午晚早]+)?([0-9零一二两三四五六七八九十百]+[点:\.时])?([0-9零一二三四五六七八九十百]+分?)?([0-9零一二三四五六七八九十百]+秒)?",
                  msg)
            if m.group(0) is not None:
                  res = {
                     "year": m.group(1),
                     "month": m.group(2),
                     "day": m.group(3),
                     "hour": m.group(5) if m.group(5) is not None else '00',
                     "minute": m.group(6) if m.group(6) is not None else '00',
                     "second": m.group(7) if m.group(7) is not None else '00',
                  }
                  params = {}

                  for name in res:
                     if res[name] is not None and len(res[name]) != 0:
                        tmp = None
                        if name == 'year':
                              tmp = year2dig(res[name][:-1])
                        else:
                              tmp = cn2dig(res[name][:-1])
                        if tmp is not None:
                              params[name] = int(tmp)
                  target_date = datetime.today().replace(**params)
                  is_pm = m.group(4)
                  if is_pm is not None:
                     if is_pm == u'下午' or is_pm == u'晚上' or is_pm =='中午':
                        hour = target_date.time().hour
                        if hour < 12:
                              target_date = target_date.replace(hour=hour + 12)
                  return target_date.strftime('%Y-%m-%d %H:%M:%S')
            else:
                  return None


      UTIL_CN_NUM = {
         '零': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4,
         '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
         '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
         '5': 5, '6': 6, '7': 7, '8': 8, '9': 9
      }
      UTIL_CN_UNIT = {'十': 10, '百': 100, '千': 1000, '万': 10000}

      def cn2dig(src):
         if src == "":
            return None
         m = re.match("\d+", src)
         if m:
            return int(m.group(0))
         rsl = 0
         unit = 1
         for item in src[::-1]:
            if item in UTIL_CN_UNIT.keys():
                  unit = UTIL_CN_UNIT[item]
            elif item in UTIL_CN_NUM.keys():
                  num = UTIL_CN_NUM[item]
                  rsl += num * unit
            else:
                  return None
         if rsl < unit:
            rsl += unit
         return rsl

      def year2dig(year):
         res = ''
         for item in year:
            if item in UTIL_CN_NUM.keys():
                  res = res + str(UTIL_CN_NUM[item])
            else:
                  res = res + item
         m = re.match("\d+", res)
         if m:
            if len(m.group(0)) == 2:
                  return int(datetime.datetime.today().year/100)*100 + int(m.group(0))
            else:
                  return int(m.group(0))
         else:
            return None

      text1 = '我要住到明天下午三点'
      print(text1, time_extract(text1), sep=':')

      text2 = '预定28号的房间'
      print(text2, time_extract(text2), sep=':')

      text3 = '我要从26号下午4点住到11月2号'
      print(text3, time_extract(text3), sep=':')

      text4 = '我要预订今天到30的房间'
      print(text4, time_extract(text4), sep=':')

      text5 = '今天30号呵呵'
      print(text5, time_extract(text5), sep=':')


3.2 地名识别
~~~~~~~~~~~~~~~~~~~~~~~~~

   地名识别中将采用基于条件随机场的方法进行地名识别任务. 条件随机场模型的实现需要先安装 CRF++, 它是一款基于 C++ 高效实现 CRF 的工具。

   - 1.CRF++ 安装

      - Windows 安装:

         - 下载二进制版本:https://taku910.github.io/crfpp/

      - Linux/macOS 安装(>gcc3.0):

         - 下载源码:https://github.com/taku910/crfpp

         - 安装

            .. code-block:: shell

               $ git clone https://github.com/taku910/crfpp
               $ cd crfpp
               $ ./configure
               $ make
               $ sudo make install

   - 2.CRF++ Python 接口, 可以通过接口加载训练好的模型

      .. code-block:: shell

         cd python
         python setup.py build
         sudo python setup.py install

   - 3.使用 CRF++ 进行地名识别

      - 任务及背景:

         - 采用的预料数据集, 是 1998 年人民日报分词数据集, 该语料数据集主要是一个词性标注集。
           可以使用其中被标记为 "ns" 的部分来构造地名识别语料。如:"[香港/ns特别/a行政区/n]ns", 
           可以提取出 "香港特别行政区(中括号以内的"ns"在这里不再单独作为一个地名)"。按照这种思路, 
           对人民日报语料进行数据分析, 并切割了部分作为测试集进行验证。

      - (1)确定标签体系

         - 如同分词和词性标注一样, 命名实体识别也有自己的标签体系。一般用户可以按照自己的想法自行设计, 
           如下为地理位置标记规范, 即针对每个字标记为:"B", "E", "M", "S", "O" 中的一个:

               ======= ==================================
               标注     含义
               ======= ==================================
               B        当前词为地理命名实体的首部
               M        当前词为地理命名实体的内部
               E        当前词为地理命名实体的尾部
               S        当前词单独构成地理命名实体
               O        当前词不是地理命名实体或组成部分
               ======= ==================================

      - (2)语料数据处理

         - CRF++ 的训练数据要求一定的格式, 一般是一行一个 token, 一句话由多行 token 组成, 多个句子之间用空行分开。
           其中每行又分成多列, 除最后一列之外, 其他列表示特征。因此一般至少需要两列, 最后一列表示要预测的标签("B", "E", "M", "S", "O")。

         ```python

            pass
            

      - (3)特征模板设计

      - (4)模型训练和测试

      - (5)模型使用
