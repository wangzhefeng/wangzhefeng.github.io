---
title: NLP-词性标注
author: 王哲峰
date: '2022-04-05'
slug: nlp-tagging
categories:
  - nlp
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
img {
    pointer-events: none;
}
</style>

<details><summary>目录</summary><p>

- [词性标注介绍](#词性标注介绍)
- [词性标注规范](#词性标注规范)
  - [北大词性标注集](#北大词性标注集)
  - [宾州词性标注集](#宾州词性标注集)
- [jieba 分词中的词性标注](#jieba-分词中的词性标注)
  - [jieba 词性标注基本思路](#jieba-词性标注基本思路)
  - [jieba 词性标注示例](#jieba-词性标注示例)
</p></details><p></p>

# 词性标注介绍

**词性** 是词汇基本的语法属性, 通常称为 **词类**
   
**词性标注(part-of-speech tagging)** 是在给定句子中判断每个词的语法范畴, 确定其词性并加以标注的过程. 例如:

- 表示人、地点、事物以及其他抽象概念的名称即为 **名词**
- 表示动作或状态变化的词为 **动词**
- 描述或修饰名词属性、状态的词为 **形容词**

词性标注最简单的方法是从预料库中统计每个词对应的高频词性, 将其作为默认的词性. 但这样显然还有提升空间. 
目前较为主流的方法是如同分词一样, 将句子的词性标注作为一个序列标注问题来解决, 那么分词中常用的手段, 
如隐含马尔科夫模型, 条件随机场模型等皆可在词性标注任务中使用.

> 在中文中, 一个词的词性很多时候都不是固定的, 一般表现为同音同形的词在不同场景下, 其表示的语法截然不同, 这就为词性标注带来很大的困难; 
  但是另外一方面, 从整体上看, 大多数词语, 尤其是实词, 一般只有一到两个词性, 且其中一个词性的使用频次远远大于另一个, 
  即使每次将高频词性选择进行标注, 也能实现 80% 以上的准确率. 如此,若我们对常用的词性能够进行很好地识别, 那么就能够覆盖绝大多数场景, 
  满足基本的准确度要求.

# 词性标注规范

- 词性标注需要有一定的标注规范, 如将词分为名词、动词、形容词, 然后用 `n`、`v`、`adj` 等来进行表示.
- 中文领域中尚无统一的标注标准, 较为主流的是以下两类, 两类标注方式各有千秋, 一般任选一种方式即可:
    - 北大词性标注集
    - 宾州词性标注集

## 北大词性标注集

北大词性标注规范表：

| 标记   |   词性   |       说明  |
|-------|----------|-----------------------------------------------------|
| `ag`  | 形语素     |    形容词性语素. 形容词代码为 a, 语素代码 g 前面置以 a |
| `a`   | 形容词     |    取英语形容词 adjective 的第 1 个字母 |
| `ad`  | 副形词     |    直接作状语的形容词. 形容词代码 a 和副词代码 d 并在一起 |
| `an`  | 名形词     |    具有名词功能的形容词. 形容词代码 a 和名词代码 n 并在一起 |
| `b$`  |  区别词    |     取汉字“别”的声母 |
| `c`   | 连词      |     取英语连词 conjunction 的第 1 个字母 |
| `dg`  | 副语素     |    副词性语素.副词代码为 d, 语素代码 ｇ 前面置以 d |
| `d`   | 副词      |     取 adverb 的第 2 个字母, 因其第 1 个字母已用于形容词 |
| `e`   | 叹词      |     取英语叹词 exclamation 的第 1 个字母 |
| `f`   | 方位词     |    取汉字“方”的声母 |
| `g`   | 语素      |     绝大多数语素都能作为合成词的“词根”, 取汉字“根”的声母 |
| `h`   | 前接成分    |    取英语 head 的第 1 个字母 |
| `i`   | 成语      |     取英语成语 idiom 的第 1 个字母 |
| `j`   | 简称略语    |    取汉字“简”的声母 |
| `k`   | 后接成分    ||
| `l`   | 习用语     |    习用语尚未成为成语, 有点“临时性”, 取“临”的声母 |
| `m`   | 数词      |     取英语 numeral 的第 3 个字母, n ,u 已有他用 |
| `ng`  | 名语素     |    名词性语素.名词代码为 n, 语素代码 g 前面置以 n |
| `n`   | 名词      |     取英语名词 noun 的第 1 个字母 |
| `nr`  | 人名      |     名词代码 n 和“人(ren)”的声母并在一起 |
| `ns`  | 地名      |     名词代码 n 和处所词代码 s 并在一起 |
| `nt`  | 机构团体    |   “团”的声母为 t, 名词代码 n 和 t 并在一起 |
| `nz`  | 其他专名    |   “专”的声母的第 1 个字母为 z, 名词代码 n 和 z 并在一起 |
| `o`   | 拟声词     |    取英语拟声词 onomatopoeia 的第 1 个字母 |
| `p`   | 介词      |     取英语介词 prepositional 的第 1 个字母 |
| `q`   | 量词      |     取英语 quantity 的第 1 个字母 |
| `r`   | 代词      |     取英语代词 pronoun 的第 2 个字母, 因 p 已用于介词 |
| `s`   | 处所词     |    取英语 space 的第 1 个字母 |
| `Tg`  | 时语素     |    时间词性语素. 时间词代码为 t, 在语素的代码 g 前面置以 t |
| `t`   | 时间词     |    取英语 time 的第1个字母 |
| `u`   | 助词      |     取英语助词 auxiliary 的第 2 个字母, 因 a 已用于形容词 |
| `vg`  | 动语素     |    动词性语素.动词代码为 v. 在语素的代码 g 前面置以 v |
| `v`   | 动词      |     取英语动词 verb 的第一个字母 |
| `vd`  | 副动词     |    直接作状语的动词. 动词和副词的代码并在一起 |
| `vn`  | 名动词     |    指具有名词功能的动词. 动词和名词的代码并在一起 |
| `w`   | 标点符号    ||
| `x`   | 非语素字    |   非语素字只是一个符号, 字母 x 通常用于代表未知数、符号 |
| `y`   | 语气词     |    取汉字“语”的声母 |
| `z`   | 状态词     |    取汉字“状”的声母的前一个字母 |


## 宾州词性标注集

| 标记   |   英语解释                             | 中文解释                            |
--------|---------------------------------------|------------------------------------|
|`AD`   | adverbs                               | 副词 |
|`AS`   | Aspect marker                         | 体态词, 体标记 (例如:了, 在, 着, 过)  |
|`BA`   | 把 in ba-const                         |“把”, “将”的词性标记 |
|`CC`   | Coordinating conjunction              | 并列连词, “和” |
|`CD`   | Cardinal numbers                      | 数字, “一百” |
|`CS`   | Subordinating conj                    | 从属连词 (例子:若, 如果, 如…)  |
|`DEC`  | 的 for relative-clause etc             |“的”词性标记 |
|`DEG`  | Associative                           | 联结词“的” |
|`DER`  | in V-de construction, and V-de-R      | “得” |
|`DEV`  | before VP                             | 地 |
|`DT`   | Determiner                            | 限定词, “这” |
|`ETC`  | Tag for words, in coordination phrase | 等, 等等 |
|`FW`   | Foreign words                         | 例子:ISO |
|`IJ`   | interjetion                           | 感叹词 |
|`JJ`   | Noun-modifier other than nouns        |   |
|`LB`   | in long bei-construction              | 例子:被, 给 |
|`LC`   | Localizer                             | 定位词, 例子:“里” |
|`M`    | Measure word(including classifiers)   | 量词, 例子:“个” |
|`MSP`  | Some particles                        | 例子:“所” |
|`NN`   | Common nouns                          | 普通名词 |
|`NR`   | Proper nouns                          | 专有名词 |
|`NT`   | Temporal nouns                        | 时序词, 表示时间的名词 |
|`OD`   | Ordinal numbers                       | 序数词, “第一” |
|`ON`   | Onomatopoeia                          | 拟声词, “哈哈” |
|`P`    | Preposition (excluding 把 and 被)       |介词 |
|`PN`   | pronouns                              | 代词 |
|`PU`   | Punctuations                          | 标点 |
|`SB`   | in long bei-construction              | 例子:“被, 给” |
|`SP`   | Sentence-final particle               | 句尾小品词, “吗” |
|`VA`   | Predicative adjective                 | 表语形容词, “红” |
|`VC`   | Copula                                | 系动词, “是” |
|`VE`   | 有 as the main verb                    |“有” |
|`VV`   | Other verbs                           | 其他动词 |



# jieba 分词中的词性标注

## jieba 词性标注基本思路

类似分词流程, `jieba` 的词性标注同样是结合规则和统计的方式, 
具体为在词性标注的过程中, 词典匹配和 HMM 共同作用。

词性标注流程如下:

1. 首先,基于正则表达式进行汉字判断, 正则表达式如下: 

```python
import re

re_han_internal = re.compile("([\u4E00-\u9FD5a-zA-Z0-9+#&\._]+)")
```

2. 若符合上面的正则表达式, 则判定为汉字, 然后基于前缀词典构建有向无环图, 再基于有向无环图计算最大概率路径, 同时在前缀词典中
   找出它所分出的词性, 若在词典中未找到, 则赋予词性为  `x` (代表未知)。当然, 若在这个过程中, 设置使用 HMM, 且待标注词为未登录词, 
   则会通过 HMM 方式进行词性标注。      
3. 若不符合上面的正则表达式, 那么将继续通过正则表达式进行类型判断, 分别赋予 `x`、`m` (数词)、`eng` (英文)

## jieba 词性标注示例

```python
import jieba.posseg as psg

def get_part_of_speech_taging(sentence, HMM = True):
    """
    词性标注
    Params:
        HMM=False: 非 HMM 词性标注
        HMM=True: HMM 词性标注
    """
    segment_list = psg.cut(sentence, HMM)
    tagged_sentence = " ".join([f"{w}/{t}" for w, t in segment_list])
    
    return tagged_sentence


if __name__ == "__main__":
    # data
    sentence = "中文分词是文本处理不可或缺的一步!"
    tagged_sentence = get_part_of_speech_taging(sentence)
    print(tagged_sentence)
```

> `Jieba` 分词支持自定义词典, 其中的词频和词性可以省略。然而需要注意的是, 若在词典中省略词性, 采用 `Jieba` 分词进行词性标注后, 
   最终切分词的词性将变成 `x`, 这在如语法分析或词性统计等场景下会对结果有一定的影响。因此, 在使用 `Jieba` 分词设置自定义词典时, 
   尽量在词典中补充完整的信息.
