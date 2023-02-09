---
title: NLP-opencc
author: 王哲峰
date: '2022-04-05'
slug: nlp-opencc
categories:
  - nlp
tags:
  - tool
---



- 介绍
    - Open Chinese Convert(OpenCC，开放中文转换)是一个开源项目，用于在繁体中文，简体中文和日文汉字(Shinjitai)之间进行转换.       
        - 支持中国大陆，台湾和香港之间的字符级和短语级转换，字符变体转换以及区域习语. 不是普通话和广东话等之间的翻译工具. 
        - 支持词汇等级的转换，异体字转换和地区习惯用词转换(中国大陆，台湾，香港，日本新字体). 不提供普通话与粤语的转换. 
- 特点
    - 严格区分“一简对多繁”和“一简对多异”
    - 完全兼容异体字，可以实现动态替换  
    - 严格审校一简对多繁词条，原则为“能分则不合”
    - 支持中国大陆，台湾，香港异体字和地区习惯用词转换，如「里」「里」，「鼠标」「滑鼠」
    - 词库和函数库完全分离，可以自由修改，引入，扩展
- 在线演示 Demo(不支持API查询)
    - https://opencc.byvoid.com/


# 1.Python opencc 安装


```bash
# Windows, Linux, Mac
workon tf
pip install opencc
```

# 2.opencc 使用


## 2.1 基本配置文件


=========== ================================================================================================================ ===============================================
文件名       from => to                                                                                                         翻译
=========== ================================================================================================================ ===============================================
s2t.json    Simplified Chinese                           to     Traditional Chinese                                            简体到繁体
t2s.json    Traditional Chinese                          to     Simplified Chinese                                             繁体到简体
s2tw.json   Simplified Chinese                           to     Traditional Chinese(Taiwan Standard)                           简体到台湾正体
tw2s.json   Traditional Chinese (Taiwan Standard)        to     Simplified Chinese                                             台湾正体到简体
s2hk.json   Simplified Chinese                           to     Traditional Chinese(Hong Kong Standard)                        简体到香港繁体(香港小学学习字词表标准)
hk2s.json   Traditional Chinese (Hong Kong Standard)     to     Simplified Chinese                                             香港繁体(香港小学学习字词表标准)到简体
s2twp.json  Simplified Chinese                           to     Traditional Chinese(Taiwan Standard) with Taiwanese idiom      简体到繁体(台湾正体标准)并转换为台湾常用词汇
tw2sp.json  Traditional Chinese (Taiwan Standard)        to     Simplified Chinese with Mainland Chinese idiom                 繁体(台湾正体标准)到简体并转换为中国大陆常用词汇
t2tw.json   Traditional Chinese (OpenCC Standard)        to     Taiwan Standard                                                繁体(OpenCC标准)到台湾正体
t2hk.json   Traditional Chinese (OpenCC Standard)        to     Hong Kong Standard                                             繁体(OpenCC标准)到香港繁体(香港小学学习字词表标准)
t2jp.json   Traditional Chinese Characters (Kyūjitai)    to     New Japanese Kanji(Shinjitai)                                  繁体(OpenCC标准，旧字体)到日文新字体
jp2t.json   New Japanese Kanji (Shinjitai)               to     Traditional Chinese Characters(Kyūjitai)                       日文新字体到繁体(OpenCC标准，旧字体)
=========== ================================================================================================================ ===============================================

## 2.2 Python API Demo


```python
import opencc

# 简体中文 => 繁体中文
converter = opencc.OpenCC("t2s.json")
t_data = "漢字"
s_data = converter.convert(t_data)
```

## 2.3 命令行模式 Demo


```bash
opencc --help
opencc_dict --help
opencc_phrase_extract --help
```
