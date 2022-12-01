---
title: RNN--语音识别
author: 王哲峰
date: '2022-07-15'
slug: dl-rnn-app-audio-recognition
categories:
  - deeplearning
tags:
  - model
---

# 语音数据的表示方式

语音通常是由音频信号构成的, 而音频信号本身又是以声波的形式传递的, 
一段语音的波形通常是一种时序状态, 也就是说音频信号是按照时间顺序播放的。

通过一些预处理和转换技术, 可以将声波转换为更小的声音单元, 即音频块. 
所以在语音识别的深度学习模型中, 输入就是原始的语音片段经过预处理之后的一个个音频块, 
这样的音频块是以序列形式存在的。

语音识别的输入是一个序列, 输出通常是以一段文字的形式呈现, 
这段文字也是按顺序排列的文本序列, 所以语音识别的输出也是一个序列。
说白了, 语音识别热舞就是建立序列输入到序列输出之间的有监督机器学习模型.
