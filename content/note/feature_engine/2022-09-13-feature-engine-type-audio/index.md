---
title: Audio
author: 王哲峰
date: '2022-09-13'
slug: feature-engine-type-audio
categories:
  - feature engine
tags:
  - ml
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

- [音频数据了解](#音频数据了解)
  - [音频文件格式介绍](#音频文件格式介绍)
  - [音频文件格式的种类](#音频文件格式的种类)
  - [参考文献](#参考文献)
- [音频数据读写](#音频数据读写)
  - [Python wave](#python-wave)
    - [wave 函数和异常](#wave-函数和异常)
    - [Wave_read 对象](#wave_read-对象)
  - [Wave_write 对象](#wave_write-对象)
    - [wave 使用示例](#wave-使用示例)
</p></details><p></p>

# 音频数据了解

## 音频文件格式介绍

- 音频文件格式专指存放音频数据的文件的格式, 存在多种不同的格式
- 一般获取音频数据的方法是:
    - 采用固定的时间间隔, 对音频电压采样(量化), 并将结果以某种分辨率(例如: CDDA 每个采样为 16 比特或 2 字节)存储
- 采样的时间间隔可以有不同的标准, 如: 
    - CDDA 采用每秒 44100 次
    - DVD 采用每秒 48000 或 96000 次
- 音频文件格式的关键参数:
    - **采样率**
    - **分辨率**
    - **声道数目(例如立体声为2声道**
- 音频文件和编解码器不同
    - 尽管一种音频文件格式可以支持多种编码, 例如 AVI 文件格式, 但多数的音频文件仅支持一种音频编码

## 音频文件格式的种类

- 无损格式:
    - WAV, FLAC, APE, ALAC, WavPack(WV)
    - 无损的音频格式(例如FLAC)压缩比大约是 2: 1, 解压时不会产生数据/质量上的损失, 
      解压产生的数据与未压缩的数据完全相同. 如需要保证音乐的原始质量, 应当选择无损音频编解码器. 
      例如, 用免费的 FLAC 无损音频编解码器你可以在一张 DVD-R 碟上存储相当于 20 张 CD 的音乐
- 有损格式
    - MP3, AAC, Ogg, Vorbis, Opus
    - 有损文件格式是基于声学心理学的模型, 除去人类很难或根本听不到的声音, 
      例如: 一个音量很高的声音后面紧跟着一个音量很低的声音. MP3就属于这一类文件
    - 有损压缩应用很多, 但在专业领域使用不多. 有损压缩具有很大的压缩比, 提供相对不错的声音质量

## 参考文献

- [维基百科(音频文件格式)](https://zh.wikipedia.org/wiki/%E9%9F%B3%E9%A2%91%E6%96%87%E4%BB%B6%E6%A0%BC%E5%BC%8F)
- [维基百科(WAV格式)](https://zh.wikipedia.org/wiki/WAV)

# 音频数据读写

## Python wave

- Python 的 `wave` 库用来读写 WAV 格式文件
- `wave` 模块提供了一个处理 WAV 声音格式的便利接口        
    - 不支持压缩、解压缩
    - 支持单声道、立体声

### wave 函数和异常

- `wave.open(file, mode = None)`
    - file: 文件名、文件类对象
    - mode: 
        - `'rb'`: 只读模式, 返回一个 `Wave_read` 对象
        - `'wb'`: 只写模式, 返回一个 `Wave_write` 对象
- **exception** `wave.Error`
    - 当不符合 WAV 格式或无法操作时引发的错误

### Wave_read 对象

**Wave_read** 对象的常用方法: 

- `Wave_read.close()`
- `Wave_read.getnchannels()`: 声道数量, 1: 单声道, 2: 立体声
- `Wave_read.getsampwidth()`: 采样字节长度
- `Wave_read.getframerate()`: 采样频率
- `Wave_read.getnframes()`: 音频总帧数
- `Wave_read.getcomptype()`: 压缩类型(只支持 'NONE' 类型)
- `Wave_read.getcompname()`: getcomptype() 的通俗版本. 使用 'not compressed' 代替 'NONE'
- `Wave_read.getparams()`: 返回一个 namedtuple() (nchannels, sampwidth, framerate, nframes, comptype, compname), 与 get*() 方法的输出相同
- `Wave_read.readframes(n)`: 读取并返回以 bytes 对象表示的最多 n 帧音频
- `Wave_read.rewind()`: 设置当前文件指针位置

- 下面两个方法是为了和 aifc 保持兼容, 实际不做任何事情
    - `Wave_read.getmarkers()`: 返回 None
    - `Wave_read.getmark()`: 引发错误异常
- 下面两个方法都使用指针, 具体实现由其底层决定
    - `Wave_read.setpos(pos)`: 设置文件指针到指定位置
    - `Wave_read.tell()`: 当前文件指针位置

## Wave_write 对象

对于可查找的输出流, wave 头将自动更新以反映实际写入的帧数. 对于不可查找的流, 当写入第一帧时 nframes 值必须准确. 
获取准确的 nframes 值可以通过调用 setnframes() 或 setparams() 并附带 close() 被调用之前将要写入的帧数, 
然后使用 writeframesraw() 来写入帧数据, 或者通过调用 writeframes() 并附带所有要写入的帧.  
在后一种情况下 writeframes() 将计算数据中的帧数并在写入帧数据之前相应地设置 nframes. 

**Wave_write** 对象的常用方法: 

- `Wave_write.close()`: 确保 nframes 是正确的, 并在文件被 wave 打开时关闭它. 
  此方法会在对象收集时被调用.  如果输出流不可查找且 nframes 与实际写入的帧数不匹配时引发异常. 
- `Wave_write.setnchannels(n)`: 设置声道数
- `Wave_write.setsampwidth(n)`: 设置采样字节长度为 n
- `Wave_write.setframerate(n)`: 设置采样频率为 n. 
- `Wave_write.setnframes(n)`: 设置总帧数为 n. 
  如果与之后实际写入的帧数不一致此值将会被更改(如果输出流不可查找则此更改尝试将引发错误)
- `Wave_write.setcomptype(type, name)`: 设置压缩格式. 目前只支持 NONE 即无压缩格式
- `Wave_write.setparams(tuple)`: tuple 应该是 (nchannels, sampwidth, framerate, nframes, comptype, compname), 
  每项的值应可用于 set*() 方法. 设置所有形参
- `Wave_write.tell()`: 返回当前文件指针, 其指针含义和` Wave_read.tell() 以及 Wave_read.setpos() 是一致的
- `Wave_write.writeframesraw(data)`: 写入音频数据但不更新 nframes, 可接受任意 bytes-like object. 
- `Wave_write.writeframes(data)`: 写入音频帧并确保 nframes 是正确的. 
  如果输出流不可查找且在 data 被写入之后写入的总帧数与之前设定的 
  has been written does not match the previously set value for nframes 值不匹配将会引发错误

***
**Note:**

* 注意在调用 `writeframes()` 或 `writeframesraw()` 之后再设置任何格式参数是无效的, 
  而且任何这样的尝试将引发 wave.Error. 
***

### wave 使用示例

