---
title: 计算机视觉概览
author: wangzf
date: '2023-04-29'
slug: cv-summary
categories:
  - computer vision
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

- [计算机视觉的三大应用任务](#计算机视觉的三大应用任务)
</p></details><p></p>

# 计算机视觉的三大应用任务

![img](images/computer_visual_task.png)

* 图像分类
    - 分类：回答一张图像中是什么的问题
* 目标检测
    - 分类 + 定位：不仅需要回答图像中有什么, 而且还得给出这些物体在图像中的位置
    - 应用
        - 无人驾驶
        - 工业产品瑕疵检测
        - 医学肺部节点检测
* 图像分割
    - 像素级的图像分割
        - 语义分割
        - 实例分割
