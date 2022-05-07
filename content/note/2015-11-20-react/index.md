---
title: React
author: 王哲峰
date: '2015-11-20'
slug: react
categories:
  - javascript
  - react
tags:
  - web
---

<style>
h1 {
  background-color: #2B90B6;
  background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
  background-size: 100%;
  -webkit-background-clip: text;
  -moz-background-clip: text;
  -webkit-text-fill-color: transparent;
  -moz-text-fill-color: transparent;
}
h2 {
  background-color: #2B90B6;
  background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
  background-size: 100%;
  -webkit-background-clip: text;
  -moz-background-clip: text;
  -webkit-text-fill-color: transparent;
  -moz-text-fill-color: transparent;
}

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

- [开发能力](#开发能力)
- [React](#react)
  - [先决条件](#先决条件)
  - [目标](#目标)
  - [官网](#官网)
  - [相关 js 库](#相关-js-库)
  - [创建虚拟DOM的两种方式](#创建虚拟dom的两种方式)
  - [JSX](#jsx)
</p></details><p></p>

# 开发能力

* 脚手架与项目脚本
* 测试体系
* 监控体系
* 项目规范(代码规范、版本规范)
* Git 管理
* 项目部署
* 项目集成
* 项目构建打包
* 性能优化
    - 长列表优化
    - 加载优化
    - 项目的可维护性
    - 首屏加载速度
* 技术
    - 服务器端渲染
    - bff 技术
    - 微前端

* React 基础
* React-Router
* PubSub
* Redux
* Ant-Design

# React

## 先决条件

* 熟悉 HTML、CSS、JavaScript 编程
* 熟悉基本的 DOM 知识
* 熟悉 ES6 语法和特性
* Node.js 和 npm 全局安装

## 目标

* 了解基本的 React 概念和相关术语，例如 Babel、Webpack、JSX、组件(components)、
  props、状态(state)和生命周期(lifecycle)
* 构建一个非常简单的 React 应用程序来演示上述概念

这是最终结果的源代码和现场演示:

* [在 GitHub 上查看源代码](https://github.com/wangzhefeng/react_tutorial)
* [查看演示]()






## 官网

- https://zh-hans.reactjs.org/

## 相关 js 库

- ``babel.min.js``
    - ES6 => ES5
    - jsx => js
- ``prop-type.js``
- ``react.development.js``
    - 核心库
- ``react-dom.development.js``
    - 扩展库，操作 DOM

## 创建虚拟DOM的两种方式

- 1.纯 js 方式(一般不用)
- 2.jsx 方式

## JSX 

- 1.全称 JavaScript XML
- 2.React 定义的一种类似于 XML 的 js 扩展语法：JS+XML
- 3.本质是 React.createElement(component, props, ...children)方法的语法糖
- 4.作用：用来简化创建虚拟 DOM
    - a.写法：``val ele = <h1>Hello JSX!</h1>``
    - b.注意1：它不是字符串，也不是 HTML/XML 标签
    - c.注意2：它最终产生的就是一个 js 对象
- 5.标签名任意：HTML 标签或其他标签





