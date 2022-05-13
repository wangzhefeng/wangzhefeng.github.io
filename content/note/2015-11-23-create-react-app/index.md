---
title: Create React App
author: 王哲峰
date: '2020-11-23'
slug: create-react-app
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

- [资料](#资料)
- [Create React App 简介](#create-react-app-简介)
- [快速升级 Create React App](#快速升级-create-react-app)
  - [更新 `create-react-app`](#更新-create-react-app)
  - [更新 `react-script`](#更新-react-script)
- [快速开始](#快速开始)
  - [快速创建应用程序](#快速创建应用程序)
- [创建应用程序](#创建应用程序)
  - [选择模板](#选择模板)
  - [选择包管理器](#选择包管理器)
  - [应用程序结构](#应用程序结构)
  - [命令脚本](#命令脚本)
- [辅助工具](#辅助工具)
  - [编辑器设置](#编辑器设置)
  - [独立开发组件](#独立开发组件)
  - [分析捆绑包大小](#分析捆绑包大小)
  - [在开发中使用 HTTPS](#在开发中使用-https)
- [部署到生产环境](#部署到生产环境)
  - [构建](#构建)
  - [部署](#部署)
</p></details><p></p>

# 资料

- https://create-react-app.dev/


# Create React App 简介

Create React App 是一个用于学习 React 的舒适环境，也是用 React 创建新的单页应用的最佳方式。

它会配置你的开发环境，以便使你能够使用最新的 JavaScript 特性，提供良好的开发体验，
并为生产环境优化你的应用程序。你需要在你的机器上安装 Node >= 14.0.0 和 npm >= 5.6

Create React App 不会处理后端逻辑或操纵数据库；它只是创建一个前端构建流水线（build pipeline），
所以你可以使用它来配合任何你想使用的后端。它在内部使用 Babel 和 webpack，但你无需了解它们的任何细节

# 快速升级 Create React App

Create React App 分为两个包:

* `create-react-app`: 用于创建项目的全局命令行程序
* `react-scripts`: 生成项目中的开发依赖项

## 更新 `create-react-app`

当运行 `npx create-react-app my-app` 时，会自动安装最新版本的 Create React App

如果之前通过以下命令全局安装过 `create-react-app`:

```bash
$ npm install -g create-react-app
```

推荐使用以下命令卸载 `create-react-app` 来确保 `npx` 永远保持最新版本:

```bash
$ npm uninstall -g create-react-app
# or
$ yarn global remove create-react-app
```

## 更新 `react-script`

更新构建工具通常是一项艰巨且耗时的任务。当 Create React App 的新版本发布时，
可以使用单个命令进行升级:

```bash
$ npm install react-script@latest
```

# 快速开始

> * npm
> * npx
> * npm 与 npx 的关系
> * nvm

## 快速创建应用程序

1. 无论使用的是 React 还是其他库，Create React App 都可以让开发者专注于代码，
而不是构建工具。要创建一个名为 `my-app` 的应用程序，请运行以下命令:

```bash
$ npx create-react-app my-app
$ cd my-app
$ npm start
```

2. 打开 [http://localhost:3000/](http://localhost:3000/)

# 创建应用程序

## 选择模板

* 模板名字通常是 `cra-template-[template-name]` 的格式

```bash
$ npx create-react-app my-app --template [template-name]
```

## 选择包管理器

* npx(npm)

```bash
$ npx create-react-app my-app
$ cd my-app
$ npm start
```

* npm

```bash
$ npm init react-app my-app
```

* Yarn

```bash
$ yarn create react-app my-app
```

## 应用程序结构

```
my-app
├── README.md
├── node_modules
├── package.json
├── .gitignore
├── public
│   ├── favicon.ico
│   ├── index.html
│   ├── logo192.png
│   ├── logo512.png
│   ├── manifest.json
│   └── robots.txt
└── src
    ├── App.css
    ├── App.js
    ├── App.test.js
    ├── index.css
    ├── index.js
    ├── logo.svg
    ├── serviceWorker.js
    └── setupTests.js
```

* `README.md`: 项目文档
* `node_modules`: 开发环境依赖库目录
* `package.json`: TODO
* `.git`、`.gitignore`: Git 版本仓库文件
* `public/index.html` 是页面模板
* `src/index.js` 是 JavaScript 入口
    - 可以在 `src` 目录里面创建子目录，但是为了更快地构建，webpack 只处理 `src` 中的文件，
      因此需要将 `.js` 和 `.css` 文件放入 `src` 
* 除了 `public` 和 `src`，其他文件可以根据项目情况删除。
  除了 `public` 和 `src` 中的文件，其他在根目录中的目录及文件不会包含在生产环境的版本中

## 命令脚本

以下是在 `my-app` 项目目录下能够使用的命令:

* 启动应用

```bash
$ npm start
# or 
$ yarn start
```

* 运行测试

```bash
$ npm test
# or
$ yarn test
```

* 构建应用
    - 将用于生产的应用程序构建到 `build` 目录中，应用程序已经准备好部署了

```bash
$ npm run build
# or
$ yarn build
```

* TODO

> 这是一个单向操作，一旦 eject，就回不去了

```bash
npm run eject
```

# 辅助工具

## 编辑器设置

## 独立开发组件

## 分析捆绑包大小

Source map explorer 用 source map 来分析 JavaScript 捆绑包，
用来分析代码膨胀的原因

1. 将 Source map explorer 加入 Create React App 项目中

```bash
$ npm install --save source-map-explorer
# or
$ yarn add source-map-explorer
```

2. 在 `package.json` 中加入配置脚本


```json
   "scripts": {
+    "analyze": "source-map-explorer 'build/static/js/*.js'",
     "start": "react-scripts start",
     "build": "react-scripts build",
     "test": "react-scripts test",
```

3. 分析捆绑包大小

```bash
$ npm run build
$ npm run analyze
```

## 在开发中使用 HTTPS

在开发过程中可能需要通过 HTTPS 提供页面，
比如，在使用代理时，代理请求一个 API 服务，
而 API 只提供 HTPPS 协议的服务

* windows(cmd.exe)

```bash
set HTTPS=true&&npm start
```

* windows(Powershell)

```bash
($env:HTTPS = "ture") -and (npm start)
```

* Linux, macOS(Bash)

```bash
HTTPS=true npm start
```

如果要设置自定义 SSL 证书，还需要将环境变量 `SSL_CRT_FILE` 设置为证书路径，
将 `SSL_KEY_FILE` 设置为密钥文件路径

* Linux, macOS(Bash)

```bash
HTTPS=true SSL_CRT_FILE=cert.crt SSL_KEY_FILE=cert.key npm start
```

* 或者在 `npm start` 命令脚本中包含这些设置

```json
{
    "start": "HTTPS=true react-scripts start"
}
```

* 或者创建一个 `.env` 文件并设置 `HTTPS=true`

# 部署到生产环境

## 构建

```bash
$ npm run build
```

## 部署

