---
title: 创建 React 应用程序
author: 王哲峰
date: '2015-11-20'
slug: react-doc1
categories:
  - react
  - javascript
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
img {
    pointer-events: none;
}
</style>


<details><summary>目录</summary><p>

- [推荐的 React 工具链](#推荐的-react-工具链)
- [从头打造工具链](#从头打造工具链)
- [在网站中添加 React](#在网站中添加-react)
- [创建新的 React 应用](#创建新的-react-应用)
- [CDN 链接](#cdn-链接)
  - [开发环境](#开发环境)
  - [生产环境](#生产环境)
  - [为什么要使用 `crossorigin`属性](#为什么要使用-crossorigin属性)
- [JSX](#jsx)
  - [为什么使用 JSX ?](#为什么使用-jsx-)
  - [在 JSX 中嵌入表达式](#在-jsx-中嵌入表达式)
  - [JSX 也是一个表达式](#jsx-也是一个表达式)
  - [JSX 特定属性](#jsx-特定属性)
  - [使用 JSX 指定子元素](#使用-jsx-指定子元素)
  - [JSX 防止注入攻击](#jsx-防止注入攻击)
  - [JSX 表示对象](#jsx-表示对象)
- [React--组件 & Props](#react--组件--props)
  - [函数组件与 class 组件](#函数组件与-class-组件)
  - [渲染组件](#渲染组件)
  - [组合组件](#组合组件)
  - [提取组件](#提取组件)
  - [Props 的只读性](#props-的只读性)
- [事件处理](#事件处理)
  - [事件处理](#事件处理-1)
  - [向事件处理程序传递参数](#向事件处理程序传递参数)
- [React--State & 生命周期](#react--state--生命周期)
  - [将函数组件转换成 class 组件](#将函数组件转换成-class-组件)
    - [时钟示例](#时钟示例)
    - [将函数组件转换成 class 组件](#将函数组件转换成-class-组件-1)
  - [向 class 组件中添加局部的 state](#向-class-组件中添加局部的-state)
- [将生命周期方法添加到 class 中](#将生命周期方法添加到-class-中)
  - [正确地使用 State](#正确地使用-state)
    - [不要直接修改 State](#不要直接修改-state)
    - [State 的更新可能是异步的](#state-的更新可能是异步的)
    - [State 的更新会被合并](#state-的更新会被合并)
  - [数据是向下流动的](#数据是向下流动的)
- [React--元素渲染](#react--元素渲染)
  - [将一个元素渲染为 DOM](#将一个元素渲染为-dom)
  - [更新已渲染的元素](#更新已渲染的元素)
  - [React 只更新它需要更新的部分](#react-只更新它需要更新的部分)
- [React--示例教程](#react--示例教程)
  - [课前准备](#课前准备)
  - [环境准备](#环境准备)
    - [在浏览器中编写代码](#在浏览器中编写代码)
    - [搭建本地开发环境](#搭建本地开发环境)
  - [概览](#概览)
    - [React 是什么?](#react-是什么)
  - [阅读初始代码](#阅读初始代码)
    - [通过 Props 传递数据](#通过-props-传递数据)
    - [给组件添加交互功能](#给组件添加交互功能)
    - [开发者工具](#开发者工具)
  - [游戏完善](#游戏完善)
  - [时间旅行](#时间旅行)
</p></details><p></p>


# 推荐的 React 工具链

* 如果你是在学习 React 或创建一个新的单页应用，请使用[Create React App](https://create-react-app.dev/)
* 如果你是在用 Node.js 构建服务端渲染的网站，试试 [Next.js](https://nextjs.org/)
* 如果你是在构建内容主导的静态网站，试试 [Gatsby](https://www.gatsbyjs.com/)
* 如果你是在打造组件库或将 React 集成到现有代码仓库，尝试更灵活的工具链
    - Neutrino 把 webpack 的强大功能和简单预设结合在一起。并且包括了 React 应用和 React 组件的预设
    - Nx 是针对全栈 monorepo 的开发工具包，其内置了 React，Next.js，Express 等
    - Parcel 是一个快速的、零配置的网页应用打包器，并且可以搭配 React 一起工作
    - Razzle 是一个无需配置的服务端渲染框架，但它提供了比 Next.js 更多的灵活性

# 从头打造工具链

https://medium.com/@JedaiSaboteur/creating-a-react-app-from-scratch-f3c693b84658

一组 JavaScript 构建工具链通常由这些组成：

* 一个 package 管理器，比如 Yarn 或 npm。它能让你充分利用庞大的第三方 package 的生态系统，并且轻松地安装或更新它们
* 一个打包器，比如 webpack 或 Parcel。它能让你编写模块化代码，并将它们组合在一起成为小的 package，以优化加载时间
* 一个编译器，例如 Babel。它能让你编写的新版本 JavaScript 代码，在旧版浏览器中依然能够工作


# 在网站中添加 React



# 创建新的 React 应用


# CDN 链接

如果需要加载指定版本的 react 和 react-dom，可以把 17 替换成所需加载的版本号

## 开发环境

```html
<script crossorigin src="https://unpkg.com/react@17/umd/react.development.js"></script>
<script crossorigin src="https://unpkg.com/react-dom@17/umd/react-dom.development.js"></script>
```

## 生产环境

```html
<script crossorigin src="https://unpkg.com/react@17/umd/react.production.min.js"></script>
<script crossorigin src="https://unpkg.com/react-dom@17/umd/react-dom.production.min.js"></script>
```

## 为什么要使用 `crossorigin`属性

如果通过 CDN 的方式引入 React

1. 建议设置 `crossorigin` 属性:

```html
<script crossorigin src="..."></script>
```

2. 同时建议验证使用的 CDN 是否设置了 `Access-Control-Allow-Origin: *` HTTP 请求头，
   这样能在 React 16 及以上的版本中有更好的错误处理体验 

![img](images/cdn-cors-header.png)


# JSX


```js
// 注意：这是简化过的结构
const element = {
    type: 'h1',
    props: {
    className: 'greeting',
    children: 'Hello, world!'
    }
}
const element = <h1>Hello, world!</h1>;
```

- 这个有趣的标签语法既不是字符串也不是 HTML
- 它被称为 JSX，是一个 JavaScript 的语法扩展
- 建议在 React 中配合使用 JSX
- JSX 可以很好地描述 UI 应该呈现出它应有交互的本质形式
- JSX 可能会使人联想到模板语言，但它具有 JavaScript 的全部功能
- JSX 可以生成 React 元素


## 为什么使用 JSX ?

- React 认为渲染逻辑本质上与其他 UI 逻辑内在耦合
- React 并没有采用将标记与逻辑进行分离到不同文件这种人为地分离方式，
  而是通过将二者共同存放在被称之为组件的松散耦合单元中，来实现关注点分离
- React 不强制要求使用 JSX，但是大多数人发现，在 JavaScript 代码中将 JSX 和 UI 放在一起时，
  会在视觉上有辅助作用，它还可以使 React 显示更多有用的错误和警告消息

## 在 JSX 中嵌入表达式

在 JSX 语法中，可以在大括号内放置任何有效的 JavaScript 表达式

- 示例 1

```js
const name = "Josh Perez";
const element = <h1>Hello, {name}</h1>;

ReactDOM.render(
element,
document.getElementById('root')
);
```

- 示例 2

```js
function formatName(user) {
   return user.firstName + " " + user.lastName;
}
const user = {
    firstName: "Harper",
    lastName: "Perez"
};

const element = (
    <h1>
        Hello, {formatName(user)}!
    </h1>
);

ReactDOM.render(
    element,
    document.getElementById("root")
);  
```

- 为了便于阅读, 建议将 JSX 拆分为多行
- 为了避免遇到自动插入分号陷阱，建议将内容包裹在括号中

## JSX 也是一个表达式

- 在编译之后，JSX 表达式也会被转为普通 JavaScript 函数调用，并且对其取值后得到 JavaScript对象，也就是说，
  可以在 if 语句和 for 循环的代码块中使用 JSX，将 JSX 赋值给变量，把 JSX 当做参数传入，以及从函数中返回 JSX

```js
function getGreeting(user) {
    if (user) {
        return <h1>Hello, {formatName(user)}!<!h1>
    }
    return <h1>Hello, Stranger.</h1>
}
```

## JSX 特定属性

- 可以通过使用引号，来将属性值指定为字符串字面量，也可以使用大括号，来在属性值中插入一个 JavaScript 表达式

```js
const element = <div tabIndex="0"></div>;
const element = <img src={user.avatarUrl}></img>;
```

## 使用 JSX 指定子元素

- 假如一个标签里面没有内容，可以使用 ``/>`` 来闭合标签，就像 XML 语法一样
- JSX 标签里能够包含很多子元素

```js
const element = <img src={user.avatarUrl} />;

const element = (
    <div>
        <h1>Hello!</h1>
        <h2>Good to see you here.</h2>
    </div>
);
```

## JSX 防止注入攻击

- 可以安全地在 JSX 当中插入用户输入内容
- React DOM 在渲染所有输入内容之前，默认会进行转义，它可以确保在你的应用中，
    永远不会注入哪些并非自己明确编写的内容，所有的内容在渲染之前都被转换成了字符串，
    这样可以有效地防止 XSS(cross-site-scripting,跨站脚本)攻击

```js
const title = reponse.potentiallyMaliciousInput;
// 直接使用时安全的
const element = <h1>{title}</h1>
```

## JSX 表示对象

- Babel 会把 JSX 转译成 `React.createElement()` 函数调用

```js
const element = (
    <h1 className="greeting">
        Hello, world!
    </h1>
);
```

```js
const element = React.createElement(
    'h1',
    {className: 'greeting'},
    'Hello, world!'
);
```

- `React.createElement()` 会预先执行一些检查，以帮助你编写无措代码，
  但实际上它创建了一个这样的对象，这些对象被称为 "React 元素"，
  它们描述了你希望在屏幕上看到的内容。React 通过读取这些对象，
  然后使用它们来构建 DOM 以及保持随时更新

```js
// 注意：这是简化过的结构
const element = {
    type: 'h1',
    props: {
        className: 'greeting',
        children: 'Hello, world!'
    }
};
```



# React--组件 & Props

- 组件允许你将 UI 拆分为独立可复用的代码片段，并对每个片段进行独立构思
- 组件，从概念上类似于 JavaScript 函数。它接受任意的入参(即"props")，
  并返回用于描述页面展示内容的 React 元素

## 函数组件与 class 组件

- 定义组件最简单的方式就是编写 JavaScript 函数

```js
function Welcome(props) {
  return <h1>Hello, {props.name}</h1>;
}
```

- 定义组件还可以使用 ES6 的 class

```js
class Welcome extends React.Component {
  render() {
     return <h1>Hello, {this.props.name}</h1>;
  }
}
```

## 渲染组件

- React 元素可以是 DOM 标签，也可以是用户自定义的组件

```js
// DOM 标签
const element = <div />;

// 用户自定义组件
const element = <Welcome name="Sara" />;
```

- 当 React 元素为用户自定义组件时，它会将 JSX 所接收的属性(attributes)以及子组件(children)转换为单个对象传递给组件，这个对象被称之为 "props".

```js
// Welcome 组件
function Welcome(props) {
   return <h1>Hello, {props.name}</h1>;
}

// React 元素用户自定义的组件
const element = <Welcome name="Sara" />;

// 元素渲染
ReactDOM.render(
   element,
   document.getElementById('root')
);
```

- (1)我们调用 `ReactDOM.render()` 函数，并传入 `<Welcome name="Sara" />` 作为参数
- (2)React 调用 Welcome 组件，并将 `{name: 'Sara'}` 作为 props 传入
- (2)Welcome 组件将 `<h1>Hello, Sara</h1>` 元素作为返回值
- (3)React DOM 将 DOM 高效地更新为 `<h1>Hello, Sara</h1>`

> - 组件名称必须以大写字母开头
> - React 会将以小写字母开头的组件视为原生 DOM 标签

## 组合组件

- 组件可以在其输出中应用其他组件，这就可以让我们用同一组件来抽象出任意层次的细节
- 在 React 应用程序中，按钮、表单、对话框，甚至整个屏幕的内容通常都会以组件的形式表示
- 通常来说，每个新的 React 应用程序的顶层组件都是 App 组件。
  但是，如果你将 React 集成到现有的应用程序中，你可能需要使用像 Button 这样的小组件，
  并自下而上地将这类组件逐步应用到视图层的每一处

```js
// 可以创建一个可以多次渲染 Welcome 组件的 App 组件
function Welcome(props) {
   return <h1>Hello, {props.name}</h1>;
}

function App() {
   return (
      <Welcome name="Sara" />
      <Welcome name="Cahal" />
      <Welcome name="Edite" />
   );
}

ReactDOM.render(
   <App />,
   document.getElementById("root")
);
```

## 提取组件

- 将组件拆分为更小的组件
- 最初看上去，提取组件可能是一件繁重的工作，但是，在大型应用中，构建可复用组件库是完全值得的。
根据经验来看，如果 UI 中有一部分被多次使用（`Button`，`Panel`，`Avatar`），
或者组件本身就足够复杂（`App`，`FeedStory`，`Comment`），那么它就是一个可提取出独立组件的候选项

```js
// 该组件用于描述一个社交媒体网站上的评论功能，它接收 author（对象），text （字符串）以及 date（日期）作为 props
function Comment(props) {
   return (
      <div className="Comment">
         <div className="UserInfo">
            <img className="Avatar" 
               src={props.author.avatarUrl}
               alt={props.author.name}
            />
            <div className="UserInfo-name">
               {props.author.name}
            </div>
         </div>
         <div className="Comment-text">
            {props.text}
         </div>
         <div className="Comment-date">
            {formatDate(props.date)}
         </div>
      </div>
   );
}
// --------------------------------------------------------------------------------
// 上面的组件由于嵌套的关系，变得难以维护，且很难复用它的各个部分。因此，让我们从中提取一些组件出来
// --------------------------------------------------------------------------------
// 1.提取 Avatar 组件
function Avatar(props) {
   return (
      <img className="Avatar" 
         src={props.user.avatarUrl}
         alt={props.user.name}
      />
   );
}

// 2.提取 UserInfo 组件
function UserInfo(props) {
   return (
      <div className="UserInfo">
         <Avatar user={props.user} />
         <div className="UserInfo-name">
            {props.user.name}
         </div>
      </div>
   );
}
// 3.简化后的 Comment 组件
function Comment(props) {
   return (
      <div className="Comment">
         <UserInfo user={props.author} />
         <div className="Comment-text">
            {props.text}
         </div>
         <div className="Comment-date">
            {formatDate(props.date)}
         </div>
      </div>
   );
}
```

## Props 的只读性

- 组件无论是使用函数声明还是通过 class 声明，都绝不能修改自身的 props
- React 非常灵活，但它有一个严格的规则：**所有 React 组件都必须像纯函数一样保护它们的 props 不被更改**

```js
// 纯函数：部署尝试更改入参，且多次调用下相同的入参始终返回相同的结果
function sum(a, b) {
   return a + b;
}

// 不是纯函数，因为它更改了自己的入参
function withdraw(account, amount) {
   account.total -= amount;
}
```

当然，应用程序的 UI 是动态的，并会伴随着时间的推移而变化。有一种新的概念，称之为 “state”。
在不违反上述规则的情况下，state 允许 React 组件随用户操作、网络响应或者其他变化而动态更改输出内容


# 事件处理

## 事件处理

- React 元素的事件处理和 DOM 元素的很相似，但是有一点语法上的不同
    - (1)React 事件的命名采用小驼峰(camelCase)，而不是纯小写
    - (2)使用 JSX 语法时你需要传入一个函数作为事件处理函数，而不是一个字符串


## 向事件处理程序传递参数


# React--State & 生命周期

- State 与 props 类似，但是 state 是私有的，并且完全受控于当前组件

## 将函数组件转换成 class 组件

### 时钟示例

- 一种更新 UI 界面的方法：通过调用 ReactDOM.render() 来修改想要渲染的元素

```js
function tick() {
   const element = (
      <div>
         <h1>Hello, world!</h1>
         <h2>It is {new Date().toLocaleTimeString()}.</h2>
      </div>
   );
   ReactDOM.render(
      element,
      document.getElementById('root')
   );
}

setInterval(tick, 1000);
```

- 封装可复用的 `Clock` 组件，它将设置自己的计时器并每秒更新一次

```js
function Clock(props) {
   return (
      <div>
         <h1>Hello, world!</h1>
         <h2>It is {props.date.toLocaleTimeString()}.</h2>
      </div>
   );
}

function tick() {
   ReactDOM.render(
      <Clock date={new Date()} />,
      document.getElementById('root')
   );
}

setInterval(tick, 1000);
```

- 理想情况下，希望只编写一次代码，便可以让 `Clock` 组件自我更新，需要在 `Clock` 组件中添加 "state" 来实现这个功能

```js
ReactDOM.render(
   <Clock />,
   document.getElementById("root")
);
```

### 将函数组件转换成 class 组件

通过以下五步将 `Clock` 的函数组件转换成 class 组件

- 1.创建一个同名的 ES6 class, 并且继承于 `React.Component`
- 2.添加一个空的 `render()` 方法
- 3.将函数体移动到 `render()` 方法之中
- 4.在 `render()` 方法中使用 `this.props` 替换 `props`
- 5.删除剩余的空函数声明，现在 `Clock` 组件被定义为 class，而不是函数

```js
class Clock extends React.Component {
   render() {
      return (
         <div>
            <h1>Hello, world!</h1>
            <h2>It is {this.props.date.toLocaleTimeString()}.</h2>
         </div>
      );
   };
}
```

> - 每次组件更新时 `render` 方法都会被调用，但只要在相同的 DOM 节点中渲染 `<Clock />`，
就仅有一个 `Clock` 组件的 class 实例被创建使用。
这就使得我们可以使用如 state 或生命周期方法等很多其他特性

## 向 class 组件中添加局部的 state

通过以下三步将 `date` 从 `props` 移动到 `state` 中

- 1.把 `render()` 方法中的 `this.props.date` 替换成 `this.state.date`

```js
class Clock extends React.Component {
   render() {
      return (
         <div>
            <h1>Hello, world!</h1>
            <h2>It is {this.state.date.toLocaleTimeString()}.</h2>
         </div>
      );
   }
}
```

- 2.添加一个 class 构造函数，然后在该函数中为 `this.state` 赋初值

```js
class Clock extends React.Component {
   // 通过这种方式将 props 传递到父类(React.Component)的构造函数中
   constructor(props) {
      super(props);
      this.state = {date: new Date()};
   }

   render() {
      return (
         <div>
            <h1>Hello, world!</h1>
            <h2>It is {this.state.date.toLocaleTimeString()}.</h2>
         </div>
      );
   }
}
```

> - class 组件应该始终使用 props 参数来调用父类的构造函数

- 3.移除 `<Clock />` 元素中的 date 属性

```js
ReactDOM.render(
   <Clock />,
   document.getElementById('root')
);
```

- 4.完成

```js
class Clock extends React.Component {
   // 通过这种方式将 props 传递到父类(React.Component)的构造函数中
   constructor(props) {
      super(props);
      this.state = {date: new Date()};
   }

   render() {
      return (
         <div>
            <h1>Hello, world!</h1>
            <h2>It is {this.state.date.toLocaleTimeString()}.</h2>
         </div>
      );
   }
}

ReactDOM.render(
   <Clock />,
   document.getElementById('root')
);
```

# 将生命周期方法添加到 class 中

- 在具有许多组件的应用程序中，当组件被销毁时释放所占用的资源是非常重要的
    - 当 `Clock` 组件第一次被渲染到 DOM 中的时候，就为其设置一个计时器。这在 React 中被称为 "挂载(mount)"
    - 同时，当 DOM 中 `Clock` 组件被删除的时候，应该清除计时器。这在 React 中被称为 "卸载(unmount)"

- 1.可以为 class 组件声明一些特殊的方法，当组件挂载或卸载时就会去执行这些方法，这些方法叫做"声明周期方法"

```js
class Clock extends React.Component {
  constructor(props) {
     super(props);
     this.state = {date: new Date()};
  }

  componentDidMount() {
  }

  componentWillUnmount() {
  }

  render() {
     return {
        <div>
           <h1>Hello, world!</h1>
           <h2>It is {this.state.date.toLocaleTimeString()}.</h2>
        </div>
     }
  }
}
```

- 2.`componentDidMount()` 方法会在组件已经被渲染到 DOM 中后运行，所以，最好在这里设置计时器

```js

componentDidMount() {
  this.timerID = setInterval(
     () => this.tick(),
     1000
  );
}
```

## 正确地使用 State


### 不要直接修改 State

### State 的更新可能是异步的

### State 的更新会被合并


## 数据是向下流动的



# React--元素渲染

```js
const element = <h1>Hello, world</h1>;
```

- 元素(Element)是构成 React 应用的最小砖块，组件()是由元素构成的
- 与浏览器的 DOM 元素不同，React 元素是创建开销绩效的普通对象
- React DOM 会负责更新 DOM 来与 React 元素保持一致

## 将一个元素渲染为 DOM

- root DOM 节点

```html
<div id="root"></div>
```

- root DOM 节点内的所有内容都由 React DOM 管理
- 仅使用 React 构建的应用通常只有单一的 root DOM 节点，如果将 React 集成进一个已有应用，那么可以在应用中包含任意多的独立 root DOM 节点

- 如果想要将一个 React 元素渲染到 root DOM 节点中，只需要把它们一起传入 `ReactDOM.render()`

```js
const element = <h1>Hello, world</h1>;
ReactDOM.render(
   element,
   document.getElementById('root')
);
```

## 更新已渲染的元素

- React 元素是不可变对象，一旦被创建，就无法更改它的子元素或者属性。一个 React 元素就像电影的单帧：它代表了某个特定时刻的 UI
- 更新 UI 唯一的方式是创建一个全新的元素，并将其传入 `ReactDOM.render()`
- 在实践中，大多数 React 应用只会调用一次 ReactDOM.render()

```js

// 在 setInterval() 回调函数，每秒都调用 ReactDOM.render()
function tick() {
   const element = (
      <div>
         <h1>Hello, world!</h1>
         <h2>It is {new Date().toLocaleTimeString()}.</h2>
      </div>
   );

   ReactDOM.render(
      element,
      document.getElementById('root')
   );
}

setInterval(tick, 1000);
```

## React 只更新它需要更新的部分

- React DOM 会将元素和它的子元素与它们之间的状态进行比较，
  并只会进行必要的更新来使 DOM 达到预期的状态
- 应该专注于 UI 在任意给定时刻的状态，而不是一视同仁地随着时间修改整个界面



# React--示例教程

## 课前准备

- 教程分为如下几部分：

   - 环境准备是学习该教程的起点
   - 概览介绍了 React 的基础知识、组件、props 和 state
   - 游戏完善介绍了在 React 开发过程中最常用的技术
   - 时间旅行可以让你更加深刻地了解 React 的独特优势

- 我们会做出什么东西？

   - 用 React 开发一个井字棋(tic-tac-toe)
   - https://codepen.io/gaearon/pen/gWWZgR

- 前置知识

   - HTML
   - CSS
   - JavaScript

## 环境准备

### 在浏览器中编写代码

- 首先，在新的浏览器选项卡中打开这个 `初始模板 <https://codepen.io/gaearon/pen/oWWQNa?editors=0010>`_ ，可以看到一个空的井字棋盘和 React 代码
- 然后，在该模板中修改 React 代码

### 搭建本地开发环境

- 1.安装最新版的 `Node.js <https://nodejs.org/en/>`_ 
- 2.按照 `Create React App 安装指南 <https://zh-hans.reactjs.org/docs/create-a-new-react-app.html#create-react-app>`_ 创建一个新的项目

```bash
$ npx create-react-app my-app
```

- 3.删除掉新项目中的 `src/` 文件夹下所有的文件

```bash
$ cd my-app
$ cd src
$ rm -f *
$ cd ..
```

- 4.在 `src/` 文件夹中创建一个名为 `index.css` 的文件
- 5.在 `src/` 文件夹下创建一个名为 `index.js` 的文件
- 6.拷贝以下三行代码到 `src/` 文件夹下的 `index.js` 文件的顶部

```js
import React from 'react';
import ReactDOM from 'react-dom';
import './index.css';
```

- 7.在项目文件夹下执行 `npm start` 命令，然后在浏览器访问 `http://localhost:3000`。
这样就可以在浏览器中看见一个空的井字棋的棋盘

## 概览

### React 是什么?

React 是一个声明式、高效且灵活的用于构建用户界面的 JavaScript 库。
使用 React 可以将一些简短、独立的代码片段组合成复杂的 UI 界面，
这些代码片段被称作 “组件”。

React 中拥有多种不同类型的组件，先从 `React.Component` 的子类开始介绍:

```js
class ShoppingList extends React.Component {
   render() {
      return (
         <div className="shopping-list">
            <h1>Shopping List for {this.props.name}</h1>
            <ul>
               <li>Instagrm</li>
               <li>WhatsApp</li>
               <li>Oculus</li>
            </ul>
         </div>
      );
   }
}
// 用法示例：<ShoppingList name="Mark" />
```

通过使用组件来告诉 React 希望在屏幕上看到什么，当数据发生改变时，React 会高效地更新并重新渲染组件

- (1)其中，ShppingList 是一个 React 组件类，或者说是一个 React 组件类型。
一个组件接收一些参数，把这些参数叫做 `props`，`props` 是 "properties" 的简写.
- (2)然后通过 `render` 方法返回需要展示在屏幕上的视图的层次结构。 
`render` 方法的返回值描述了你希望在屏幕上看到的内容。React 根据描述，然后把结果展示出来。更具体地来说，
`render` 返回了一个 React 元素，这是一种对渲染内容的轻量级描述。大多数的 React 开发者使用了一种名为 "JSX" 的特殊语法，
JSX 可以让你更轻松地书写这些结构.
- (3)语法 `<div />` 会被编译成 `React.createElement('div')`。因此上述代码等同于:

```js
return React.createElement(
   'div', 
   {className: 'shopping-list'},
   React.createElement('h1', /* ... h1 children ... */),
   React.createELement('ul', /* ... ul children ... */)
);
```

## 阅读初始代码

- 三个 React 组件
  - Square
     - Square 组件渲染了一个单独的 `<button>`
  - Board
     - Board 组件渲染了 9(25) 个方块
  - Game
     - Game 组件渲染了含有默认值的一个棋盘

### 通过 Props 传递数据

- 1.将数据从 Board 组件传递到 Square 组件中

```js
// 传递一个名为 value 的 prop 到 Square 当中
class Board extends React.Component {
   renderSquare(i) {
      return <Square value={i} />;
   }
}
```

- 2.修改 Square 组件中的 render 方法，把 `{/* TODO */}` 替换为 `{this.props.value}`，以显示上文中传入的值

```js
class Square extends React.Component {
   render() {
      return (
         <button className="square">
            {this.props.value}
         </button>
      );
   }
}
```

- 3.刚刚成功地把一个 prop 从父组件 Board “传递” 给了子组件 Square。
在 React 应用中，数据通过 props 的传递，从父组件流向子组件.

### 给组件添加交互功能

让棋盘的每一个格子在点击之后能落下一颗 "X" 作为棋子

- 1.首先，把 Square 组件中的 `render()` 方法的返回值中的 button 标签修改为如下内容

```js
class Square extends React.Component {
   render() {
      return (
         <button className="square" onClick={function() { alert("click"); }}>
            {this.props.value}
         </button>
      );
   }
}

// 为了少输入代码，同时为了避免 this 造成的困扰，建议使用箭头函数来进行事件处理
class Square extends React.Component {
   render() {
      return (
         <button className="square" onClick={() => { alert("click"); }}>
            {this.props.value}
         </button>
      );
   }
}
```

- 2.接下来，希望 Square 组件可以记住它被点击过，然后用 "X" 来填充对应的方格，
用 state 来实现所谓"记忆"的功能。可以通过在 React 组件的构造函数中设置 `this.state` 来初始化 state。
`this.state` 应该被视为一个组件的私有属性，在 `this.state` 中存储当前每个方格(Square)的值，
并且在每次方格被点击的时候改变这个值

- (2.1)首先，向这个 class 中添加一个构造函数，用来初始化 state

```js
class Square extends React.Component {
   constructor(props) {
      super(props);
      this.state = {
         value: null,
      };
   }

   render() {
      return (
            <button className="square" onClick={() => { alert("click"); }}>
               {this.props.value}
            </button>
      );
   }
}
```

- (2.2)现在，修改一下 Square 组件的 render 方法，这样，每当方格被点击的时候，就可以显示当前 state 的值了
  - 在 `<button>` 标签中，把 `this.props.value` 替换为 `this.state.value`
  - 将 `onClick={...}` 事件监听函数替换为 `onClick={() => this.setState({value: 'X'})}`
  - 为了更好的可读性，将 `className` 和 `onClick` 的 prop 分两行书写

```js
class Square extends React.Component {
   constructor(props) {
      super(props);
      this.state = {
         value: null,
      };
   }

   render() {
      return (
            <button 
               className="square" 
               onClick={() => this.setState({value: 'X'})}
            >
               {this.state.value}
            </button>
      );
   }
}
```

- (2.3)在 Square 组件 `render` 方法中 onClick 事件监听函数中调用 this.setState，
我们就可以在每次 <button> 被点击的时候通知 React 去重新渲染 Square 组件。组件更新之后，
Square 组件的 this.state.value 的值会变成 "X"，因此，我们在游戏棋盘是哪个就能看见 `X` 了。
点击任意一个方格，`X` 就会出现了。


### 开发者工具

- `Chrome <https://chrome.google.com/webstore/detail/react-developer-tools/fmkadmapgofadopljbjfkapdkoienihi?hl=en>`_ 
- `Firefox <https://addons.mozilla.org/en-US/firefox/addon/react-devtools/>`_ 

## 游戏完善


## 时间旅行
