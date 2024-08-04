---
title: Streamlit
author: 王哲峰
date: '2024-08-04'
slug: streamlit
categories:
  - streamlit
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

- [Streamlit 简介](#streamlit-简介)
- [参考](#参考)
</p></details><p></p>


# Streamlit 简介

Streamlit 是一个用于快速创建数据应用程序的开源 Python 库。
它的设计目标是让数据科学家能够轻松地将数据分析和机器学习模型转化为具有交互性的 Web 应用程序，
而无需深入了解 Web 开发。和常规 Web 框架，如 Flask/Django 的不同之处在于，
它不需要你去编写任何客户端代码（HTML/CSS/JS），只需要编写普通的 Python 模块，
就可以在很短的时间内创建美观并具备高度交互性的界面，从而快速生成数据分析或者机器学习的结果；
另一方面，和那些只能通过拖拽生成的工具也不同的是，你仍然具有对代码的完整控制权。

Streamlit 提供了一组简单而强大的基础模块，用于构建数据应用程序：

* `st.write()`：这是最基本的模块之一，用于在应用程序中呈现文本、图像、表格等内容。
* `st.title()`、`st.header()`、`st.subheader()`：这些模块用于添加标题、子标题和分组标题，以组织应用程序的布局。
* `st.text()`、`st.markdown()`：用于添加文本内容，支持 Markdown 语法。
* `st.image()`：用于添加图像到应用程序中。
* `st.dataframe()`：用于呈现 Pandas 数据框。
* `st.table()`：用于呈现简单的数据表格。
* `st.pyplot()`、`st.altair_chart()`、`st.plotly_chart()`：用于呈现 Matplotlib、Altair 或 Plotly 绘制的图表。
* `st.selectbox()`、`st.multiselect()`、`st.slider()`、`st.text_input()`：用于添加交互式小部件，
  允许用户在应用程序中进行选择、输入或滑动操作。
* `st.button()`、`st.checkbox()`、`st.radio()`：用于添加按钮、复选框和单选按钮，以触发特定的操作。

这些基础模块使得通过 Streamlit 能够轻松地构建交互式数据应用程序，
并且在使用时可以根据需要进行组合和定制，更多内容请查看[Streamlit 官方文档]()





# 参考

* [Streamlit 官方文档]()