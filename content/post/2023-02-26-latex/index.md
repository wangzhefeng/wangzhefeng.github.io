---
title: Latex 常用命令
author: 王哲峰
date: '2023-02-26'
slug: latex
categories:
  - markdown
tags:
  - tool
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
h3 {
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

- [数学模式重音符](#数学模式重音符)
- [大小写希腊字母](#大小写希腊字母)
- [二元关系符](#二元关系符)
- [二元运算符](#二元运算符)
- [大尺寸运算符](#大尺寸运算符)
- [箭头](#箭头)
- [定界符](#定界符)
- [大尺寸定界符](#大尺寸定界符)
- [其他符号](#其他符号)
- [非数学符号](#非数学符号)
- [AMS 定界符](#ams-定界符)
- [AMS 希腊和希伯来字母](#ams-希腊和希伯来字母)
- [AMS 二元关系符](#ams-二元关系符)
- [AMS 箭头](#ams-箭头)
- [AMS 二元否定关系符和箭头](#ams-二元否定关系符和箭头)
- [AMS 二元运算符](#ams-二元运算符)
- [AMS 其他符号](#ams-其他符号)
- [数学字母](#数学字母)
</p></details><p></p>

# 数学模式重音符

| 符号 | code |
|----|----|
| `$\hat{a}$` | `\hat{a}` |
| `$\check{a}$` | `\check{a}` |
| `$\tilde{a}$` | `\tilde{a}` |
| `$\acute{a}$` | `\acute{a}` |
| `$\grave{a}$` | `\grave{a}` |
| `$\dot{a}$` | `\dot{a}` |
| `$\ddot{a}$` | `\ddot{a}` |
| `$\breve{a}$` | `\breve{a}` |
| `$\bar{a}$` | `\bar{a}` |
| `$\vec{a}$` | `\vec{a}` |
| `$\widehat{A}$` | `\widehat{A}` |
| `$\widetilde{A}$` | `\widetilde{A}` |

# 大小写希腊字母

| 符号 | code | 大写符号 | Code |
|------------|-----------|----|-----|
| `$\alpha$` | `\alpha` |  |  |
| `$\beta$` | `\beta` |  |  |
| `$\gamma$` | `\gamma` | `$\Gamma$` |  `\Gamma` |
| `$\delta$` | `\delta` | `$\Delta$` |  `\Delta` |
| `$\epsilon$` | `\epsilon` |  |  |
| `$\varepsilon$` | `\varepsilon` | |  |
| `$\zeta$` | `\zeta` |  |  |
| `$\eta$` | `\eta` |  |  |
| `$\theta$` | `\theta` | `$\Theta$` | `\Theta`|
| `$\vartheta$` | `\vartheta` | | |
| `$\iota$` | `\iota` |  |  |
| `$\kappa$` | `\kappa` |  |  |
| `$\lambda$` | `\lambda` | `$\Lambda$` | `\Lambda`|
| `$\mu$` | `\mu` |  |  |
| `$\nu$` | `\nu` |  |  |
| `$\xi$` | `\xi` | `$\Xi$` | `\Xi` |
| `$o$` | `o` |  |  |
| `$\pi$` | `\pi` | `$\Pi$` | `\Pi` |
| `$\varpi$` | `\varpi` | |  |
| `$\rho$` | `\rho` |  |  |
| `$\varrho$` | `\varrho` | |  |
| `$\sigma$` | `\sigma` | `$\Sigma$` | `\Sigma`|
| `$\varsigma$` | `\varsigma` | |  |
| `$\tau$` | `\tau` |  |  |
| `$\upsilon$` | `\upsilon` | `$\Upsilon$` | `\Upsilon`|
| `$\phi$` | `\phi` | `$\Phi$` | `\Phi` |
| `$\varphi$` | `\varphi` | | |
| `$\chi$` | `\chi` |  |  |
| `$\psi$` | `\psi` | `$\Psi$` | `\Psi`|
| `$\omega$` | `\omega` | `$\Omega$` | `\Omega` |

# 二元关系符

你可以在下述命令的前面加上 `\not` 来得到其否定形式

| 符号 | code |
|----|----|
| `$<$` | `<` |
| `$>$` | `>` |
| `$=$` | `=` |
| `$\neq$` | `\neq` |
| `$\leq$` | `\leq` |
| `$\geq$` | `\geq` |
| `$\equiv$` | `\equiv` |
| `$\ll$` | `\ll` |
| `$\gg$` | `\gg` |
| `$\doteq$` | `\doteq` |
| `$\prec$` | `\prec` |
| `$\preceq$` | `\preceq` |
| `$\succ$` | `\succ` |
| `$\succeq$` | `\succeq` |
| `$\sim$` | `\sim` |
| `$\simeq$` | `\simeq` |
| `$\subset$` | `\subset` |
| `$\subseteq$` | `\subseteq` |
| `$\supset$` | `\supset` |
| `$\supseteq$` | `\supseteq` |
| `$\sqsubset$` | `\sqsubset` |
| `$\sqsubseteq$` | `\sqsubseteq` |
| `$\sqsupset$` | `\sqsupset` |
| `$\sqsupseteq$` | `\sqsupseteq` |
| `$\cong$` | `\cong` |
| `$\Join$` | `\Join` |
| `$\bowtie$` | `\bowtie` |
| `$\in$` | `\in` |
| `$\notin$` | `\notin` |
| `$\ni$` | `\ni` |
| `$\propto$` | `\propto` |
| `$\vdash$` | `\vdash` |
| `$\dashv$` | `\dashv` |
| `$\models$` | `\models` |
| `$\mid$` | `\mid` |
| `$\parallel$` | `\parallel` |
| `$\perp$` | `\perp` |
| `$\smile$` | `\smile` |
| `$\frown$` | `\frown` |
| `$\asymp$` | `\asymp` |
| `$:$` | `:` |




# 二元运算符


# 大尺寸运算符


# 箭头

# 定界符


# 大尺寸定界符

# 其他符号

# 非数学符号

# AMS 定界符

# AMS 希腊和希伯来字母

# AMS 二元关系符


# AMS 箭头


# AMS 二元否定关系符和箭头


# AMS 二元运算符


# AMS 其他符号

# 数学字母
