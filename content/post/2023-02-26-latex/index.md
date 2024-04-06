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
img {
    pointer-events: none;
}
</style>


<details><summary>目录</summary><p>

- [数学符号](#数学符号)
    - [数学模式重音符](#数学模式重音符)
    - [大小写希腊字母](#大小写希腊字母)
    - [数学字母](#数学字母)
    - [矩阵](#矩阵)
    - [等号对齐](#等号对齐)
    - [方程组](#方程组)
    - [开根号](#开根号)
    - [上下线、括号](#上下线括号)
- [二元关系符](#二元关系符)
    - [普通二元关系符](#普通二元关系符)
    - [AMS 二元关系符](#ams-二元关系符)
- [运算符](#运算符)
    - [二元运算符](#二元运算符)
    - [大尺寸运算符](#大尺寸运算符)
    - [定界符](#定界符)
    - [大尺寸定界符](#大尺寸定界符)
    - [箭头](#箭头)
    - [其他数学符号](#其他数学符号)
- [参考](#参考)
</p></details><p></p>

# 数学符号

## 数学模式重音符

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

## 大小写希腊字母

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

## 数学字母

| 描述 | code | 符号 | 备注 |
|------|-----|------|------|
| 正常字母 | ABCdef | `$ABCdef$` | |
| 正体字母 | \mathrm{ABCdef} | `$\mathrm{ABCdef}$` | |
| 粗体字母 | \mathbf{ABCdef} | `$\mathbf{ABCdef}$` | |
| 斜体字母 | \mathit{ABCdef} | `$\mathit{ABCdef}$` | |
| 花体字母-1 | \mathcal{ABC} | `$\mathcal{ABC}$` | 仅适用于大写字母 |
| 花体字母-2 | \mathscr{ABC} | `$\mathscr{ABC}$` | 仅适用于大写字母 |
| 花体字母-3 | \mathfrak{ABCdef} | `$\mathfrak{ABCdef}$` | |
| 空心字母 | \mathbb{ABCdef} | `$\mathbb{ABC}$` | 仅适用于大写字母 |
| 某种字体 | \mathsf{ABCdef} | `$\mathsf{ABCdef}$` | |
| 某种字体 | \mathtt{ABCdef} | `$\mathtt{ABCdef}$` | |
| 某种字体 | \mathnormal{ABCdef} | `$\mathnormal{12345}$` |  存在问题 |

## 矩阵

```
`$$\begin{matrix}
1 & 0 \\
0 & 1
\end{matrix}$$`
```

`$$\begin{matrix}
1 & 0 \\
0 & 1
\end{matrix}$$`

```
`$$\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}$$`
```

`$$\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}$$`

```
`$$\begin{Bmatrix}
1 & 0 \\
0 & 1
\end{Bmatrix}$$`
```

`$$\begin{Bmatrix}
1 & 0 \\
0 & 1
\end{Bmatrix}$$`

## 等号对齐

```
`$$\begin{align}
f(x) 
&= (x + 1)^{2} \\
&= x^{2} + 2x + 1
\end{align}$$`
```

`$$\begin{align}
f(x) 
&= (x + 1)^{2} \\
&= x^{2} + 2x + 1
\end{align}$$`

## 方程组

```
`$$f(x) = \begin{cases}
0, x > 0, \\
1, x \leq 0
\end{cases}$$`
```

`$$f(x) = \begin{cases}
0, x > 0, \\
1, x \leq 0
\end{cases}$$`

```
`$$\begin{cases}
x + 1 = 0 \\
x + 2 = 1
\end{cases}$$`
```

`$$\begin{cases}
x + 1 = 0 \\
x + 2 = 1
\end{cases}$$`

## 开根号

```
`$$\sqrt{x}$$`
```

`$$\sqrt{x}$$`

```
`$$\sqrt[3]{2}$$`
```

`$$\sqrt[3]{2}$$`

## 上下线、括号

```
`$$\overline{m+n}$$`
```

`$$\overline{m+n}$$`

```
`$$\underline{m+n}$$`
```

`$$\underline{m+n}$$`


```
`$$\underbrace{ a + b + \cdots + z }_{26}$$`
```

`$$\underbrace{ a + b + \cdots + z }_{26}$$`

```
`$$\vec a$$`
```

`$$\vec a$$`


```
`$$\overrightarrow{AB}$$`
```

`$$\overrightarrow{AB}$$`

```
`$1\frac{1}{2}$`
```

`$$1\frac{1}{2}$$`


# 二元关系符

## 普通二元关系符

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

## AMS 二元关系符

| code | 符号 |
|----|----|
| \leqslant | `$\leqslant$` |
| \geqslant | `$\geqslant$` |
| \lll| `$\lll$` |
| \ggg | `$\ggg$` |
| \lesseqgtr| `$\lesseqgtr$` |
| \gtreqless | `$\gtreqless$` |
| \thicksim | `$\thicksim$` |
| \thickapprox | `$\thickapprox$` |
| \backsim | `$\backsim$` |
| \therefore | `$\therefore$` |
| \because | `$\because$` |
| \varpropto | `$\varpropto$` |

# 运算符

## 二元运算符

| 符号 | code |
|----|----|
| `$+$` | + |
| `$-$` | - |
| `$\times$` | \times |
| `$\cdot$` | \cdot |
| `$\div$` | \div |
| `$\setminus$` | \setminus |
| `$\pm$` | \pm |
| `$\mp$` | \mp |
| `$\oplus$` | \oplus |
| `$\ominus$` | \ominus |
| `$\otimes$` | \otimes |
| `$\odot$` | \odot |
| `$\oslash$` | \oslash |
| `$\star$` | \star |
| `$\ast$` | \ast |
| `$\circ$` | \circ |
| `$\bullet$` | \bullet |
| `$\diamond$` | \diamond |
| `$\triangleleft$` | \triangleleft |
| `$\triangleright$` | \triangleright |
| `$\bigtriangleup$` | \bigtriangleup |
| `$\bigtriangledown$` | \bigtriangledown |
| `$\lhd$` | \lhd |
| `$\rhd$` | \rhd |
| `$\unlhd$` | \unlhd |
| `$\unrhd$` | \unrhd |

## 大尺寸运算符

| 符号     | code |
|----------|----|
| `$\sum$` | \sum |
| `$\prod$` | \prod |
| `$\coprod$` | \coprod |
| `$\int$` | \int |
| `$\oint$` | \oint |
| `$\bigcup$` | \bigcup |
| `$\bigcap$` | \bigcap |
| `$\bigsqcup$` | \bigsqcup |
| `$\bigvee$` | \bigvee |
| `$\bigwedge$` | \bigwedge |
| `$\bigoplus$` | \bigoplus |
| `$\bigotimes$` | \bigotimes |
| `$\bigodot$` | \bigodot |
| `$\biguplus$` | \biguplus |

## 定界符

| 符号   | code |
|-------|----|
| `$($` | ( |
| `$)$` | ) |
| `$[$` | [ |
| `$]$` | ] |
| `$\{$` | \{ |
| `$\}$` | \} |
| `$\uparrow$` | \uparrow |
| `$\downarrow$` | \downarrow |
| `$\updownarrow$` | \updownarrow |
| `$\Uparrow$` | \Uparrow |
| `$\Downarrow$` | \Downarrow |
| `$\Updownarrow$` | \Updownarrow |
| `$\langle$` | \langle |
| `$\rangle$` | \rangle |
| `$\|$` | `\|` or \vert |
| `$\\|$` | `\\|` or \Vert |
| `$\lfloor$` | \lfloor |
| `$\rfloor$` | \rfloor |
| `$\lceil$` | \lceil |
| `$\rceil$` | \rceil |

## 大尺寸定界符

| 符号 | code |
|----|----|
| `$\lgroup$` | \lgroup |
| `$\rgroup$` | \rgroup |
| `$\lmoustache$` | \lmoustache |
| `$\rmoustache$` | \rmoustache |
| `$\arrowvert$` | \arrowvert |
| `$\Arrowvert$` | \Arrowvert |
| `$\bracevert$` | \bracevert |

## 箭头

| 符号                | code            |
|---------------------|-----------------|
| `$\leftarrow$`      | \leftarrow      |
| `$\rightarrow$`     | \rightarrow     |
| `$\leftrightarrow$` | \leftrightarrow |
| `$\Leftarrow$`      | \Leftarrow      |
| `$\Rightarrow$`     | \Rightarrow     |
| `$\Leftrightarrow$` | \Leftrightarrow |
| `$\uparrow$`        | \uparrow        |
| `$\downarrow$`      | \downarrow      |

## 其他数学符号

| 符号 | code |
|----|----|
| `$\dots$` | \dots |
| `$\cdots$` | \cdots |
| `$\vdots$` | \vdots |
| `$\ddots$` | \ddots |
| `$\forall$` | \forall |
| `$\exists$` | \exists |
| `$\partial$` | \partial |
| `$'$` | ' |
| `$\prime$` | \prime |
| `$\emptyset$` | \emptyset |
| `$\infty$` | \infty |
| `$\nabla$` | \nabla |
| `$\triangle$` | \triangle |
| `$\bot$` | \bot |
| `$\top$` | \top |
| `$\angle$` | \angle |
| `$\neg$` | \neg or \lnot |

# 参考

* [常用数学符号的 LaTeX 表示方法](http://mohu.org/info/symbols/symbols.htm)

