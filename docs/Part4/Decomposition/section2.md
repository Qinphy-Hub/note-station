# 线性规划：复杂约束

## 问题描述

&emsp;&emsp;与复杂约束问题类似，考虑这样一个一般性问题：

$$
\begin{align*}
\begin{array}{ccccccccccc}
(\mathcal{G})\quad\min\limits_{\boldsymbol{x}_1,\boldsymbol{x}_2,...,\boldsymbol{x}_l,\boldsymbol{y}} & \boldsymbol{c}_1^\top\boldsymbol{x}_1 & + & \boldsymbol{c}_2^\top\boldsymbol{x}_2 & + & \cdots & + &\boldsymbol{c}_l^\top\boldsymbol{x}_l & + & \boldsymbol{c}_0^\top\boldsymbol{y} & \\
\text{s.t.} & A_1\boldsymbol{x}_1 & & & & & & & + & B_1\boldsymbol{y} & =\boldsymbol{b}_1 \\
& & & A_2\boldsymbol{x}_2 & & & & & + & B_2\boldsymbol{y} &=\boldsymbol{b}_2 \\
& & & & & \ddots & & & \vdots & \vdots & \vdots \\
& & & & & & & A_l\boldsymbol{x}_l & + & B_l\boldsymbol{y} & =\boldsymbol{b}_l
\end{array} \\
\boldsymbol{y}\geq0,\boldsymbol{x}_i\geq\boldsymbol{0},\quad i=1,2,...,l\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad \notag
\end{align*}
$$

其中，我们能把决策变量分为$l+1$块，使用$i=1,2,...,l$索引相互独立的分块，决策变量分块表示为$\boldsymbol{x}_i$，剩下一个分块决策变量表示为$\boldsymbol{y}$；相应的成本系数、简单约束的右侧常数项和共有约束系数矩阵也被分为$l$块，分别表示为$\boldsymbol{c}_i,\boldsymbol{b}_i,B_i$，块决策变量$\boldsymbol{y}$的成本系数表示为$\boldsymbol{c}_0$。我们记第$i$组约束的约束数量为$m_i$，也即$\boldsymbol{b}_i$的维度为$m_i$，整个问题的约束数量为$\sum\limits_{i=1}^lm_i$。记第$i$组决策变量的长度为$n_i$，也即$\boldsymbol{x}_i$的维度为$n_i$，$\boldsymbol{y}$的维度记为$n_0$，则整个问题的决策变量数量为$\sum\limits_{i=0}^ln_i$。那么，可以推断$A_i$是一个$m_i\times n_i$的矩阵，$B_i$是一个$m_i\times n_0$的矩阵。

&emsp;&emsp;这个问题的特点在于，我们依旧可以把变量分为$l+1$块，从约束角度看，前$l$个块决策变量均与最后一个块决策变量相关，而前$l$个块决策变量相互之间无关。我们把这种与前$l$个独立的块决策变量相关的变量，称为**复杂变量**(complicated constraints)。显然这个问题中，如果没有变量$\boldsymbol{y}$，这个问题也可以分割为独立的$l$个问题单独求解，而不会有任何损失。

&emsp;&emsp;为方便描述我们延续之前的$\boldsymbol{x},\boldsymbol{c}$记号以外，补充以下符号：

* 关于复杂变量$\boldsymbol{y}$的约束为$B=\left[B_1^\top|B_2^\top|\cdots|B_l^\top\right]^\top$。
* 记整个问题的右侧常数项$\boldsymbol{b}=\left[\boldsymbol{b}_1^\top|\boldsymbol{b}_2^\top|\cdots|\boldsymbol{b}_l^\top\right]^\top$。
* 关于$\boldsymbol{x}$的系数矩阵记为$A=\left[\begin{array}{cccc} A_1 & \boldsymbol{0} & \cdots & \boldsymbol{0} \\ \boldsymbol{0} & A_2 & \cdots & \boldsymbol{0} \\ \vdots & \vdots & \ddots & \vdots \\ \boldsymbol{0} & \boldsymbol{0} & \cdots & A_l \end{array} \right]$。
* 记整个问题的决策变量为$\boldsymbol{v}=\left(\boldsymbol{x}^\top,\boldsymbol{y}^\top\right)^\top$。

## 基本思想

!!! danger "定理 2.1 (原始与对偶的可分解性)"

    &emsp;&emsp;如果一个线性规划问题具有具有复杂约束的可分解结构，那么它的对偶线性规划问题也具有具有复杂变量的可分解的结构。相反，如果一个线性规划问题具有具有复杂变量的可分解结构，那么它的对偶线性规划问题也具有复杂约束的可分解的结构。

!!! quote "证"

    &emsp;&emsp;根据线性规划对偶问题的定义，若原始问题的约束为$A$，则其对偶问题的约束为$A^\top$。然后，结合复杂约束问题的定义，可以看出其约束矩阵的转置就是复杂变量问题的约束矩阵形式，反之亦然。

&emsp;&emsp;根据定理2.1，我们自然会有这种想法：使用DW方法求解问题$(\mathcal{G})$的对偶问题，从而得到原问题的最优解。这一节我们尝试使用新的分解方法，这种方法常常应用在<u>复杂变量为整数变量</u>的时候，用于把整数变量分解出来。我们将上述一般问题$(\mathcal{G})$简化：

$$
\begin{equation}\label{simple}\tag{1}
\begin{split}
\min\limits_{\boldsymbol{x},\boldsymbol{y}}\quad&\boldsymbol{c}^\top\boldsymbol{x}+\boldsymbol{c}_0^\top\boldsymbol{y} \\
\text{s.t.}\quad&A\boldsymbol{x}+B\boldsymbol{y}=\boldsymbol{b} \\
&\boldsymbol{x}\geq0,\boldsymbol{y}\geq0
\end{split}
\end{equation}
$$

&emsp;&emsp;我们的目标是把决策变量$\boldsymbol{y}$分离出来（它可能是整数变量），因为只关于$\boldsymbol{x}$的部分是一个线性规划问题，求解更加简单。对于原问题$(\ref{simple})$，暴力拆分为以下两个问题：

$$
\begin{align*}
(\mathcal{M})\quad\min\limits_{\boldsymbol{x}}\quad&\boldsymbol{c}^\top\boldsymbol{x}+\alpha(\boldsymbol{x}) \\
\text{s.t.}\quad&\boldsymbol{x}\geq0
\end{align*}
$$

其中，$\alpha(\boldsymbol{x})$取决于下面的优化问题的最优值：

$$
\begin{align*}
(\mathcal{S})\quad\alpha(\boldsymbol{x})=\min\limits_{\boldsymbol{y}}\quad&\boldsymbol{c}^\top_0\boldsymbol{y} \\
\text{s.t.}\quad&B\boldsymbol{y}=\boldsymbol{b}-A\boldsymbol{x} \\
&\boldsymbol{y}\geq0
\end{align*}
$$

&emsp;&emsp;关键在于{==如何通过子问题$(\mathcal{S})$给$\alpha$函数一个线性的约束，从而实现分离出的主问题$(\mathcal{M})$逼近原问题==}。

!!! danger "定理 2.2 ($\alpha(\boldsymbol{x})$是凸函数)"

    &emsp;&emsp;由上式$(\mathcal{S})$定义的函数$\alpha(\boldsymbol{x})$是凸的。

!!! quote "证"

    &emsp;&emsp;证明函数是凸的，也即证明$\lambda\alpha\left(\boldsymbol{x}^{(1)}\right)+(1-\lambda)\alpha\left(\boldsymbol{x}^{(2)}\right)\geq\alpha\left(\boldsymbol{x}^{(3)}\right),\lambda\in[0,1]$。

    &emsp;&emsp;考虑原始问题的两个可行解$\boldsymbol{v}^{(1)}$和$\boldsymbol{v}^{(2)}$，它们分别对应$\boldsymbol{x}^{(1)},\boldsymbol{y}^{(1)}$和$\boldsymbol{x}^{(2)},\boldsymbol{y}^{(2)}$。那么，根据拆分的主问题$(\mathcal{M})$和子问题$(\mathcal{S})$有：

    $$
    \alpha\left(\boldsymbol{x}^{(1)}\right)=\boldsymbol{c}^\top_0\boldsymbol{y}^{(1)},\quad\alpha\left(\boldsymbol{x}^{(2)}\right)=\boldsymbol{c}^\top_0\boldsymbol{y}^{(2)}
    $$

    &emsp;&emsp;考虑$\boldsymbol{v}^{(1)},\boldsymbol{v}^{(2)}$的凸组合$\boldsymbol{v}^{(3)}$，注意到原问题$(\mathcal{G})$的可行域是凸的：

    $$
    \begin{equation}\label{convex}\tag{2}
    \begin{split}
    \boldsymbol{v}^{(3)}=&\lambda\boldsymbol{v}^{(1)}+(1-\lambda)\boldsymbol{v}^{(2)} \\
    \left(\begin{array}{c}\boldsymbol{x}^{(3)} \\ \boldsymbol{y}^{(3)} \end{array}\right)=&\lambda\left(\begin{array}{c}\boldsymbol{x}^{(1)} \\ \boldsymbol{y}^{(1)} \end{array}\right)+(1-\lambda)\left(\begin{array}{c}\boldsymbol{x}^{(2)} \\ \boldsymbol{y}^{(2)} \end{array}\right)
    \end{split}
    \end{equation}
    $$

    &emsp;&emsp;所以得到$\lambda\boldsymbol{y}^{(1)}+(1-\lambda)\boldsymbol{y}^{(2)}=\boldsymbol{y}^{(3)}$和$\lambda\boldsymbol{x}^{(1)}+(1-\lambda)\boldsymbol{x}^{(2)}=\boldsymbol{x}^{(3)}$。

    &emsp;&emsp;那么计算$\lambda\alpha\left(\boldsymbol{x}^{(1)}\right)+(1-\lambda)\alpha\left(\boldsymbol{x}^{(2)}\right)$，则有：

    $$
    \begin{align*}
    \lambda\alpha\left(\boldsymbol{x}^{(1)}\right)+(1-\lambda)\alpha\left(\boldsymbol{x}^{(2)}\right)=&\lambda\alpha\left(\boldsymbol{x}^{(1)}\right)+(1-\lambda)\alpha\left(\boldsymbol{x}^{(2)}\right) \\
    =&\lambda\boldsymbol{c}_0^\top\boldsymbol{y}^{(1)}+(1-\lambda)\boldsymbol{c}^\top_0\boldsymbol{y}^{(2)} \\
    =&\boldsymbol{c}^\top_0\boldsymbol{y}^{(3)}
    \end{align*}
    $$

    &emsp;&emsp;考虑$\alpha\left(\boldsymbol{x}^{(3)}\right)$对应的问题为：

    $$
    \begin{equation}\label{alpha3}\tag{3}
    \begin{split}
    \alpha\left(\boldsymbol{x}^{(3)}\right)=\min\limits_{\boldsymbol{y}}\quad&\boldsymbol{c}^\top_0\boldsymbol{y} \\
    \text{s.t.}\quad&B\boldsymbol{y}=\boldsymbol{b}-A\boldsymbol{x}^{(3)} \\
    &y_i\geq0,i=1,2,...,n_0
    \end{split}
    \end{equation}
    $$

    &emsp;&emsp;根据$B\boldsymbol{y}^{(1)}\leq\boldsymbol{b}-A\boldsymbol{x}^{(1)}$、$B\boldsymbol{y}^{(2)}\leq\boldsymbol{b}-A\boldsymbol{x}^{(2)}$以及$\boldsymbol{x},\boldsymbol{y}$的凸性$(\ref{convex})$，可知$B\boldsymbol{y}^{(3)}\leq\boldsymbol{b}-A\boldsymbol{x}^{(3)}$，也即$\boldsymbol{y}^{(3)}$属于问题$(\ref{alpha3})$的可行域。置问题$(\ref{alpha3})$的最优解为$\boldsymbol{y}^*$，显然有：

    $$
    \alpha\left(\boldsymbol{x}^{(3)}\right)=\boldsymbol{c}_0^\top\boldsymbol{y}^*\leq\boldsymbol{c}_0^\top\boldsymbol{y}^{(3)}
    $$

    综上可得：

    $$
    \lambda\alpha\left(\boldsymbol{x}^{(1)}\right)+(1-\lambda)\alpha\left(\boldsymbol{x}^{(2)}\right)\geq\alpha\left(\boldsymbol{x}^{(3)}\right),\lambda\in[0,1]
    $$

    即得证。

&emsp;&emsp;我们把主问题$(\mathcal{M})$称为原问题$(\ref{simple})$的松弛问题。在第$k$次迭代中，由主问题得到的解$\boldsymbol{x}^{(k)}$对应的最优值是原问题的一个<u>下界</u>。

&emsp;&emsp;在给定的$\boldsymbol{x}^{(k)}$条件下，考虑以下子问题：

$$
\begin{align*}
\alpha\left(\boldsymbol{x}^{(k)}\right)=\min\limits_{\boldsymbol{y}}\quad&\boldsymbol{c}^\top_0\boldsymbol{y} \\
\text{s.t.}\quad&B\boldsymbol{y}=\boldsymbol{b}-A\boldsymbol{x}^{(k)} \\
&\boldsymbol{y}\geq0
\end{align*}
$$

&emsp;&emsp;记这个子问题的解为$\boldsymbol{y}^{(k)}$。我们可以观察这个子问题的对偶问题：

$$
\begin{align*}
(\mathcal{DS})\quad\max\limits_{\boldsymbol{v}}\quad&\boldsymbol{v}^\top\left(\boldsymbol{b}-A\boldsymbol{x}^{(k)}\right) \\
\text{s.t.}\quad&B^\top\boldsymbol{v}\leq\boldsymbol{c}_0
\end{align*}
$$

&emsp;&emsp;由对偶理论可知$\boldsymbol{v}^\top\left(\boldsymbol{b}-A\boldsymbol{x}^{(k)}\right)=\boldsymbol{c}_0^\top\boldsymbol{y}^{(k)}$，因为$\boldsymbol{x}^{(k)}$是松弛解，那么它作为约束对子问题来说是过于严格的，所以子问题的解$\boldsymbol{y}^{(k)}$对应的最优值是原问题的一个<u>上界</u>。而在这个对偶问题中，我们还发现$\boldsymbol{x}$只与目标函数相关，根据上界$\alpha\left(\boldsymbol{x}^{(k)}\right)=\boldsymbol{c}_0^\top\boldsymbol{y}^{(k)}\geq\boldsymbol{v}^\top\left(\boldsymbol{b}-A\boldsymbol{x}\right)$，我们就成功把整数约束转化为了线性约束。

> 当$\boldsymbol{x}^{(k)}$在可行域外，那么满足$B\boldsymbol{y}=\boldsymbol{b}-A\boldsymbol{x}^{(k)}$的$\boldsymbol{y}$回会更少，所以更严格。

&emsp;&emsp;当然由于是迭代方法，在$\boldsymbol{x}^{(k)}$找得过于严格的情况下，子问题可能无解，也就是其对偶问题无界。根据线性对偶理论中的Farkas引理，子问题的可行域$B\boldsymbol{y}=\boldsymbol{b}-A\boldsymbol{x}^{(k)},\boldsymbol{x}\geq0$要有可行解，那么要使对偶问题的线性系统$B^\top\boldsymbol{v}\leq0,\boldsymbol{v}^\top\left(\boldsymbol{b}-A\boldsymbol{x}^{(k)}\right)>0$无解。那么添加约束：

$$
\boldsymbol{v}^\top\left(\boldsymbol{b}-A\boldsymbol{x}\right)\leq0
$$

也就是满足这个约束的$\boldsymbol{x}$能够使得子问题有可行解。

&emsp;&emsp;当我们得到关于$\boldsymbol{x}$和$\alpha$足够多的约束，就能逼近原问题的最优解。令$I$是$(\mathcal{DS})$的最优解索引集合，$J$是$(\mathcal{DS})$无界时所有极射线的索引集合。那么主问题可以写为：

$$
\begin{align*}
\min\limits_{\boldsymbol{x},\alpha}\quad&\boldsymbol{c}^\top\boldsymbol{x}+\alpha \\
\text{s.t.}\quad&\boldsymbol{v}_i^\top(\boldsymbol{b}-A\boldsymbol{x})\leq\alpha\quad\forall i\in I \\
&\boldsymbol{v}_j^\top(\boldsymbol{b}-A\boldsymbol{x})\leq0\quad\forall j\in J \\
&\boldsymbol{x}\geq0
\end{align*}
$$

## 算法描述

1. **Step 0** - (初始化)：初始化迭代计数器$k=1$，约束索引$I=\emptyset,J=\emptyset$。求解初始主问题：

    $$
    \begin{align*}
    \min\limits_{\boldsymbol{x},\alpha}\quad&\boldsymbol{c}^\top\boldsymbol{x}+\alpha \\
    \text{s.t.}\quad
    &\boldsymbol{x}\geq0
    \end{align*}
    $$

    &emsp;&emsp;必须限制$\alpha$，否则这个问题将会无界：可以根据实际问题限制。如果当$\boldsymbol{x}^{(k)}=0$或者其它值，能够使得子问题有解，预求解一个子问题，也是限制$\alpha$的方法。求解这个问题可以得到$\boldsymbol{x}^{(k)}$和$\alpha^{(k)}$。

2. **Step 1** - (求解子问题)：根据主问题得到$\boldsymbol{x}^{(k)}$，我们可以构建如下子问题。

    $$
    \begin{align*}
    \min\limits_{\boldsymbol{y}}\quad&\boldsymbol{c}^\top_0\boldsymbol{y} \\
    \text{s.t.}\quad&B\boldsymbol{y}=\boldsymbol{b}-A\boldsymbol{x}^{(k)} \\
    &\boldsymbol{y}\geq0
    \end{align*}
    $$

    &emsp;&emsp;当子问题有解$\boldsymbol{y}^{(k)}$，添加约束$\boldsymbol{v}_k^\top(\boldsymbol{b}-A\boldsymbol{x})\leq\alpha$，并添加约束索引$I=\{k\}\cup I$。如果子问题无解，则添加约束$\boldsymbol{v}_j^\top(\boldsymbol{b}-A\boldsymbol{x})\leq0$，并添加约束索引$J=\{k\}\cup J$。

3. **Step 2** - (收敛性检测)：计算原始问题目标函数最优值为上界。

    $$
    \begin{equation}\label{ub}\tag{4}
    z_{\mathrm{up}}^{(k)}=\boldsymbol{c}^\top\boldsymbol{x}^{(k)}+\boldsymbol{c}_0^\top\boldsymbol{y}^{(k)}
    \end{equation}
    $$

    &emsp;&emsp;计算由主问题(也是松弛问题)的目标函数值，得到下界：

    $$
    \begin{equation}\label{lb}\tag{5}
    z_{\mathrm{low}}^{(k)}=\boldsymbol{c}^\top\boldsymbol{x}^{(k)}+\alpha^{(k)}
    \end{equation}
    $$

    &emsp;&emsp;如果$z_{\mathrm{up}}^{(k)}-z_{\mathrm{low}}^{(k)}=\boldsymbol{c}_0^\top\boldsymbol{y}^{(k)}-\alpha^{(k)}<\varepsilon$，停止迭代，最优解为$\boldsymbol{x}^{(k)},\boldsymbol{y}^{(k)}$；否则进入下一步。

3. **Step 3** - (求解主问题)：接下来，解决以下主问题。

    $$
    \begin{align*}
    \min\limits_{\boldsymbol{x},\alpha}\quad&\boldsymbol{c}^\top\boldsymbol{x}+\alpha \\
    \text{s.t.}\quad&\boldsymbol{v}_i^\top(\boldsymbol{b}-A\boldsymbol{x})\leq\alpha\quad\forall i\in I \\
    &\boldsymbol{v}_j^\top(\boldsymbol{b}-A\boldsymbol{x})\leq0\quad\forall j\in J \\
    &\boldsymbol{x}\geq0
    \end{align*}
    $$

    &emsp;&emsp;我们可以得到$\boldsymbol{x}^{(k+1)}$和$\alpha^{(k+1)}$。更新计数器$k=k+1$，回到Step 1。

## 示例与代码

> 本文示例均来自[MatrixOptim网站](https://bookdown.org/edxu96/matrixoptim/benders-for-standard-milp.html)。

&emsp;&emsp;我们给定如下形式的混合整数规划问题：

$$
\begin{align*}
\min\quad&5x_1+3x_2-3y_1+y_2 \\
\text{s.t.}\quad&x_1+3x_2+2y_1-4y_2\geq4 \\
&2x_1+x_2-y_1+2y_2\geq0 \\
&x_1-5x_2-3y_1+y_2\geq-13 \\
&x_1,x_2\geq0 \\
&y_1,y_2\in\mathbb{N}
\end{align*}
$$

&emsp;&emsp;我们可以转化为如下的标准形式：

$$
\begin{align*}
\min\quad&\boldsymbol{c}^\top\boldsymbol{x}+\boldsymbol{f}^\top\boldsymbol{y} \\
&A\boldsymbol{x}+B\boldsymbol{y}\geq\boldsymbol{b} \\
&\boldsymbol{x}\geq0,\boldsymbol{y}\in\mathbb{N}^2\subseteq\mathbb{R}^2
\end{align*}
$$

其中，成本参数$c=\left[\begin{array}{c}5\\3\end{array}\right],f=\left[\begin{array}{c}-3\\1\end{array}\right]$；右常数$\boldsymbol{b}=\left[\begin{array}{c}4\\0\\-13\end{array}\right]$；连续变量$\boldsymbol{x}=\left[\begin{array}{c}x_1\\x_2\end{array}\right]$和整数变量$\boldsymbol{y}=\left[\begin{array}{c}y_1\\y_2\end{array}\right]$；约束矩阵$A=\left[\begin{array}{cc}1 & 3\\2 & 1 \\ 1&-5\end{array}\right],B=\left[\begin{array}{cc}2 & -4\\-1&2\\-3&1\end{array}\right]$。

&emsp;&emsp;因为每次迭代、以及每个子问题和主问题都会用到这些参数，且过程中不会发生变化，所以我们可以提前设置这些参数为全局变量：

```python
import numpy as np

c = np.array([5, 3])
f = np.array([-3, 1])
b = np.array([4, 0, -13])
A = np.array([[1, 3], [2, 1], [1, -5]])
B = np.array([[2, -4], [-1, 2], [-3, 1]])
```

1. 首先，我们先把原问题拆分，其子问题为：

    $$
    \begin{align*}
    (\mathcal{S})\quad\min\quad&\boldsymbol{c}^\top\boldsymbol{x} \\
    \text{s.t.}\quad&A\boldsymbol{x}\geq\boldsymbol{b}-B\overline{\boldsymbol{y}} \\
    &\boldsymbol{x}\geq0
    \end{align*}
    $$

    其中，$\overline{\boldsymbol{y}}=\left[\begin{array}{c}\overline{y}_1\\\overline{y}_2\end{array}\right]$是已知的辅助参数。

    $$
    \begin{align*}
    (\mathcal{S})\quad\min\quad&5x_1+3x_2 \\
    \text{s.t.}\quad&x_1+3x_2\geq4-2\overline{y}_1+4\overline{y}_2 \\
    &2x_1+x_2\geq \overline{y}_1-2\overline{y}_2 \\
    &x_1-5x_2\geq-13+3\overline{y}_1-\overline{y}_2 \\
    &x_1,x_2\geq0
    \end{align*}
    $$

	&emsp;&emsp;虽然迭代过程主要使用其对偶问题，但最后确认$\boldsymbol{x}$的最优解时可以快速由$\boldsymbol{y}$的最优解得到：

    ```python
    from gurobipy import Model, GRB

    def sub_problem(y):
        m = Model()
        x = m.addMVar(2, vtype=GRB.CONTINUOUS, name='x')
        m.setObjective(c @ x, GRB.MINIMIZE)
        m.addConstr(A @ x >= b - B @ y)
        m.setParam('outPutFlag', 0)
        m.update()
        m.optimize()
        if m.status == GRB.Status.OPTIMAL:
            return x.X
        return None
    ```

2. 期间我们需要得到其对偶问题：

    $$
    \begin{align*}
    (\mathcal{DS})\quad\max\quad&\boldsymbol{\alpha}^\top(\boldsymbol{b}-B\overline{\boldsymbol{y}}) \\
    \text{s.t.}\quad&A^\top\boldsymbol{\alpha}\leq\boldsymbol{c} \\
    &\boldsymbol{\alpha}\geq0
    \end{align*}
    $$

    其中，$\boldsymbol{\alpha}=\left[\begin{array}{c}\alpha_1\\\alpha_2\\\alpha_3\end{array}\right]$是对偶决策变量。

    $$
    \begin{align*}
    (\mathcal{DS})\quad\max\quad&(4-2\overline{y}_1+4\overline{y}_2)\alpha_1+(\overline{y}_1-2\overline{y}_2)\alpha_2 \\
    &\quad+(-13+3\overline{y}_1-\overline{y}_2)\alpha_3 \\
    \text{s.t.}\quad&\alpha_1+2\alpha_2+\alpha_3\leq5\\
    &3\alpha_1+\alpha_2-5\alpha_3\leq3 \\
    &\alpha_1,\alpha_2,\alpha_3\geq0
    \end{align*}
    $$

	&emsp;&emsp;在每次迭代中，$\overline{\boldsymbol{y}}$是唯一的变量，我们可以把这个子问题写为一个函数。同时求解这个对偶问题时还需要记录其解$\boldsymbol{\alpha}$，这里我们直接把对应解记录在集合$I$和集合$J$中，而非记录其索引：

    ```python
    I = []
    J = []
    def daul_problem(y):
        m = Model('dual_' + str(y))
        alpha = m.addMVar(3, name='alpha')
        m.setObjective(alpha @ (b - B @ y), GRB.MAXIMIZE)
        m.addConstr(A.transpose() @ alpha <= c)
        m.setParam('outPutFlag', 0)
        m.update()
        m.optimize()
        if m.status == GRB.Status.OPTIMAL:
            if I == [] or not np.any(np.all(alpha.X == I, axis=1)):
                I.append(alpha.X)
            return m.ObjVal
        elif m.status == GRB.Status.UNBOUNDED:
            if J == [] or not np.any(np.all(alpha.UnbdRay == J, axis=1)):
                J.append(alpha.UnbdRay)
            return m.ObjVal
        return None
    ```

3. 从子问题处我们获得了其解$\boldsymbol{\alpha}$和最优值$z^*_{\mathcal{DSP}}$和由式子($\ref{ub}$)得到的上界，进而得到新的主问题为：

    $$
    \begin{align*}
    (\mathcal{M})\quad\min\quad&\boldsymbol{f}^\top\boldsymbol{y}+q \\
    \text{s.t.}\quad&(\boldsymbol{\alpha}^i_p)^\top(\boldsymbol{b}-B\boldsymbol{y})\leq q\quad\forall i\in I \\
    &(\boldsymbol{\alpha}_r^j)^\top(\boldsymbol{b}-B\boldsymbol{y})\leq0\quad\forall j\in J \\
    &\boldsymbol{y}\in\mathbb{N}^2\subseteq\mathbb{R}^2 \\
    \end{align*}
    $$

    &emsp;&emsp;初次迭代得到极点为$\alpha_0=1,\alpha_2=0,\alpha_3=0$：

    $$
    \begin{align*}
    (\mathcal{M})\quad\min\quad&-3y_1+y_2+q \\
    \text{s.t.}\quad&4-2y_1+4y_2\leq q \\
    &y_1,y_2\in\mathbb{N}
    \end{align*}
    $$

	&emsp;&emsp;在构建主问题的代码时，由于主问题在迭代过程的变化，需要提前记录约束的数据在全局变量中。

    ```python
    def master_problem():
        m = Model()
        y = m.addMVar(2, vtype=GRB.INTEGER, name='y')
        q = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='q')
        m.setObjective(f @ y + q, GRB.MINIMIZE)
        for i in I:
            m.addConstr(i.transpose() @ (b - B @ y) <= q)
        for j in J:
            m.addConstr(j.transpose() @ (b - B @ y) <= 0)
        m.setParam('outPutFlag', 0)
        m.update()
        m.optimize()
        if m.status == GRB.Status.OPTIMAL:
            return y.X, m.ObjVal
        if m.status == GRB.Status.UNBOUNDED:
            return y.X, -GRB.INFINITY
        return None, None
    ```

	&emsp;&emsp;求解上述主问题，得到新的$\overline{\boldsymbol{y}}$和$q$，以及由上式($\ref{lb}$)得到的下界。比较上下界的差距即可判断程序出口。

&emsp;&emsp;基本代码在前文已经准备好了，以下实现主要的迭代过程：

```python
ub = GRB.INFINITY
lb = -GRB.INFINITY
y_hat = np.random.random((2,))
flag = False
while lb < ub:
    z_s = daul_problem(y_hat)
    if z_s is not None:
        ub = f @ y_hat + z_s
    else:
        flag = True
        break
    y_hat, z_m = master_problem()
    if y_hat is not None:
        lb = z_m
    else:
        flag = True
        break

print("Benders Decomposition:")
if flag:
    print('  No solution')
else:
    print('  obj = ub = lb =', ub)
    print('  y=', y_hat)
    print('  x=', sub_problem(y_hat))
```

&emsp;&emsp;我们可以直接使用Gurobi求解原问题，验证算法的正确性：

```python
def primal_problem():
    m = Model()
    x = m.addMVar(2, vtype=GRB.CONTINUOUS, name='x')
    y = m.addMVar(2, vtype=GRB.INTEGER, name='y')
    m.setObjective(c @ x + f @ y, GRB.MINIMIZE)
    m.addConstr(A @ x + B @ y >= b)
    m.setParam('outPutFlag', 0)
    m.update()
    m.optimize()
    print('primal problem:')
    if m.status == GRB.Status.OPTIMAL:
        print('  x=', x.X)
        print('  y=', y.X)
        print('  obj=', m.ObjVal)
    else:
        print("No Solution!")

primal_problem()
```