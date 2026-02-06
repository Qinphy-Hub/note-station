# 线性规划：复杂约束

## 问题描述

> 明白一个问题：什么是复杂约束

&emsp;&emsp;我们考虑这样一个线性规划问题：

$$
\begin{align*}
\begin{array}{ccccccccc}
(\mathcal{P})\quad\min\limits_{\boldsymbol{x}_1,\boldsymbol{x}_2,...,\boldsymbol{x}_l} & \boldsymbol{c}_1^\top\boldsymbol{x}_1 & + & \boldsymbol{c}_2^\top\boldsymbol{x}_2 & + & \cdots & + &\boldsymbol{c}_l^\top\boldsymbol{x}_l & \\
\text{s.t.} & A_1\boldsymbol{x}_1 & & & & & & & =\boldsymbol{b}_1 \\
& & & A_2\boldsymbol{x}_2 & & & & & =\boldsymbol{b}_2 \\
& & & & & \ddots & & & \vdots \\
& & & & & & & A_l\boldsymbol{x}_l & =\boldsymbol{b}_l \\
& B_1\boldsymbol{x}_1 & + & B_2\boldsymbol{x}_2 & + & \cdots & + & B_l\boldsymbol{x}_l & =\boldsymbol{b}_0
\end{array} \\
\boldsymbol{x}_i\geq\boldsymbol{0},\quad i=1,2,...,l\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad
\end{align*}
$$

其中，根据简单约束我们能把决策变量分为$l$块，使用$i=1,2,...,l$索引分块，决策变量分块表示为$\boldsymbol{x}_i$；相应的成本系数、简单约束的右侧常数项和共有约束系数矩阵也被分为$l$块，分别表示为$\boldsymbol{c}_i,\boldsymbol{b}_i,B_i$；而复杂约束的右侧常数项表示为$\boldsymbol{b}_0$。我们记第$i$组约束的约束数量为$m_i$，也即$\boldsymbol{b}_i$的维度为$m_i$，整个问题的约束数量为$\sum\limits_{i=0}^lm_i$。记第$i$组决策变量的长度为$n_i$，也即$\boldsymbol{x}_i$的维度为$n_i$，整个问题的决策变量数量为$\sum\limits_{i=1}^ln_i$。那么，可以推断$A_i$是一个$m_i\times n_i$的矩阵，$B_i$是一个$m_0\times n_i$的矩阵。

&emsp;&emsp;这个问题的特点是，我们能明显把决策变量分为$l$块，这$l$块的变量共有约束$m_0$个约束，其余约束都是每块独有的。我们把这种共有约束称为**复杂约束**(complicated constraints)。<u>如果没有复杂约束，这个问题可直接分割为$l$个独立的线性规划子问题，而不会有任何损失</u>。

&emsp;&emsp;为方便后续问题的描述，我们简记如下的符号：

* 整个问题的决策变量记为$\boldsymbol{x}=\left[\boldsymbol{x}^\top_1|\boldsymbol{x}_2^\top|\cdots|\boldsymbol{x}_l^\top\right]^\top$，记其长度为$n=\sum\limits_{i=1}^l$，则$\boldsymbol{x}=(x_1,x_2,...,x_n)^\top$。
* 根据$\boldsymbol{x}$的表示，每个分块可表示为：$\boldsymbol{x}_i=(x_j,x_{j+1},...,x_{j+n_i-1})^\top$，其中$j=1+\sum\limits_{k=1}^{i-1}n_k$，令$\sum\limits_{k=1}^0n_k=0$。
* 同理，成本系数也有类似于$\boldsymbol{x}$的表示：$\boldsymbol{c}=\left[\boldsymbol{c}_1^\top|\boldsymbol{c}_2^\top|\cdots|\boldsymbol{c}_l^\top\right]^\top=(c_1,c_2,...,c_n)^\top$。
* 记复杂约束的系数矩阵为$B=\left[B_1|B_2|\cdots|B_l\right]$，则复杂约束可表示为$B\boldsymbol{x}=\boldsymbol{b}_0$。

> 后文迭代过程中，原问题的成本系数与子问题的成本系数都具有上述关系，后文不再赘述。

## Dantzig-Wolfe分解方法

### 基本思想

&emsp;&emsp;直觉上，我们希望先不考虑复杂约束，使得求解变得简单。去除复杂约束之后的问题，我们称为原问题的松弛问题。

!!! success "基本观察"

    &emsp;&emsp;线性规划问题的基本可行解的线性凸组合是该问题的可行解。([点此回顾线性规划内容](../Linear/section1.md))

&emsp;&emsp;假设我们找到了松弛问题的$p$个基本可行解，记为$\boldsymbol{x}^{(1)},\boldsymbol{x}^{(2)},...,\boldsymbol{x}^{(p)}$。那么这$p$个基本可行解的凸组合依旧是该松弛问题的可行解，记为$\boldsymbol{y}=\sum\limits_{s=1}^p\lambda_s\boldsymbol{x}^{(s)}$，其中$\sum\limits_{s=1}^p\lambda_s=1$。那么，{++我们用这样的凸组合表示所有可行解，然后找到满足复杂约束的解，即为原问题的可行解++}。如果在原问题的可行解中，能够使得目标函数值最小，那么就是原问题的最优解。这也就是**主问题**(master problem)的思路。

&emsp;&emsp;因为线性性，一些值可以表示为：

* 基本可行解下的复杂约束的值可表示为$\boldsymbol{r}^{(s)}=B\boldsymbol{x}^{(s)}$，其维度与$\boldsymbol{b}_0$相同，即$\boldsymbol{r}^{(s)}=\left(r^{(s)}_1,r^{(s)}_2,...,r^{(s)}_{m_0}\right)^\top$。
* 基本可行解下的目标函数值可表示为$z^{(s)}=\boldsymbol{c}^\top\boldsymbol{x}^{(s)}$。
* 任意可行解$\boldsymbol{y}$下的复杂约束的值为$B\boldsymbol{y}=\sum\limits_{s=1}^p\lambda_s\boldsymbol{r}^{(s)}$。
* 任意可行解$\boldsymbol{y}$下的目标函数值为$\boldsymbol{c}^\top\boldsymbol{y}=\sum\limits_{s=1}^p\lambda_sz^{(s)}$。

&emsp;&emsp;那么原问题可拆分的主问题为：

$$
\begin{align}
(\mathcal{M})\quad\min\limits_{\lambda_1,\lambda_2,...,\lambda_p}\quad&\sum\limits_{s=1}^pz^{(s)}\lambda_s \notag \\
\text{s.t.}\quad&\sum\limits_{s=1}^p\lambda_s\boldsymbol{r}^{(s)}=\boldsymbol{b}_0 \label{constr1} \tag{1} \\
&\sum\limits_{s=1}^p\lambda_s=1 \label{constr2} \tag{2} \\
&\lambda_s\geq0,s=1,2,...,p \notag
\end{align}
$$

&emsp;&emsp;为进行后续的分析，这里需要定义问题$(\mathcal{M})$的对偶变量：

* 与约束$(\ref{constr1})$对应的对偶变量记为$\boldsymbol{\gamma}$，它的维数与$\boldsymbol{r}^{(s)}$和$\boldsymbol{b}_0$相同，即$\boldsymbol{\gamma}=(\gamma_1,\gamma_2,...,\gamma_{m_0})^\top$。
* 与约束$(\ref{constr2})$对应的对偶变量记为$\sigma$。

&emsp;&emsp;但就目前的想法还存在问题，{++一次性找出松弛问题的所有基本可行解是困难的++}，而且原问题的最优解由部分基本可行解表示出来是有可能的。所以，我们需要通过迭代的思想一次次找出新的可行解，再判断主问题的解。于是有了下面加权的主问题：

$$
\begin{align*}
(\mathcal{M}_w)\quad\min\limits_{\lambda_1,\lambda_2,...,\lambda_p,\lambda}\quad&\sum\limits_{s=1}^pz^{(s)}\lambda_s+z\lambda \\
\text{s.t.}\quad&\sum\limits_{s=1}^p\lambda_s\boldsymbol{r}^{(s)}+\lambda\boldsymbol{r}=\boldsymbol{b}_0 \\
&\sum\limits_{s=1}^p\lambda_s+\lambda=1 \\
&\lambda\geq0,\lambda_s\geq0,s=1,2,...,p
\end{align*}
$$

其中，$z$是新的基本可行解$\boldsymbol{x}_{\mathrm{new}}$得到的目标函数最优值，$\boldsymbol{r}$是$\boldsymbol{x}_{\mathrm{new}}$条件下的复杂约束的值，$\lambda$是$\boldsymbol{x}_{\mathrm{new}}$的权重。

&emsp;&emsp;那么接下来的问题是：{++我们如何把这$p$个可行解以最高效的方式陆陆续续找出来++}。一个新的有效的可行解，在添加之后，我们希望目标函数的成本是下降的。我们有这样一个下降成本计算式：

$$
\begin{equation}\label{reduce}\tag{3}
d=z-\boldsymbol{\gamma}^\top\boldsymbol{r}-\sigma
\end{equation}
$$

!!! quote "参考书籍"

    [1] Bazaraa, M. S., Sherali, H. D., and Shetty, C. M. Nonlinear Programming, Theory and Algorithms, 2nd ed. John Wiley & Sons, New York, 1993.

    [2] Castillo, E., Conejo, A., Pedregal, P., Garc´ıa, R., and Alguacil, N. Building and Solving Mathematical Programming Models in Engineering and Science. John Wiley & Sons, Inc., New York, 2001. Pure and Applied Mathematics: A WileyInterscience Series of Texts, Monographs and Tracts.

&emsp;&emsp;有了下降成本的表示，我们反过来可以定义子问题的目标函数，以此，找到一个可以使主问题成本下降的新的基本可行解。因此，我们需要把下降成本$d$表示为与决策变量相关的形式：

$$
\begin{equation}\label{f1}\tag{4}
d=\boldsymbol{c}^\top\boldsymbol{x}-\boldsymbol{\gamma}^\top B\boldsymbol{x}-\sigma
\end{equation}
$$

&emsp;&emsp;以此，我们就得到了子问题的目标函数，注意松弛问题根据分块是可以拆分为$l$个子问题的，所以对于第$i$个子问题，它的数学表达式为：

$$
\begin{align*}
\min\limits_{\boldsymbol{x}_i}\quad&(\boldsymbol{c}_i^\top-\boldsymbol{\gamma}^\top B_i)\boldsymbol{x}_i \\
\text{s.t.}\quad&A_i\boldsymbol{x}_i=\boldsymbol{b}_i \\
&\boldsymbol{x}_i\geq0
\end{align*}
$$

&emsp;&emsp;每个子问题求解得到松弛问题的一个分块$\boldsymbol{x}_i,i=1,2,...,l$，由所有子问题构成的新的基本可行解为$\boldsymbol{x}=\left[\boldsymbol{x}^\top_1|\boldsymbol{x}_2^\top|\cdots|\boldsymbol{x}_l^\top\right]^\top$。

&emsp;&emsp;最后，我们可以得到松弛问题能使得主问题下降的成本为$(\ref{f1})$式表示，如果$d\geq0$，则说明新的解无法使成本下降；如果$d<0$，则说明新的可行解可以使成本下降，可进行下一步迭代。

### 算法实践

&emsp;&emsp;上述思想就是Dantzig-Wolfe算法的核心，我们可整理出如下算法：

1. **Step 0**-(初始化)：迭代计数器$k=0$。

    &emsp;&emsp;找到{==多个==}初始解。为{==每个子问题==}随机$p^{(0)}$个成本系数$\hat{\boldsymbol{c}}^{(s)}_i,s=1,2,...,p^{(0)},i=1,2,...,l$，求解子问题：

    $$
    \begin{align*}
    \min\limits_{\boldsymbol{x}_i}\quad&\left(\boldsymbol{c}_i^{(s)}\right)^\top\boldsymbol{x}_i^{(s)} \\
    \text{s.t.}\quad&A_i\boldsymbol{x}_i^{(s)}=\boldsymbol{b}_i \\
    &\boldsymbol{x}_i^{(s)}\geq0
    \end{align*}
    $$

    &emsp;&emsp;那么，求解这$p^{(k)}$个{==松弛松弛问题==}，对于第$s=1,2,...,p^{(k)}$个松弛问题，我们可以得到：

    * $p^{(0)}$个初始的可行解$\boldsymbol{x}^{(s)},s=1,2,...,p^{(0)}$。
    * 目标函数值$z^{(s)}=\hat{\boldsymbol{c}}^{(s)}\boldsymbol{x}^{(s)}$。
    * 复杂约束的值$\boldsymbol{r}^{(s)}=B\boldsymbol{x}^{(s)}$。
    
2. **Step 1**-(求解主问题)：根据已知的可行解，求解主问题：

    $$
    \begin{align*}
    \min\limits_{\lambda_1,\lambda_2,...,\lambda_{p^{(k)}}}\quad&\sum\limits_{s=1}^{p^{(k)}}z^{(s)}\lambda_s \\
    \text{s.t.}\quad&\sum\limits_{s=1}^{p^{(k)}}\boldsymbol{r}^{(s)}\lambda_s=\boldsymbol{b}_0 \\
    &\sum\limits_{s=1}^{p^{(k)}}\lambda_s=1 \\
    &\lambda_s\geq0,s=1,2,...,p^{(k)}
    \end{align*}
    $$

    &emsp;&emsp;那么，求解这个主问题我们可以得到：

    * 当前每个子问题解的权重$\lambda_1^{(k)},\lambda_2^{(k)},...,\lambda_{p^{(k)}}^{(k)}$。
    * 其对偶变量的值$\boldsymbol{\gamma}^{(k)}$和$\sigma^{(k)}$。

3. **Step 2**-(求解子问题)：根据主问题的解，我们需要能使原问题目标函数下降的可行解。构建这些子问题：

    $$
    \begin{align*}
    \min\limits_{\boldsymbol{x}_i}\quad&(\boldsymbol{c}_i^\top-\boldsymbol{\gamma}^\top B_i)\boldsymbol{x}_i \\
    \text{s.t.}\quad&A_i\boldsymbol{x}_i=\boldsymbol{b}_i \\
    &\boldsymbol{x}_i\geq0
    \end{align*}
    $$

    &emsp;&emsp;记每个子问题的解为$\boldsymbol{x}_i^{(p^{(k)}+1)}$。那么：
    
    * 整个松弛问题的解记为$\boldsymbol{x}^{(p^{(k)}+1)}$。
    * 子问题(下降成本计算)最优值为$d^{(k)}=(\boldsymbol{c}^\top-\boldsymbol{\gamma}^\top B)\boldsymbol{x}^{(p^{(k)}+1)}-\sigma^{(k)}$。
    * 新解在原问题中的目标函数值为$z^{p^{(k)}+1}=\boldsymbol{c}^\top\boldsymbol{x}^{(p^{(k)}+1)}$。
    * 新解的复杂约束的值为$\boldsymbol{r}^{(p^{(k)}+1)}=B\boldsymbol{x}^{(p^{(k)}+1)}$。

4. **Step 3**-(收敛性检查)：

    * 如果$d^{(k)}\geq0$，那么得到了原问题的最优解：

        $$
        \boldsymbol{x}^*=\sum\limits_{s=1}^{p^{(k)}}\lambda_s^{(k)}\boldsymbol{x}^{(s)}
        $$

        算法结束。
    
    * 如果$d^{(k)}<0$，那么松弛为题的解还可以改进主问题，更新可行解的数量$p^{(k+1)}=p^{(k)}+1$。最后更新迭代计数器$k=k+1$，返回**Step 1**。
    
### 一些改进

!!! question

    牺牲求解精度，以换取更快的求解时间。

&emsp;&emsp;关于迭代出口，可以选择上下界足够接近的时候，以减少迭代次数。

&emsp;&emsp;在第$k$次迭代中，我们可以很容易从主函数的最优解$\lambda_1^{(k)},\lambda_2^{(k)},...,\lambda_p^{(k)}$中得到上界：

$$
z^{(k)}_{\mathrm{up}}=\sum\limits_{s=1}^{p^{(k)}}z^{(s)}\lambda_s^{(k)}
$$

&emsp;&emsp;接着求解所有的子问题，得到松弛问题的最优解$\boldsymbol{x}^{(k)}$和最优值$z^{(k)}_{\mathrm{sub}}$。因为松弛问题的解是不满足复杂约束的，所以，添加复杂约束之后的最优值一定是松弛问题可行域内的点，记为$\boldsymbol{x}$，且满足$B\boldsymbol{x}=\boldsymbol{b}_0$。那么对这个松弛问题就有：

$$
\begin{align*}
z^{(k)}_{\mathrm{sub}}\leq&\left(\boldsymbol{c}^\top-(\boldsymbol{\gamma}^{(k)})^\top B\right)\boldsymbol{x} \\
z^{(k)}_{\mathrm{sub}}+(\boldsymbol{\gamma}^{(k)})^\top B\boldsymbol{x}\leq&\boldsymbol{c}^\top\boldsymbol{x} \\
z^{(k)}_{\mathrm{sub}}+(\boldsymbol{\gamma}^{(k)})^\top \boldsymbol{b}_0\leq&\boldsymbol{c}^\top\boldsymbol{x}
\end{align*}
$$

&emsp;&emsp;所以，下界就可以定义为：

$$
z^{(k)}_{\mathrm{low}}=z^{(k)}_{\mathrm{sub}}+(\boldsymbol{\gamma}^{(k)})^\top \boldsymbol{b}_0
$$

!!! question

    当初始$p^{(0)}$个解，在主问题中找不到可行解。
    
&emsp;&emsp;解决办法：添加人工变量，使其一定有界，但同时也增加了问题的规模。

$$
\begin{align*}
\min\limits_{\lambda_1,\lambda_2,...,\lambda_p,\boldsymbol{v},w}\quad&\sum\limits_{s=1}^pz^{(s)}\lambda_s+M\left(\sum\limits_{j=1}^qv_j+w\right) \\
\text{s.t.}\quad&\sum\limits_{s=1}^p\lambda_s\boldsymbol{r}^{(s)}+\boldsymbol{v}-w\boldsymbol{e}=\boldsymbol{b}_0 \\
&\sum\limits_{s=1}^p\lambda_s=1 \\
&\boldsymbol{v}\geq0,w\geq0,\lambda_s\geq0,s=1,2,...,p
\end{align*}
$$

其中，$\boldsymbol{v}$是一个$m_0$长的人工决策变量，$w$是一个决策变量。$M$是一个足够大的数，因为我们期望人工变量$\boldsymbol{v},w$为$0$，找到满足复杂约束的$\boldsymbol{x}$。

## 示例与代码

!!! example "例 1.1"

    &emsp;&emsp;考虑这样一个问题：

    $$
    \begin{align*}
    \begin{array}{rrrrrr}
    \min & -4x_1 & -x_2 & -6x_3 & & \\
    \mathrm{s.t.} & -x_1 & & & \leq & -1 \\
    & x_1 & & & \leq & 2 \\
    & & -x_2 & & \leq & -1 \\
    & & x_2 & & \leq & 2 \\
    & & & -x_3 &\leq & -1 \\
    & & & x_3 &\leq & 2 \\
    &3x_1 & +2x_2 & +4x_3 & \leq & 17 \\
    \end{array} \\
    x_1,x_2,x_3\geq0\quad\quad
    \end{align*}
    $$

!!! tip "Solution"

    > 我们可以使用原始求解器得到最优解以验证我们的算法是否正确：$x_1^*=2,x_2^*=1.5,x_3^*=2$

    &emsp;&emsp;由题，我们简写以下数据：
    
    $$
    \boldsymbol{c}=(-4,-1,-6)^\top,B=\left[3,2,4\right]
    $$

    1. Step 0-初始化：$k=0,p^{(0)}=2$，所以我们需要找两个初始解出来。

        * 随机给定一个成本系数：$\hat{c}_1^{(0)}=-1,\hat{c}_2^{(0)}=-1,\hat{c}_3^{(0)}=-1$，则求解以下子问题得到第一个初始解。

            $$
            \begin{align*}
            \begin{array}{lllll}
            \min\quad -x_1 & \quad & \min\quad -x_2 & \quad & \min\quad -x_3 \\
            \mathrm{s.t.}\quad 1\leq x_1\leq 2 & \quad & \mathrm{s.t.}\quad 1\leq x_2\leq 2 &\quad & \mathrm{s.t.}\quad 1\leq x_3\leq 2 
            \end{array}
            \end{align*}
            $$

            解得$x_1^{(0)}=2,x_2^{(0)}=2,x_3^{(0)}=2$，目标函数值$z^{(0)}=-22$，且：

            $$
            r^{(0)}=B\boldsymbol{x}=(3,2,4)\cdot(2,2,2)^\top=18
            $$

        * 随机给定第二个成本系数：$\hat{c}_1^{(1)}=1,\hat{c}_2^{(1)}=1,\hat{c}_3^{(1)}=-1$，则求解以下子问题得到第二个初始解。

            $$
            \begin{align*}
            \begin{array}{lllll}
            \min\quad x_1 & \quad & \min\quad x_2 & \quad & \min\quad -x_3 \\
            \mathrm{s.t.}\quad 1\leq x_1\leq 2 & \quad & \mathrm{s.t.}\quad 1\leq x_2\leq 2 &\quad & \mathrm{s.t.}\quad 1\leq x_3\leq 2 
            \end{array}
            \end{align*}
            $$
            
            解得$x_1^{(1)}=1,x_2^{(1)}=1,x_3^{(1)}=2$，目标函数值$z^{(1)}=-17$，且$r^{(1)}=13$。
    
    2. 第一次迭代：

        * Step 1-求解主问题：

            $$
            \begin{align*}
            \min\quad&-22\lambda_1-17\lambda_2 \\
            \mathrm{s.t.}\quad&18\lambda_1+13\lambda_2\leq 17 \\
            &\lambda_1+\lambda_2=1 \\
            &\lambda_,\lambda_2\geq0
            \end{align*}
            $$

            解得$\lambda_1^{(1)}=0.8,\lambda_2^{(1)}=0.2$，其约束对应的对偶变量值为$\gamma_1^{(1)}=-1,\sigma^{(1)}=-4$。

        * Step 2-求解子问题：根据公式求解新的系数$\boldsymbol{c}_{\mathrm{new}}=(\boldsymbol{c}-\boldsymbol{\gamma}^\top B)$，那么

            $$ 
            \hat{c}_1^{(2)}=-4-(-1) \times 3 = -1, \hat{c}_2^{(2)}=-1-(-1)\times 2=1, \hat{c}_3^{(2)}=-6-(-1)\times 4=-2 
            $$

            所以求解的子问题如下：

            $$
            \begin{align*}
            \begin{array}{lllll}
            \min\quad -x_1 & \quad & \min\quad x_2 & \quad & \min\quad -2 x_3 \\
            \mathrm{s.t.}\quad 1\leq x_1\leq 2 & \quad & \mathrm{s.t.}\quad 1\leq x_2\leq 2 &\quad & \mathrm{s.t.}\quad 1\leq x_3\leq 2 
            \end{array}
            \end{align*}
            $$

            解得$x_1^{(2)}=2,x_2^{(2)}=1,x_3^{(2)}=2$，目标函数值$z^{(2)}=-21$，且$r^{(2)}=16$。
        
        * Step 3-收敛性检查：计算下降成本$d=\hat{c}_1^{(2)}\times x_1+\hat{c}_2^{(2)}\times x2 + \hat{c}_3^{(2)} \times x_3 - \sigma^{(1)} = -1 < 0$，因此需要进入下一步迭代，$p^{(1)}=2+1=3,k=1+1=2$。

    2. 第二次迭代：

        * Step 1-求解下面的主问题：

            $$
            \begin{align*}
            \min\quad&-22\lambda_1-17\lambda_2-21\lambda_3 \\
            \mathrm{s.t.}\quad&18\lambda_1+13\lambda_2+16\lambda_3\leq 17 \\
            &\lambda_1+\lambda_2+\lambda_3=1 \\
            &\lambda_,\lambda_2,\lambda_3\geq0
            \end{align*}
            $$

            解得$\lambda_1^{(2)}=0.5,\lambda_2^{(2)}=0,\lambda_3^{(2)}=0.5$，其约束对应的对偶变量值为$\gamma_1^{(2)}=-0.5,\sigma^{(2)}=-13$。
        
        * Step 2-求解子问题：

            $$ 
            \hat{c}_1^{(3)}=-2.5, \hat{c}_2^{(3)}=0, \hat{c}_3^{(3)}=-4 
            $$

            所以求解的子问题如下：

            $$
            \begin{align*}
            \begin{array}{lllll}
            \min\quad -2.5x_1 & \quad & \min\quad 0\cdot x_2 & \quad & \min\quad -4 x_3 \\
            \mathrm{s.t.}\quad 1\leq x_1\leq 2 & \quad & \mathrm{s.t.}\quad 1\leq x_2\leq 2 &\quad & \mathrm{s.t.}\quad 1\leq x_3\leq 2 
            \end{array}
            \end{align*}
            $$

            解得$x_1^{(3)}=2,x_2^{(3)}=1,x_3^{(3)}=2$，目标函数值$z^{(3)}=-21$，且$r^{(3)}=16$。
        
        * Step 3-收敛性检查：$d=-2.5\times 2-4\times 2 - (-13) = 0$，停止迭代，则计算最终结果：

            $$
            \boldsymbol{x}=\frac{1}{2}\left(\begin{array}{c} 2 \\ 2 \\2 \end{array}\right)+0\left(\begin{array}{c} 1 \\ 1 \\2 \end{array}\right)+\frac{1}{2}\left(\begin{array}{c} 2 \\ 1 \\2 \end{array}\right) = \left(\begin{array}{c} 2 \\ 1.5 \\2 \end{array}\right)
            $$
            
&emsp;&emsp;在`gurobi`和`numpy`环境：

```python
from gurobipy import GRB, Model
import numpy as np
```

1. 数据准备

    ```python
    c1 = np.array([-4])
    c2 = np.array([-1])
    c3 = np.array([-6])
    b1 = np.array([-1, 2])
    b2 = np.array([-1, 2])
    b3 = np.array([-1, 2])
    b4 = np.array([17])
    A1 = np.array([[-1], [1]])
    A2 = np.array([[-1], [1]])
    A3 = np.array([[-1], [1]])
    A4 = np.array([3])
    A5 = np.array([2])
    A6 = np.array([4])
    ```

2. 子问题准备：

    ```python
    def sub_problem(cost, A, b):
        sub1_m = Model()
        sub1_m.setParam('OutputFlag', 0)
        x = sub1_m.addMVar(1, vtype=GRB.CONTINUOUS, name='x1')
        sub1_m.setObjective(cost @ x, GRB.MINIMIZE)
        sub1_m.addConstr(A @ x <= b)
        sub1_m.update()
        sub1_m.optimize()
        return x.X
    ```

3. 主问题准备：

    ```python
    """ 主问题
    @parameters p 现有解的个数
    @parameters z_list 现有解对应的最优值
    @parameters r_list 现有解对应的r
    """
    def master_problem(p, z_list, r_list):
        master_m = Model()
        master_m.setParam('OutputFlag', 0)
        l = master_m.addMVar(p, vtype=GRB.CONTINUOUS, name="lambda")
        master_m.setObjective(np.array(z_list) @ l, GRB.MINIMIZE)
        master_m.addConstr(np.array(r_list).transpose() @ l == b4)
        master_m.addConstr(l.sum() == 1)
        master_m.update()
        master_m.optimize()
        if master_m.Status != GRB.Status.OPTIMAL:
            print("No Solution: master!")
            return None, None, None
        # 获取对偶变量的值
        dual_val = []
        for constr in master_m.getConstrs():
            dual_val.append(constr.Pi)
        return l.X, dual_val[0: -1], dual_val[-1]
    ```

4. 算法实现

    ```python
    max_iter = 400  # 最大迭代次数
    p = 2           # 现有解的个数
    # 存储中间数据
    x1_list = []
    x2_list = []
    x3_list = []
    r_list = []
    z_list = []
    # 初始化参数
    c1_list = [np.array([-1]), np.array([1])]
    c2_list = [np.array([-1]), np.array([1])]
    c3_list = [np.array([-1]), np.array([-1])]
    # 获取最初的p个解
    for i in range(p):
        x1_list.append(sub_problem(c1_list[i], A1, b1))
        x2_list.append(sub_problem(c2_list[i], A2, b2))
        x3_list.append(sub_problem(c3_list[i], A3, b3))
        z_list.append(c1 @ x1_list[i] + c2 @ x2_list[i] + c3 @ x3_list[i])
        r_list.append(A4 @ x1_list[i] + A5 @ x2_list[i] + A6 @ x3_list[i])

    # 算法迭代
    for j in range(mxa_iter):
        # 求解主问题
        l, g, s = master_problem(p, z_list, r_list)

        # 更新成本系数
        c1_list.append(c1 - g @ A4)
        c2_list.append(c2 - g @ A5)
        c3_list.append(c3 - g @ A6)

        # 求解子问题
        x1_list.append(sub_problem(c1_list[-1], A1, b1))
        x2_list.append(sub_problem(c2_list[-1], A2, b2))
        x3_list.append(sub_problem(c3_list[-1], A3, b3))

        # 收敛性检查
        d = c1_list[-1] @ x1_list[-1] + c2_list[-1] @ x2_list[-1] + c3_list[-1] @ x3_list[-1]  - s
        if d >= -1e-4:
            # 获取最终结果
            x1 = np.array([0.0])
            x2 = np.array([0.0])
            x3 = np.array([0.0])
            for i in range(p):
                x1 += (l[i] * x1_list[i])
                x2 += (l[i] * x2_list[i])
                x3 += (l[i] * x3_list[i])
            print("optimal:", x1, x2, x3, c1 @ x1 + c2 @ x2 + c3 @ x3)
            break
        
        # 更新参数
        r_list.append(A4 @ x1_list[-1] + A5 @ x2_list[-1] + A6 @ x3_list[-1])
        z_list.append(c1 @ x1_list[-1] + c2 @ x2_list[-1] + c3 @ x3_list[-1])
        p += 1
    ```

5. 补充主问题无解时的方法：上述例子在$p=1$的时候就需要启用这个算法。

    ```python
    def master_problem(p, z_list, r_list):
        M = 100000.0
        master_m = Model()
        master_m.setParam('OutputFlag', 0)
        l = master_m.addMVar(p, vtype=GRB.CONTINUOUS, name="lambda")
        v = master_m.addMVar(1, vtype=GRB.CONTINUOUS, name='v')
        w = master_m.addMVar(1, vtype=GRB.CONTINUOUS, name='w')
        master_m.setObjective(np.array(z_list) @ l + M * (v.sum() + w), GRB.MINIMIZE)
        master_m.addConstr(np.array(r_list).transpose() @ l + v - w == b4)
        master_m.addConstr(l.sum() == 1)
        master_m.update()
        master_m.optimize()
        if master_m.Status != GRB.Status.OPTIMAL:
            print("No Solution: master!")
            return None, None, None
        # 获取对偶变量的值
        dual_val = []
        for constr in master_m.getConstrs():
            dual_val.append(constr.Pi)
        return l.X, dual_val[0: -1], dual_val[-1]
    ```