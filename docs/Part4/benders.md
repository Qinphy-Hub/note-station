# Benders Decomposition

# 问题描述

## 应用场景

&emsp;&emsp;当一个带约束优化问题存在一个易于解决的变量，即单独看这个变量的部分(其他变量都视为常数)是一个简单的问题。这时候，我们就可以使用Benders分解把这个变量分离涉及的问题单独抽离出来求解，以加快求解速度。常见的应用场景是混合**整数线性规划**(Mixed Integer Linear Programming, **MILP**)问题，我们会把连续变量和整数变量分离开来，构成一个线性规划问题和一个整数规划问题求解。

&emsp;&emsp;也即考虑如下形式的混合整数线性规划问题：

$$
\begin{equation}\notag
\begin{split}
\min\quad&\boldsymbol{c}^\top\boldsymbol{x}+\boldsymbol{f}^\top\boldsymbol{y} \\
\text{s.t.}\quad&A\boldsymbol{x}+B\boldsymbol{y}=\boldsymbol{b} \\
&\boldsymbol{x}\geq0 \\
&\boldsymbol{y}\in Y\subseteq\mathbb{R}^n
\end{split}
\end{equation}
$$

其中，$\boldsymbol{c}\in\mathbb{R}^m,\boldsymbol{f}\in\mathbb{R}^n$是成本参数向量；$\boldsymbol{x}$和$\boldsymbol{y}$分别是$m$维和$n$维的决策变量；$A_{r\times m},B_{r\times n}$是约束矩阵；$\boldsymbol{b}\in\mathbb{R}^{r\times 1}$是右端常数。

## 基本思想

&emsp;&emsp;首先，我们观察到，如果给定$\boldsymbol{y}$的值，假定为$\overline{\boldsymbol{y}}$，剩余只涉及连续变量$\boldsymbol{x}$的部分就是一个简单的**线性规划**(Linear Programming, **LP**)问题。在这里，我们称这个问题为**子问题**(Subproblem, **SP**)：

$$
\begin{align*}
(\mathcal{SP})\quad\min\quad&\boldsymbol{c}^\top\boldsymbol{x} \\
\text{s.t.}\quad&A\boldsymbol{x}=\boldsymbol{b}-B\overline{\boldsymbol{y}} \\
&\boldsymbol{x}\geq0
\end{align*}
$$

&emsp;&emsp;经过观察，实际上$\mathcal{SP}$的对偶问题在这里更好处理：

$$
\begin{align*}
(\mathcal{DSP})\quad\max\quad&\boldsymbol{\alpha}^\top(\boldsymbol{b}-B\overline{\boldsymbol{y}}) \\
\text{s.t.}\quad&A^\top\boldsymbol{\alpha}\leq\boldsymbol{c}
\end{align*}
$$

其中，$\boldsymbol{\alpha}$是自由变量(free variable)。

&emsp;&emsp;我们可以发现$\mathcal{DSP}$具有以下特点：

- $F_1$：$\mathcal{DSP}$的可行域不依赖于$\overline{\boldsymbol{y}}$的取值。
- $F_2$：根据弱对偶性，$z_{\boldsymbol{\alpha}}\leq z_{\boldsymbol{x}}$，即$\mathcal{DSP}$为$\mathcal{SP}$提供了一个下界。
- $F_3$：如果$\mathcal{SP}$和$\mathcal{DSP}$都有可行解，此时强对偶成立(<span data-type="text" style="color: var(--b3-font-color2);">线性规划问题特点，在实际问题中一定要注意这点</span>)，那么它们的最优值相等，即$z_{\boldsymbol{x}}^*=z_{\boldsymbol{\alpha}}^*$。

&emsp;&emsp;正因为有了以上这三个特点，我们把$\boldsymbol{x}$分离出去之后，就可以借助这三个性质把$\boldsymbol{x}$的影响保留下来，从而不影响原问题的最优解。我们直接来看原问题的剩余部分，这个剩余部分一般被称为**主问题**(Masterproblem, **MP**)：

$$
\begin{align}
(\mathcal{MP})\quad\min\quad&\boldsymbol{f}^\top\boldsymbol{y}+q \notag \\
\text{s.t.}\quad
&(\boldsymbol{\alpha}^i_p)^\top(\boldsymbol{b}-B\boldsymbol{y})\leq q\quad\forall i\in I \label{f1}\tag{1} \\
&(\boldsymbol{\alpha}_r^j)^\top(\boldsymbol{b}-B\boldsymbol{y})\leq0\quad\forall j\in J \label{f2}\tag{2}\\
&\boldsymbol{y}\in Y\subseteq\mathbb{R}^n \notag
\end{align}
$$

其中，$q$是反映$\mathcal{SP}$的最优目标函数值的决策变量，这样$\mathcal{MP}$的最优目标函数值就能与原问题<span data-type="text" style="color: var(--b3-font-color9);">(1)</span>保持一致，这取决于特点$F_1$。$(\boldsymbol{\alpha}_p^1,\boldsymbol{\alpha}_p^2,...,\boldsymbol{\alpha}_p^I)$和$(\boldsymbol{\alpha}_r^1,\boldsymbol{\alpha}_r^2,...,\boldsymbol{\alpha}_r^J)$分别是$\mathcal{DSP}$的可行域**极点**(extreme points)和**极射线**(extreme rays)，其索引集合分别表示为$I$和$J$。

- 约束($\ref{f1}$)：被称为Benders最优割(Benders optimality cut)。
- 约束($\ref{f2}$)：被称为Benders可行割(Benders feasible cut)，它会在$\mathcal{DSP}$最优值无界的时候出现。

&emsp;&emsp;可以看出来，Benders最优割和Benders可行割借助了问题$\mathcal{DSP}$的可行域特点去描述$\boldsymbol{x}$的特征，所以它具有的特征$F_1$是非常好的性质。其次，$F_2$也是关键特征，由它构建了两种割。这样构建的$\boldsymbol{y}$的可行域就是由$\boldsymbol{x}$划定的。$\mathcal{MP}$与原问题是等价的，具体证明参考专业的书籍。

&emsp;&emsp;现在的问题是，一个可行域的所有极点就已经很难找到了，不然线性规划用穷举法就行。要直接得到所有的Benders最优割和可行割是一件很困难的事。并且对于$\boldsymbol{y}$新的可行域，也并不一定需要把所有的割找到才能找到才能搜索到最优解。因此，我们采用迭代法，通过搜索不断压缩上下界，得到最优值。

## 算法步骤

1. [**Step 1**(初始化)] - 我们将原问题拆分为一个子问题和一个主问题，首先记初始主问题为：

    $$
    \begin{align*}
    (\mathcal{MP}_0)\quad\min\quad&\boldsymbol{f}^\top\boldsymbol{y}+q \\
    \text{s.t.}\quad
    &(\boldsymbol{\alpha}^i_p)^\top(\boldsymbol{b}-B\boldsymbol{y})\leq q\quad\forall i\in I=\emptyset \\
    &(\boldsymbol{\alpha}_r^j)^\top(\boldsymbol{b}-B\boldsymbol{y})\leq0\quad\forall j\in J=\emptyset \\
    &\boldsymbol{y}\in Y\subseteq\mathbb{R}^n
    \end{align*}
    $$

    &emsp;&emsp;然后，明确子问题的对偶问题，即上述的$\mathcal{DSP}$。

    &emsp;&emsp;最后，初始化必要参数：

    - 这个问题的特点是其可行域与$\overline{\boldsymbol{y}}$的取值无关，所以我们可以令$\overline{\boldsymbol{y}}_0=\boldsymbol{0}$(也可以random）；
    - 索引集合$I=\emptyset,J=\emptyset$；
    - 全局上下界$UB=+\infty,LB=-\infty$；
    - 迭代计数器$t=0$。

2. [**Step 2**(计算对偶问题)] - 现在已知$\overline{\boldsymbol{y}}_t$，带入并求解$\mathcal{DSP}$。

    - 若解是极点，记为$\boldsymbol{\alpha}_p^t$，构建Benders最优割，将其添加到$\mathcal{MP}_t$中，即$I=\{t\}\cup I$。

    - 若解是极射线，记为$\boldsymbol{\alpha}_r^t$，构建Benders可行割，添加到$\mathcal{MP}_t$中，即$J=\{t\}\cup J$。
    - 同时得到子问题的目标函数最优值$z_{\mathcal{DSP}}^*$，更新上界：

    $$
    \begin{equation}\tag{3}\label{ub}
    UB=\boldsymbol{f}^\top\overline{\boldsymbol{y}}_t+z_{\mathcal{DSP}}^*
    \end{equation}
    $$

    &emsp;&emsp;此时，已经得到新的主问题，记为$\mathcal{MP}_{t+1}$。

3. [**Step 3**(求解主问题)] - 求解主问题$\mathcal{MP}_{t+1}$，可以得到其解$\overline{\boldsymbol{y}}_{t+1}$和$q$。此时，可以更新下界：

    $$
    \begin{equation}\tag{4}\label{lb}
    LB=z_{\mathcal{MP}_{t+1}}^*=\boldsymbol{f}^\top\overline{\boldsymbol{y}}_{t+1}+q
    \end{equation}
    $$

    &emsp;&emsp;如果此时有$UB=LB$(<span data-type="text" style="color: var(--b3-font-color2);">不要求精确，要求速度，可设置最小差距</span>$\varepsilon$)，则说明算法收敛，得到最优值$z_{\mathcal{MP}_{t+1}}^*$和最优解$\overline{\boldsymbol{y}}_{t+1}$，将$\overline{\boldsymbol{y}}_{t+1}$带入问题$\mathcal{SP}$即可得到最优解$\boldsymbol{x}$。

    &emsp;&emsp;反之，则回到Step2。

# 问题实例

> 本文示例均来自[MatrixOptim网站](https://bookdown.org/edxu96/matrixoptim/benders-for-standard-milp.html)。

## 1. 示例

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

## 2. 求解

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

## 3. 代码

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