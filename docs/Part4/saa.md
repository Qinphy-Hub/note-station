# Sample Average Approximation

# 问题描述

1. 抽样平均近似(Sample Average Approximation, **SAA**)方法用于解决随机优化问题，其问题描述如下：

    $$
    \begin{equation}\label{f1}\tag{1}
    \min\limits_{\boldsymbol{x}\in\mathcal{D}}\quad\mathbb{E}_{\boldsymbol{\xi}\sim\mathrm{P}}f(\boldsymbol{x},\boldsymbol{\xi})
    \end{equation}
    $$

    其中，$\boldsymbol{\xi}$是一个随机变量，服从$\boldsymbol{\xi}\sim\mathrm{P}$分布；$\boldsymbol{x}$是优化变量；$\mathcal{D}$是变量$\boldsymbol{x}$需要满足的可行域。

2. 主要考虑以下**问题情景**：

    - 期望值函数$\mathbb{E}f(\boldsymbol{x},\boldsymbol{\xi})$不能写成封闭形式，或者其值不容易计算。
    - 对于给定地$\boldsymbol{x}$和$\boldsymbol{\xi}$，函$f(\boldsymbol{x},\boldsymbol{\xi})$很容易计算。
    - 可行解集$\mathcal{D}$虽然是有限的，但非常大，因此枚举方法是不可行的。

3. 本文的**基本思想**：

	&emsp;&emsp;众所周知，许多离散优化问题是很难求解的。另一个困难是目标函数$f(\boldsymbol{x},\boldsymbol{\xi})$可能很复杂，甚至难以近似计算。有大量文献讨论随机离散优化问题，其中可行解的数量足够小，可以估计每个解$\boldsymbol{x}$的函数值$f(\boldsymbol{x},\boldsymbol{\xi})$。而本文讨论解集$\mathcal{D}$有限但足够大的情况。

	&emsp;&emsp;本文研究了随机离散优化问题的一种基于**蒙特卡洛**(Monte Carlo)模拟方法，其基本思想非常简单：生成$W$的随机样本，用相应的样本平均函数逼近期望值函数，求解得到样本平均优化问题，并且该过程重复多次，直到满足停止准则。我们讨论了<u>收敛速度</u>，<u>停止规则</u>和<u>计算复杂性</u>的过程，并提出了一个数值例子的随机背包问题。

	&emsp;&emsp;抽象平均近似法的中心思想是使用某种抽样方法把随机变量表示为确定的值，从而将不确定优化问题转化为确定性问题。也就是说，设随机变量$\boldsymbol{\xi}$抽取的样本分别为$\boldsymbol{\xi}_1,\boldsymbol{\xi}_2,...,\boldsymbol{\xi}_N$，则原问题可以转化为：

    $$
    \begin{equation}\label{f2}\tag{2}
    \min\limits_{\boldsymbol{x}\in\mathcal{D}}\quad\frac{1}{N}\sum\limits_{i=1}^Nf(\boldsymbol{x},\boldsymbol{\xi}_i)
    \end{equation}
    $$

    &emsp;&emsp;抽样平均近似法能够有效是基于以下两个性质：

    - 渐进收敛性：随着样本数量$N$​趋于无穷大，问题($\ref{f2}$)的最优解和最优值收敛于原始问题($\ref{f1}$)的最优解和最优值。
    - 易处理性：对于大多数函数$f(\boldsymbol{x},\boldsymbol{\xi})$和可行解集$\mathcal{D}$​，找到最优化问题($\ref{f2}$)的最优解和最优值，在计算上是容易处理的。

# 算法描述

## 1. 示例

&emsp;&emsp;在此以经典的随机优化问题——报童问题(The Newsvender Problem)为例。想象一个报童，每天早上去报纸厂批发一批报纸，然后拿到街上去卖。报纸批发太少，利润就少（这时可视为损失了本该赚到的钱）；但批发太多，卖不出去就亏了，虽然报纸厂会提供一定的回收服务，但还是会亏本。那么如何批发报纸能让损失最小呢？报童应该根据需求确定批发数量。但需求具有一定的不确定性，因此需要引入概率进行刻画。

&emsp;&emsp;问题建模如下：

$$
\begin{align}
\max\quad&\mathbb{E}[x_p]\label{f3-1} \tag{3-1}\\
\text{s.t.}\quad&0\leq x_s\leq\min\{y,d\}\label{f3-2} \tag{3-2}\\
&0\leq x_d\leq\max\{0,y-x_s\}\label{f3-3} \tag{3-3} \\
&x_p=x_s\cdot c_s-y\cdot c_o+x_d\cdot c_d\label{f3-4} \tag{3-4}
\end{align}
$$

其中，$d$是需求大小；$x_s$是销售额；$x_d$是滞销数量；$x_p$是利润；$y$是进货量；$c_s$是售价；$c_o$是进货成本；$c_d$为滞销回收价格。注意到$d$是市场需求量，是不确定的，为随机变量。

&emsp;&emsp;所以，上述建模的含义是：

- 目标函数($\ref{f3-1}$)：因为需求具有不确定性，我们希望收益的期望值最大。
- 约束($\ref{f3-2}$)：报纸实际的销售量不能大于进货量和实际需求量。
- 约束($\ref{f3-3}$)：滞销的报纸数量不能小于\$0\$以及不能超过进货量与销售量的差值。
- 约束($\ref{f3-4}$)：实际销售利润的计算式。

## 2. SAA方法

&emsp;&emsp;首先我们直到需求量$d$的分布，根据SAA的思想，我们可以对其进行抽样\(一般使用蒙特卡洛抽样方法\)。这样我们可以得到需求量的$N$个样本：$\{d_1,d_2,...,d_N\}$。根据这$N$个样本我们可以构建$N$个最优化问题，我们可以把这$N$个问题合并为下面这样的优化问题：

$$
\begin{equation}\label{f4}\tag{4}
\begin{split}
\max\quad&\frac{1}{N}\sum\limits_{i=1}^Nx_{p,i}\\
\text{s.t.}\quad&0\leq x_{s,i}\leq\min\{y,d_i\}  \quad\forall i\\
&0\leq x_{d,i}\leq\max\{0,y-x_{s,i}\}  \quad\forall i\\
&x_{p,i}=c_sx_{s,i}-c_oy+c_dx_{d,i} \quad\forall i
\end{split}
\end{equation}
$$

其中，$x_{p,i},x_{s,i},x_{d,i}$是第$i$个抽样对应的优化问题的优化变量；$d_i$为第$i$次抽样的值。根据SAA的理论，只要$N$​足够大，上述优化问题($\ref{f4}$)是可以近似逼近到原问题($\ref{f3-1}$)的。

## 3. 求解

&emsp;&emsp;实际上，上面的问题($\ref{f4}$)我们可以进一步化简为下式：

$$
\begin{align}
\max\quad&\frac{1}{N}\sum\limits_{i=1}^Nx_{p,i}\label{f5-1} \tag{5-1}\\
\text{s.t.}\quad&0\leq x_{s,i}\leq d_i  \quad\forall i\label{f5-2} \tag{5-2}\\
& x_{s,i}+x_{d,i}=y  \quad\forall i\label{f5-3} \tag{5-3}\\
&x_{d,i}\geq0  \quad\forall i\label{f5-4} \tag{5-4}\\
&x_{p,i}=c_sx_{s,i}-c_oy+c_dx_{d,i} \quad\forall i\label{f5-5} \tag{5-5}
\end{align}
$$

&emsp;&emsp;因为销售量$x_s$、滞销量$x_d$和进货量之间存在关系$x_s+x_d=y$，并且由于是最大化问题，$x_d$不会取负值，因此自然满足$y-x_s\geq0$​，于是有了上述问题($\ref{f5-1}$)的简单表达式。

&emsp;&emsp;那么，根据问题($\ref{f5-1}$)我们使用Gurobi求解，本示例问题取自Gurobi官网的学习材料[Solving Simple Stochastic Optimization Problems with Gurobi](https://www.gurobi.com/events/solving-simple-stochastic-optimization-problems-with-gurobi/)：

```python
from gurobipy import Model, GRB
import random

random.seed(a=100)
# 准备数据
cost    = 2     # 成本
retail  = 15    # 售价
recover = -3    # 滞销回收价
samples = 10000 # 采样数量

# 采样
sigma   = 100
mu      = 400
demand  = [max(random.normalvariate(mu, sigma), 0) for i in range(samples)]
maxrev  = max(demand) * (retail - cost)
minrev  = max(demand) * (recover - cost) + min(demand) * retail

m = Model()
# 设置为求解最大值
m.ModelSense = GRB.MAXIMIZE
# 设置优化变量(默认都是>=0)
order    = m.addVar(name='order')
profit   = m.addVars(samples, obj=1.0/samples, lb=minrev, ub=maxrev, name='profit') # 目标函数系数
sales    = m.addVars(samples, ub=demand, name='sales')  # 变量约束(5-2)
discount = m.addVars(samples, name='discount')
# 设置约束
m.addConstrs((profit[i] == -order * cost + sales[i] * retail + recover * discount[i] for i in range(samples)), name='profit')   # 约束(5-5)
m.addConstrs((sales[i] + discount[i] == order for i in range(samples)), name='demand')  # 约束(5-3)
m.update()
m.optimize()
if m.status == GRB.OPTIMAL:
    print('进货量:', order.X)
    print('总利润:', m.ObjVal)
else:
    print("NO SOLUTION!")
```

