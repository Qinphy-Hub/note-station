# 01 导论

# 整数规划的数学描述

??? info "符号说明"

    * $\boldsymbol{x}$：$n$维列向量。整数变量。此外，$\boldsymbol{z}$也可以表示整数变量。
    * $\boldsymbol{y}$：$p$维列向量。实数变量。此外，$\boldsymbol{w}$也可以表示整数变量。
    * $\boldsymbol{A}$：$m\times n$的矩阵。表示整数变量的约束系数矩阵。
    * $\boldsymbol{G}$：$m\times p$的矩阵。表示实数变量的约束系数矩阵。
    * $\boldsymbol{c}$：$n$维行向量。对应整数变量约束的常数。
    * $\boldsymbol{h}$：$p$维行向量。对应实数变量约束的常数。

&emsp;&emsp;我们主要讨论以下三种类型的问题：

!!! question "Problem 1.1 (Integer Program)"

    &emsp;&emsp;整数规划，所有变量均为整数。

    $$
    \begin{align*}
    (\mathrm{IP})\quad\max\quad&\boldsymbol{c}\boldsymbol{x} \\
    \text{s.t.}\quad&\boldsymbol{A}\boldsymbol{x}\leq\boldsymbol{b} \\
    &\boldsymbol{x}\geq0,\boldsymbol{x}\in\mathbb{Z}^n
    \end{align*}
    $$

!!! question "Problem 1.2 (Mixed Integer Program)"

    &emsp;&emsp;混合整数规划，变量有整数有实数类型。

    $$
    \begin{align*}
    (\mathrm{MIP})\quad\max\quad&\boldsymbol{c}\boldsymbol{x}+\boldsymbol{h}\boldsymbol{y} \\
    \text{s.t.}\quad&\boldsymbol{A}\boldsymbol{x}+\boldsymbol{G}\boldsymbol{y}\leq \boldsymbol{b} \\
    &\boldsymbol{x}\geq0,\boldsymbol{x}\in\mathbb{Z}^n,y\geq0
    \end{align*}
    $$

!!! question "Problem 1.3 (Binary Integer Program)"

    &emsp;&emsp;0-1整数规划，所有变量只能在$0$和$1$之中取值。

    $$
    \begin{align*}
    (\mathrm{BIP})\quad\max\quad&\boldsymbol{c}\boldsymbol{x} \\
    \text{s.t.}\quad&\boldsymbol{A}\boldsymbol{x}\leq b \\
    &\boldsymbol{x}\in\{0,1\}^n
    \end{align*}
    $$
    
&emsp;&emsp;除此之外，另一类问题也与整数规划问题关系密切，即“组合优化问题”。

!!! question "(Combinatorial Optimization Problem)"

    &emsp;&emsp;通常我们给定一个有限集合$N=\{1,2,...,n\}$，每个$j\in N$赋予一个权值$c_j$；$\mathcal{F}$是$N$的一个可行子集。求最小权值的可行子集的问题可以表示为：

    $$
    (\mathrm{COP})\quad\min\limits_{S\subseteq N}\left\{\sum\limits_{j\in S}c_j:S\in\mathcal{F}\right\}
    $$

&emsp;&emsp;在后面的小节中，我们将会看到$\mathrm{COP}$通常可以表示为$\mathrm{IP}$或$\mathrm{BIP}$。

# 构建整数规划问题

&emsp;&emsp;将问题描述为系统的公式需要明确区分数据和变量，大致过程如下：

- 定义必要的变量。
- 使用这些变量定义一些约束，使可行点对应可行解。
- 使用这些变量来定义目标函数。

&emsp;&emsp;有时定义变量和约束会可能并不总是如线性规划问题那么简单，特别是$\mathrm{COP}$。我们需要定义一组额外的变量进行迭代。下面我们将举一些典型的例子帮助理解。

## 分配问题

!!! question "The Assignment Problem"

    &emsp;&emsp;有$n$个人可以完成$n$项工作，每个人只能被指派执行一项工作。有些人比其他人更适合某项特定的工作，因此人员$i$被分配到工作$j$会有一个成本$c_{ij}$。目标是找到成本最小的分配方式。

- 定义变量：$x_{ij}=1$表示人员$i$被分配到工作$j$，反之亦然。
- 定义约束。

    $\sum\limits_{j=1}^nx_{ij}=1$：每个人只能被指派一项工作。

    $\sum\limits_{i=1}x_{ij}=1$：每项工作只能由一个人完成。

    $x_{ij}\in\{0,1\},i=1,2,...,n,j=1,2,...,n$：自变量约束。

- 定义目标函数，即最小化分配成本：

    $$
    \min\quad\sum\limits_{i=1}^n\sum\limits_{j=1}^nc_{ij}x_{ij}
    $$

## 0-1背包问题

!!! question "The 0-1 Knapsack Problem"

    &emsp;&emsp;预算$b$，考虑投资$n$个项目，其中$a_j$是项目$j$的支出，$c_j$是项目$j$的期望收益。目标是选择一组项目使预算不超支，期望收益最大化。

- 定义变量：$x_j=1$项目$j$被选择，反之未选中。
- 定义约束。

	$\sum\limits_{j=1}^na_jx_j\leq b$：预算不能超支。

	$x_j\in\{0,1\},j=1,2,...,n$：自变量约束。

- 定义目标函数，期望收益最大化。

    $$
    \max\quad\sum\limits_{j=1}^nc_jx_j
    $$

## 集合覆盖问题

!!! question "The Set Covering Problem"

    &emsp;&emsp;给定一定数量的地点作为服务中心，每个地点的服务中心能服务的区域范围是已知的。但是在不同区域简历服务中心的成本是不同的，那么，目标是选择一组成本最低的简历服务中心地点，要求覆盖到每个区域。

&emsp;&emsp;首先，我们可以把上述问题抽象为$\mathrm{COP}$。令$M=\{1,2,...,m\}$是所有区域构成的集合，$N=\{1,2,...,n\}$表示所有潜在服务中心地点构成的集合；令$S_j\subseteq M$表示服务中心$j\in N$可服务区域集合，其建设成本为$c_j$：

$$
\min\limits_{T\subseteq N}\quad\left\{\sum\limits_{j\in T}c_j:\bigcup\limits_{j\in T}S_j=M\right\}
$$

&emsp;&emsp;接下来，我们试着把它转化为$\mathrm{BIP}$。为便于表述，我们需要先构造一个0-1关联聚矩阵(incidence matrix)$\boldsymbol{A}$，使得当$i\in S_j$时，$a_{ij}=1$否则$a_{ij}=0$。接下来，我们形式化该问题：

- 定义变量：$x_j=1$表示地点$j$被选择作为服务中心，否则未选中。
- 定义约束。

	$\sum\limits_{j=1}^na_{ij}x_j\geq 1\quad i=1,2,...,m$：每个区域至少有一个建设了的服务中心。

	$x_j\in\{0,1\}\quad j=1,2,...,n$：自变量约束。

- 定义目标函数，总的建设成本最小化。

    $$
    \min\quad\sum\limits_{j=1}^nc_jx_j
    $$

## 旅行商问题

!!! question "The Travel Salesman Problem, TSP"

    &emsp;&emsp;推销员必须访问$n$个城市，且只能访问一次，然后返回起点。从城市$i$到城市$j$所花的时间时$c_{ij}$。找出访问顺序，使其能最快完成访问。

- 定义变量：$x_{ij}=1$销售员从城市$i$到城市$j$，反之亦然。
- 定义约束。

	$\sum\limits_{j:j\neq i}x_{ij}=1\quad i=1,2,...,n$：离开城市$i$只有一次。

	$\sum\limits_{i:i\neq j}x_{ij}=1\quad j=1,2,...,n$：到达城市$i$只有一次。

	$x_{ij}\in\{0,1\}\quad i=1,2,...,n,j=1,2,...,n(i\neq j)$：自变量约束。

&emsp;&emsp;目前，这些都是分配问题的约束条件。它可能带来一组不连通图的结果，为消除这样的结果，需要一个保证连通性的约束，即**割集约束**(cut-set constraints)：

$$
\sum\limits_{i\in S}\sum\limits_{j\notin S}x_{ij}\geq1\quad S\subset N,S\neq\emptyset
$$

!!! quote "理解"

    &emsp;&emsp;$S\subset N$节点集合无论如何分割，至少存在一条边连接两个集合$S$和$S_N^c$，保证连通性。

&emsp;&emsp;还有一种方法可<u>替代上一个</u>约束，称为**旅程消除约束**(subtour elimination constraints)：

$$
\sum\limits_{i\in S}\sum\limits_{j\in S}x_{ij}\leq |S|-1\quad S\subset N,2\leq|S|\leq n-1
$$

!!! quote "理解"

    &emsp;&emsp;任意一个真子集都不会形成比环还复杂的图。因为如果图中有环，就会有一个边数大于等于$|S|$的子集。结合之前分配问题的约束条件，保证了连通性。具体我们可以通过下图理解：

    <div class="grid cards" markdown>

    -   <figure markdown="span">
        ![subtour_elimination1](./images/01-1.png){width="300"}
        <figcaption>图 1.1 真子集不能形成环</figcaption>
        </figure>

    -   <figure markdown="span">
        ![subtour_elimination2](./images/01-2.png){ width="300" }
        <figcaption>图 1.2 C中的点不满足分配问题约束</figcaption>
        </figure>
    </div>

- 定义目标函数，使得旅行时间最小化：

    $$
    \min\quad\sum\limits_{i=1}^n\sum\limits_{j=1}^nc_{ij}x_{ij}
    $$

## 不限容量的设施选址问题

!!! question "Uncapacitated Facility Location, UFL"

    &emsp;&emsp;给定一个潜在仓库地点集合$N=\{1,2,...,n\}$和客户集合$M=\{1,2,...,m\}$。假设在$j$处建设仓库的成本为$f_j$，客户$i$在仓库$j$的交付成本为$c_{ij}$。问题是选择合适的建设仓库的地点，使得建设费用和用户的交付成本之和最小。

- 定义变量：$x_j=1$。选择在$j$处建造仓库。$y_{ij}$：表示客户$i$选择仓库$j$交付需求的大小。

- 定义约束。

	$\sum\limits_{j=1}^ny_{ij}=1$：客户$i$的所有需求都必须被满足。

	$\sum\limits_{i\in M}y_{ij}\leq mx_j\quad \forall i\in M,j\in N$：单个仓库一定能满足任意客户的需求。

	$y_{ij}\geq0,x_j\in\{0,1\}$：自变量约束。

- 定义目标函数，使得所有成本最低：

    $$
    \min\quad\sum\limits_{i\in M}\sum\limits_{j\in N}c_{ij}y_{ij}+\sum\limits_{j\in N}f_jc_j
    $$

## 批量生产问题

!!! question "Uncapacitated Lot-Sizing, ULS"

    &emsp;&emsp;解决一个产品在$n$个周期内的生产计划，使得满足用户需求，且总成本最低。已知，$t\in\{1,2,...,n\}$是生产周期索引。$f_t$表示第$t$个时期生产的固定成本；$p_t$表示第$t$个时期单位生产成本；$h_t$表示$t$时期单位存储成本；$d_t$表示$t$时期的需求。

- 定义变量。

	$y_t$：表示在第$t$个时期生产产量。

	$s_t$：表示第$t$个时期的库存。

	$x_t$：表示是否在第$t$个时期进行生产。

- 约束条件。

	$s_{t-1}+y_t=d_t+s_t\quad t=1,2,...,n$：$t$时期的库存平衡约束。

	$y_t\leq Mx_t$：$M$是一个很大的值，用于控制产量上限。

    $s_0=0,s_t,y_t\geq0,x_t\in\{0,1\},t=1,2,...,n$：自变量约束。

!!! quote "理解"

    &emsp;&emsp;这里巧妙地避免了$y_tx_t$这样的二次变量。但是也存在问题，就是对$M$值的选取。

- 目标函数，使成本最小化：

    $$
    \min\quad\sum\limits_{t=1}^n(p_ty_t+h_ts_t+f_tx_t)
    $$

&emsp;&emsp;如果我们要求在最后一个周期把所有商品卖完，也即将$s_n$收紧到$0$，那么我们就可以把$y_t$的上界收紧到$y_t\leq\left(\sum\limits_{i=t}^nd_i\right)x_t$。注意，此时需要替换$s_t=\sum\limits_{i=1}^t(y_i-d_i)$。同样，目标函数改为：

$$
\sum\limits_{t=1}^n\left(c_ty_t+f_tx_t-h_t\sum\limits_{i=1}^td_i\right)
$$

其中，$c_t=p_t+h_t+h_{t+1}+...+h_n$。

## 最小费用流问题

!!! question "Minimum Cost Network Flows"

    &emsp;&emsp;给定一个有向图$G=(V,A)$，其边$(i,j)$的容量上限为$h_{ij}$，其运输成本为$c_{ij}$。另外每个顶点都有一个需求$b_i$，若$b_i>0$表示货物流入需求，若$b_i<0$表示货物运出需求，若$b_i=0$是货物中转站。问题是找到成本最小的运输方案。

&emsp;&emsp;定义节点$i$入度的集合$V_i^+=\{k:(i,k)\in A\}$，

&emsp;&emsp;定义节点$i$出度集合为$V_i^-=\{k:(k,i)\in A\}$。

- 定义变量：$x_{ij}=1$表示选择边$(i,j)$运输货物。
- 定义约束：

	$\sum\limits_{k\in V_i^+}x_{ik}-\sum\limits_{k\in V_i^-}x_{ki}=b_i\quad\forall i\in V$：中转站出度等于入度，其它点入度-出度等于需求。

	$0\leq x_{ij}\leq h_{ij},x_{ij}\in\mathbb{Z}$：自变量约束。

- 定义目标函数使得总运输成本最小：

    $$
    \min\quad\sum\limits_{(i,j)\in A}c_{ij}x_{ij}
    $$

## 最短路径问题

!!! question "Shortest Path Problem"

    &emsp;&emsp;给定两个点$s,t\in V$，边$(i,j)$的出行成本为$c_{ij}$，问题是找到一个从$s$到$t$的最小成本的路径。

- 定义变量：$x_{ij}=1$表示边$(i,j)$被选中出行。
- 约束条件。

	$\sum\limits_{k\in V_i^+}x_{ik}-\sum\limits_{k\in V_i^-}x_{ki}=1\quad i=s$：路径的起始点出度等于$1$，入度为$0$。

	$\sum\limits_{k\in V_i^+}x_{ik}-\sum\limits_{k\in V_i^-}x_{ki}=0\quad i\in V\setminus\{s,t\}$：非起始点和终止点，出度与入度差为$0$。

	$\sum\limits_{k\in V_i^+}x_{ik}-\sum\limits_{k\in V_i^-}x_{ki}=-1\quad i=t$：路径的终止点出度为$0$，入度为$1$。

	$x_{ij}\in\{0,1\}\quad(i,j)\in A$：自变量约束。

- 定义目标函数使得出行成本最低：

    $$
    \min\quad\sum\limits_{(i,j)\in A}c_{ij}x_{ij}
    $$

## 最大流问题

!!! question "Maximum Flow Problem"

    &emsp;&emsp;给定两点$s,t\in V$。一条增广路是指从$s$到$t$的一条路并附带一个值$x_{st}$，这个值取决于这条路上所有用的最小权重的边。定义边$(i,j)$的容量为$h_{ij}$，找到一个最大流从$s$到$t$。

&emsp;&emsp;为便于表述，我们添加从$t$到$s$的边，并赋予权重$h_{ts}=+\infty$。

- 定义变量：$x_{ij}=1$表示选中$(i,j)\in A$这条边。
- 定义约束。

	$\sum\limits_{k\in V_i^+}x_{ik}-\sum\limits_{k\in V_i^-}x_{ki}=0\quad i\in V$：选出增广路。

	$0\leq x_{ij}\leq h_{ij},(i,j)\in A$：自变量约束。

- 定义目标函数使得网络流最大：

    $$
    \max\quad x_{st}
    $$

# 替换表达式

&emsp;&emsp;前文已经公式化了一小部分整数规划问题，验证公式的正确性这并不太困难。这一小节我们将认识替代公式，并试着理解这些公式为什么更好。首先，我们要明确公式的意义。

!!! note "Definition 1.1 (polyhedron)"

    &emsp;&emsp;由一组线性约束的$\mathbb{R}^n$的有限子集$P=\{\boldsymbol{x}\in\mathbb{R}^n:\boldsymbol{A}\boldsymbol{x}\leq\boldsymbol{b}\}$是一个多胞体。

!!! note "Definition 1.2 (formulation)"

    &emsp;&emsp;当且仅当$X=P\cap(\mathbb{Z}^n\times\mathbb{R}^p)$时，多胞体$P\subseteq\mathbb{R}^{n+p}$是集合$X\subseteq\mathbb{Z}^n\times\mathbb{R}^p$的公式。

&emsp;&emsp;接下来，我们举三个等价表达式的例子：

- 在UFL问题中，仓库能够满足每个客户的需求的约束可以替换为：

    $$
    \sum\limits_{i\in M}y_{ij}\leq mx_j\quad\rightarrow\quad y_{ij}\leq x_j\quad\forall j \in N
    $$

    &emsp;&emsp;替换式的意义是：只要客户$i$在$j$处有需求，那么仓库$j$一定需要建立$x_j=1$。反之，如果仓库不在$j$处建立$x_j=0$，那么，没有客户会在仓库$j$处有需求。

- 同样在UFL问题中，根据需求大小的含义和限制，我们还可以添加一项约束，对原问题的解并没有影响：

    $$
    0\leq y_{ij}\leq x_j\quad \forall i\in M
    $$

- 我们也可以进一步扩展ULS公式。有时我们想知道某个时期生产的产品具体满足了哪个阶段的需求，我们可以设置一个变量为$w_{it}$，含义是第$i$个时期生产的产品满足了第$t$个时期的需求大小。那么它需要满足的约束是：

	$\sum\limits_{i=1}^tw_{it}=d_t\quad\forall t\in\{0,1,2,...,n\}$：$t$时期的需求一定会被前$t$个时期生产的商品满足。

	$w_{it}\leq d_tx_i\quad\forall i,t(i\leq t)$：用于满足$t$时期的需求的产品两不能超过$t$时期的需求量。

	$w_{it}\geq0\quad\forall i,t(i\leq t)$：自变量约束。

	&emsp;&emsp;那么注意到$y_i=\sum\limits_{t=i}^nw_{it}$，此时，目标函数需要更改为：

    $$
    \min\quad\sum\limits_{i=1}^n\sum\limits_{t=i}^mc_iw_{it}+\sum\limits_{t=1}^nf_tx_t
    $$

&emsp;&emsp;最后，我们举一个更加形象的例子。

!!! example "例 1.1"

    &emsp;&emsp;对于下图1.3，在整数规划中，其可行域是$\{(1,1),(1,2),(2,1),(2,2),(2,3),(3,1),(3,2)\}$。但在实际问题中，存在多种可能的$P$把它覆盖住。如图中的$P_1,P_2$，但我们希望得到一个刚刚好的覆盖，如图中的$P_3$，这样就可以使用线性规划求解。

    ![image](./images/01-3.png){ width="300" }
    /// caption
    图 1.3 等效覆盖的例子
    ///

&emsp;&emsp;在此之前，我们需要回顾一下优化问题中几个关键概念。

!!! note "Definition 1.3 (convex hull)"

    &emsp;&emsp;给定集合$\boldsymbol{X}\subseteq\mathbb{R}^n$，$\boldsymbol{X}$的凸包表示为$\mathrm{conv}(\boldsymbol{X})$，定义在$\boldsymbol{X}$任意有限子集$\{\boldsymbol{x}^1,\boldsymbol{x}^2,...,\boldsymbol{x}^t\}$上：

    $$
    \mathrm{conv}(\boldsymbol{X})=\left\{\boldsymbol{x}:\boldsymbol{x}=\sum\limits_{i=1}^t\lambda_i\boldsymbol{x}^i,\sum\limits_{i=1}^t\lambda_i=1,\lambda_i\geq0,\forall i=1,2,...,t\right\}
    $$

!!! tip "Proposition 1.1"

    &emsp;&emsp;如果$\boldsymbol{X}=\{\boldsymbol{x}\in\mathbb{Z}^n:\boldsymbol{A}\boldsymbol{x}\leq \boldsymbol{b}\}$，则$\mathrm{conv}(\boldsymbol{X})$是一个多胞体。

!!! note "Definition 1.4 (extreme point)"

    &emsp;&emsp;给定一个多胞体$P$，如果$\boldsymbol{x}^1,\boldsymbol{x}^2\in P$满足$\boldsymbol{x}=\lambda\boldsymbol{x}^1+(1-\lambda)\boldsymbol{x}^2(0<\lambda<1)$一定能够得到$\boldsymbol{x}=\boldsymbol{x}^1=\boldsymbol{x}^2$，那么我们称$\boldsymbol{x}$是$P$的一个极点。

!!! tip "Proposition 1.2"

    &emsp;&emsp;如果一个线性规划$\max\{\boldsymbol{c}\boldsymbol{x}:\boldsymbol{x}\in P\boldsymbol{X}\}$，其中$P$是一个多胞体，有有限个最优解，那么$P$中存在一个极点是其最优解。

&emsp;&emsp;有了上述两个结果，我们可以把IP($\max\{\boldsymbol{c}\boldsymbol{x}:\boldsymbol{x}\in \boldsymbol{X}\}$\)等价转化为LP($\max\{\boldsymbol{c}\boldsymbol{x}:\boldsymbol{x}\in\mathrm{conv}(\boldsymbol{X})\}$)。然而，在大多数情况下，我们需要大量的不等式来描述这个$\mathrm{conv}(\boldsymbol{X})$。所以，当给定整数规划的可行域$\boldsymbol{X}$的两种表达式$P_1$和$P_2$，什么时候我们可以说其中一个比另一个好？根据理想解$\mathrm{conv}(\boldsymbol{X})$有这样的性质：对于任意的$P$，都有$\boldsymbol{X}\subseteq\mathrm{conv}(\boldsymbol{X})\subseteq P$成立。我们可以使用如下定义来描述。

!!! note "Definition 1.5 (better formulation)"

    &emsp;&emsp;给定集合$\boldsymbol{X}\subseteq\mathbb{Z}^n$和$\boldsymbol{X}$的两种表达式$P_1$和$P_2$。如果$P_1\subset P_2$，我们成$P_1$是比$P_2$更好的表达式。

&emsp;&emsp;在下一章节中，我们将看到这是一个有用的定义，也是拓展为一个更有效公式的重要思想。现在，我们先根据这个定义来验证一些表达式。

!!! example "例 1.2"

    &emsp;&emsp;在UFL问题中，对每个仓库$j$，令$P_1$为约束$\sum\limits_{i\in M}y_{ij}\leq mx_j$构成的可行域，$P_2$是由$y_{ij}\leq x_j\ \forall i\in M$构成的可行域。那么，哪一个可行域表达式更好呢？

!!! tip "Solution"

    &emsp;&emsp;首先我们可以证明$P_2\subseteq P_1$。如果$(x,y)$满足约束$y_{ij}\leq x_j$，那么它一定也满足$\sum_iy_{ij}\leq mx_j$。

	&emsp;&emsp;接下来我们证明$P_2\subset P_1$，只需要找到一个符合$P_1$但不符合$P_2$的点即可。假设$n$整除$m$，$m=kn(k\geq2)$，每个仓库服务$k$个客户：$y_{ij}=1$服务客户$i=k(j-1)+1,k(j-1)+2,...,kj,j=1,2,...,n$，否则$y_{ij}=0$。$x_j=k/m,j=1,2,...,n$。这样的点位于$P_1\setminus P_2$中。具体分析：

    - 每个仓库服务$k$个客户，则$\sum\limits_iy_{ij}=k$，而$mx_j=k$。所以满足$P_1$。
    - 存在$j$使得$y_{ij}=1$，而对应的$x_j=k/m<1$。所以是不满足$P_2$的。

	&emsp;&emsp;综上，可得$P_2\subset P_1$，所以约束表达式$P_2$是优于$P_1$的。

!!! note "Definition 1.6 (projection)"

    &emsp;&emsp;给定一个多胞体$Q\subseteq\mathbb{R}^n\times\mathbb{R}^p$，$Q$在$\mathbb{R}^n$上的投影表示为$\mathrm{proj}_{\boldsymbol{x}}\boldsymbol{Q}$，定义为：

    $$
    \mathrm{proj}_{\boldsymbol{x}}\boldsymbol{Q}=\{x\in\mathbb{R}^n:(\boldsymbol{x},\boldsymbol{w})\in\boldsymbol{Q},\boldsymbol{w}\in\mathbb{R}^p\}
    $$

&emsp;&emsp;所以，对于扩展后的约束公式$\boldsymbol{Q}$，我们可以获得其在$\boldsymbol{x}$变量的子空间$\mathbb{R}^n$中的投影$\mathrm{proj}_{\boldsymbol{x}}\boldsymbol{Q}\subseteq\mathbb{R}^n$，这样我们就可以把$\boldsymbol{Q}$与其他约束公式$P\in\mathbb{R}^n$进行比较。

!!! example "例 1.3"

    &emsp;&emsp;在ULS问题中，我们把原问题的约束公式表示为$P_1$，把添加$w_{it}$变量之后的约束公式表示为$Q_2$，比较$P_1$和$Q_2$哪个更好？

!!! tip "Solution"

    &emsp;&emsp;因为$P_1$与$Q_2$的维度不一样，所以，我们使用投影的方式获得$P_2=\mathrm{proj}_{\boldsymbol{x},\boldsymbol{s},\boldsymbol{y}}\boldsymbol{Q}_2$，这样$P_2$与$P_1$就可以进行比较。

	&emsp;&emsp;首先，因为$w_{it}$决定$y_i$，而不是$y_i$决定$w_{it}$，所以，$P_2\subseteq P_1$。

	&emsp;&emsp;接下来，我们可以找到只属于$P_1$不属于$P_2$的点：$y_t=d_t,x_t=d_t/M$。分析：

    - 根据$\sum\limits_{t=i}^nw_{it}=y_i=d_i$和$w_{it}\leq d_tx_i$可知：$M\leq\sum\limits_{t=i}^nd_t$，对任意的$t$都成立，则$M\leq d_n$。
    - 根据$\sum\limits_{i=1}^tw_{it}=d_t$和$w_{it}\leq d_tx_i$可知：$M\leq\sum\limits_{i=1}^td_i$，对任意的$i$都成立。，则$M\leq d_1$。
    - 那么，可能存在$d_t>d_1$或$d_t>d_n$的情况，这就使得$x_t>1$，不满足约束。

	&emsp;&emsp;所以$P_2\subset P_1$，那么$Q_2$优于$P_1$。
