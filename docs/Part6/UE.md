# 用户均衡(User Equilibrium, UE)

&emsp;&emsp;用户均衡模型是一种用于交通网络分析的工具。一般情况下，我们把交通网抽象为有向图$G=(V,A)$，其中$V$表示节点(node)集合，节点通常指交通网络中的路口；$A$表示弧线(arc)集合，弧线通常指两个节点之间的路段(segment,link,edge)。而交通网络分析就是研究出行者的行为对交通状态的影响，主要体现在路段流量、路径流量和OD(Origin-Destination)流量这三个层面上。最简单的做法是基于Wardrop第一原则，假设每个出行者在相应的出行OD对间能够找到出行时间最短的路径，即构成了用户均衡。

## 基本理论

&emsp;&emsp;Wardrop第一原则认为，所有出行者独立地做出令子集旅行时间最小的决策，结果形成这样的网络流状态。在相同OD对之间，所有使用路径的旅行时间相等并且最小，所有未被使用路径的旅行时间大于或等于使用路径的旅行时间。满足Wardrop第一原则的交通网络流状态，通常称为用户均衡(User Equilibrium, UE)。在用户均衡状态下，没有用户能够通过单方面的路径变更行为，来减少自己的旅行时间。在1956年，Beckmann等人提出用最优化模型来替代上述用户均衡最优化条件，从此，交通网络流问题演变为一个数学规划问题，称其为Beckmann变换式。首先来看目标函数，用户均衡模型的目标是每一条路段路阻函数的积分累加，可以写为如下形式：

$$
\begin{equation}\label{f:ue-obj}\tag{1}
	\min\quad Z=\sum\limits_{a\in A}\int_0^{x_a}t_a(x)\mathrm{d}x
\end{equation}
$$

&emsp;&emsp;然而，比较遗憾的是，这个目标函数并没有一个直观的经济学或行为学解释，只能将其看成一个严格的数学表达式以用来解决均衡问题。其中，路阻函数$t_a(x_a)$要体现出交通拥堵的规律，通常要求具备非线性，并严格地随着流量递增等性质(nonlinear, positive, and strictly increasing with flow)。最经典的路阻函数表达式是BPR(Bureau of Public Road, 美国联邦公路局)模型，即：

$$
\begin{equation}\label{f:BPR}\tag{2}
	t_a(x_a)=t_a^0\left[1+\alpha\left(\frac{x_a}{C_a}\right)^\beta\right]
\end{equation}
$$

其中，$t_a^0$表示路段$a$在自由流状态(车流量极小，车辆可以自由行驶的状态)；$C_a$是指路段$a$的通行能力；参数$\alpha,\beta$与公路性质相关，通常取值$\alpha=0.15,\beta=4$。

&emsp;&emsp;接下来，根据交通网络中路径流量、OD流量、路段流量之间的关系，构建约束条件。OD对的流量$q^{od}$与路段流量$x_a$通过路径流量$f_k^{od}$关联在一起，即在一个OD对上，所有路径流量之和应等于OD流量。即：

$$
\begin{equation}\label{f:us-constr-1}\tag{3}
	q^{od}=\sum\limits_{k\in K^{od}}f_k^{od}\quad\forall od\in P
\end{equation}
$$

&emsp;&emsp;路段和路径之间存在流量的恒等约束，即在一个路段上，对于所有通过该路段的路径，其流量之和等于该路段的流量。则：

$$
\begin{equation}\label{f:ue-constr-2}\tag{4}
	x_a=\sum\limits_{od\in J}\sum\limits_{k\in K^{od}}f_k^{od}\delta_{ak}^{od}\quad \forall a\in  A
\end{equation}
$$

其中，$\delta_{ak}^{od}$是一个二元变量，当$\delta_{ak}^{od}=1$表示在$(o,d)$对的第$k$条路径包含了路段$a$，反之不然。

&emsp;&emsp;最后，路段流量和路径流量都是非负的，即：

$$
\begin{equation}\label{f:ue-constr-3}\tag{5}
	\begin{split}
		x_a\geq&0\quad\forall a\in  A \\
		f_k^{od}\geq&0\quad\forall od\in P,\forall k\in K^{od}
	\end{split}
\end{equation}
$$

上述目标函数($\ref{f:ue-obj}$)和约束($\ref{f:us-constr-1},\ref{f:ue-constr-2},\ref{f:ue-constr-3}$)构成了{==用户均衡模型的Beckmann变换式==}。

## 经典算法

&emsp;&emsp;1975年，Leblanc等将Frank-Wolfe算法用于求解Beckmann变换式，形成了目前广泛使用的均衡交通分配求解法。Frank-Wolfe算法的思想是把目标函数近似为一个线性函数，从而求解一个线性规划问题。但是对于大型交通网络，算法就因计算量庞大而变得不实用。但交通分配问题有自己的特点，我们可以把上述线性规划问题等价为一个最短里问题，从而大大节省计算时间。下面就具体介绍UE问题的Frank-Wolfe算法。

&emsp;&emsp;首先，明确其中变量的意义，$\boldsymbol{x}$的各个分量表示路段的车流量大小$(x_a)$。我们同样对目标函数($\ref{f:ue-obj}$)在第$t$次迭代点$\boldsymbol{x}_t$处进行一阶泰勒展开：

$$
\begin{equation}\notag
	Z_L^t(\boldsymbol{x})=[\nabla Z(\boldsymbol{x}_t)]^\top\cdot\boldsymbol{x}=\sum\limits_{a\in A}t_a(x_a^t)x_a
\end{equation}
$$

&emsp;&emsp;于是，原问题被线性化为：

$$
\begin{equation}\label{f:ue-linear}\tag{7}
	\begin{split}
		\min\quad&Z_L^t=\sum\limits_{a\in  A}x_at_a(x_a^t) \\
		\text{s.t.}\quad&q^{od}=\sum\limits_{k\in K^{od}}f_k^{od}\quad\forall od\in P \\
		&x_a=\sum\limits_{od\in P}\sum\limits_{k\in K^{od}}f_k^{od}\delta_{ak}^{od}\quad \forall a\in  A \\
		&x_a\geq0\quad\forall a\in  A \\
		&f_k^{od}\geq0\quad\forall od\in P,\forall k\in K^{od}
	\end{split}
\end{equation}
$$

&emsp;&emsp;理解上式($\ref{f:ue-linear}$)的意义，其中$t_a(x_a^t)$是已知量，表示路段$a$的行驶时间。而$x_a$是未知量，表示路段$a$的车流量。显然，它们的乘积表示网络中所有车辆的行驶时间之和。因为每个路段行驶时间已固定，也就是说，这个目标函数就是在找每个OD的最短路(以时间为权重)。这样求得$\boldsymbol{y}_t$，进而得到可行下降方向$(\boldsymbol{y}_t-\boldsymbol{x}_t)$，不像单纯形法那样复杂。

&emsp;&emsp;根据Frank-Wolfe方法，现在已知下降方向$\boldsymbol{y}_t-\boldsymbol{x}_t$，接下来就需要确定最优步长。我们假设使用BPR函数($\ref{f:BPR}$)作为路阻函数，令$x_a'=x_a^t+\lambda(y_a^t-x_a^t)$，则优化步长的目标函数可写为：

$$
\begin{equation}\label{f:ue-step-obj}\tag{8}
	\begin{split}
		Z[\boldsymbol{x}_t+\lambda(\boldsymbol{y}_t-\boldsymbol{x}_t)]=&\sum\limits_{a\in  A}\int_0^{x_a'}t_a(x)\mathrm{d}x \\
		=&\sum\limits_{a\in A}t_a^0\left[x_a'+\frac{\alpha C_a}{\beta + 1}\left(\frac{x_a'}{C_a}\right)^{\beta+1}\right]
	\end{split}
\end{equation}
$$

&emsp;&emsp;根据Frank-Wolfe算法，我们得到优化步长的表达式：

$$
\begin{equation}\label{f:ue-step}\tag{9}
	\min\limits_{0\leq\lambda\leq1}\quad \sum\limits_{a\in A}t_a^0\left[x_a'+\frac{\alpha C_a}{\beta + 1}\left(\frac{x_a'}{C_a}\right)^{\beta+1}\right]
\end{equation}
$$

&emsp;&emsp;这是一个关于$\lambda$的一元非线性方程，有许多一维搜索技术可以求出驻点$\lambda^*$的值。那么最优步长可取为$0,1,\lambda^*$这三者中使得($\ref{f:ue-step}$)最小的。

## 代码实现

&emsp;&emsp;我们使用Python语言实现。交通网络图的部分使用networkx工具包，以及开源的SiouxFalls路网数据；数据计算使用numpy包；以及有优化工具的开源包scipy。在Python中安装环境如下：
```shell
!pip install networkx
!pip install numpy
!pip install scipy
```
```python
import networkx as nx
import numpy as np
from scipy.optimize import line_search
```

&emsp;&emsp;首先实现一些已知的函数。

* 路阻函数：
```python
""" 路阻函数
@parameter flow 路段的车流量
@parameter t0   自由流下的行驶时间
@parameter c    路段容量
"""
def BPR(flow, t0, c, alpha=0.15, beta=4):
    return t0 * (1 + alpha * (flow / c) ** beta)
```

* 用户均衡模型地目标函数是路阻函数的积分形式，可以显式地写出来：
```python
def INT_BPR(flow, t0, c, alpha=0.15, beta=4):
    return t0 * (flow + (alpha * c / (beta + 1)) * ((flow / c) ** (beta + 1)))
```

* 寻找最短路径的函数，由networkx提供的工具实现，具体算法为Dijskra算法。这里我们为提高算法效率做一些工程化的处理：因为Dijskra算法找一个O与D之间的最短路径，等效于找以O为起点的到其它所有点的最短路径。因此，当OD对非常多的时候，一个OD找一次是非常不划算的。我们根据OD数量设计不同的算法：
```python
# 衡量OD数量
spm = True if len(set([t[0] for t in list(ods.keys())])) >= len(G.nodes()) else False
"""
@return: 所有OD之间的最短路径 path[o][d]=list(node)
@note: 以边属性'weight'衡量路径长度。
"""
def __all_pairs_shortest_paths__(G, ods):
    if spm:
        # OD数量足够，直接找出所有最短路径：
        paths = nx.all_pairs_dijkstra_path(G, weight="weight")
        return dict(paths)
    else:
        # OD数量较少，一个OD找一次：
        paths = {}
        for r in ods.keys():
            if r[0] not in paths.keys():
                paths[r[0]] = nx.single_source_dijkstra_path(G, r[0], weight="weight")
        return paths
```

&emsp;&emsp;其次，需要准备一些数据，采用开源的[SiouxFalls路网数据](https://github.com/bstabler/TransportationNetworks/tree/master/SiouxFalls)，使用networkx整理：

* 必要的边的属性：'FFT'表示$t_0^a$，'Capacity'表示道路容量$C_a$。
* 把OD对表示为字典格式：key为`(o, d)`，其中$o$表示起始点的id，$d$表示终止点的id；value为需求的大小。
* 最终我们得到数据`G`表示SiouxFalls交通网络，`ods`表示OD需求。

&emsp;&emsp;然后需要初始化数据：
```python
arc_flow = {}       # 路段流量
ods_flow = {}       # OD分配在每个路段上的流量，可用于推导路径流量
ods_flag = False    # 是否计算OD分配在路段上的流量
iter_arc = {}       # 迭代中间值
iter_ods = {}       # 迭代中间值

# 初始化ods_flow:
def init_ods_flow():
    for r in ods.keys():
        ods_flow[r] = {}
        for e in G.edges():
            ods_flow[r][e] = 0

# 初始化中间变量
def init_temp_variables():
    for e in G.edges():
        iter_arc[e] = 0
    if ods_flag:
        for r in ods.keys():
            iter_ods[r] = {}
            for e in G.edges():
                iter_ods[r][e] = 0

# 初始化第一步，道路没有车的时候第一次流量分配，让arc_flow和ods_flow真正地初始化
def init_data():
    for e in G.edges():
        arc_flow[e] = 0
        G.edges[e]['weight'] = G.edges[e]['d']
    paths = all_pairs_shortest_paths()
    for r in ods.keys():
        path = paths[r[0]][r[1]]
        for i in range(len(path) - 1):
            arc_flow[(path[i], path[i + 1])] += ods[r]
            if ods_flag:
                ods_flow[r][(path[i], path[i + 1])] += ods[r]
    for n1, n2, data in G.edges(data=True):
        # 边'weight'属性设置为行驶时间成本
        data["weight"] = BPR(arc_flow[(n1, n2)], data["FFT"], data["C"])
```
!!! note "`ods_flag`的意义"
    根据UE的Frank-Wolfe算法，每次按照BPR计算的时间代价找最短路径。这样直接得到每个路段的流量，而其他流量(路径流量)不能直接得到。为了计算路径流量，必须记录每次迭代的中间结果，增加的时间代价。因此，在某些并不需要计算路径流量的情况，可以使用`ods_flag`规避掉这部分计算，以减小时间开销。

&emsp;&emsp;接下来，就可以根据算法实现：

1. 依据行驶时间成本，再次寻找最短路径并分配车流量(新的车流量与上一次迭代的车流量构成迭代梯度)：
```python
def shortest_path_by_flow():
    init_temp_variables()
    paths = all_pairs_shortest_paths()
    for r in ods.keys():
        path = paths[r[0]][r[1]]
        for i in range(len(path) - 1):
            if ods_flag:
                iter_ods[r][(path[i], path[i + 1])] += ods[r]
            iter_arc[(path[i], path[i + 1])] += ods[r]
```

2. 使用线搜素计算迭代下一步位置，需要实现线搜索目标函数、查找步长长度、更新路段流量：
```python
# 线搜索目标函数
def objective_step(x):
    expr = 0
    i = 0
    for _, _, data in G.edges(data=True):
        expr += (INT_BPR(x[i], data["FFT"], data["C"]))
        i += 1
    return expr

# 查找步长
def gradient_step(x):
    expr = []
    i = 0
    for _, _, data in G.edges(data=True):
        expr.append(BPR(x[i], data["FFT"], data["C"]))
        i += 1
    return expr

# 更新到下一个迭代点
def update( step):
    if ods_flag:
        for r in ods.keys():
            for e in G.edges():
                ods_flow[r][e] += (step * (iter_ods[r][e] - ods_flow[r][e]))
    for e in G.edges():
        arc_flow[e] += (step * (iter_arc[e] - arc_flow[e]))
        G.edges[e]['weight'] = BPR(arc_flow[e], G.edges[e]["FFT"], G.edges[e]["C"])
```

3. 最后实现Frank-Wolfe算法：
```python
""" Frank-Wolfe算法
@parameter eps 误差
@parameter max_iter 最大迭代次数
"""
def opt(eps=5e-5, max_iter=3000):
    init_data()
    x0 = np.array(list(arc_flow.values()))
    for i in range(max_iter):
        shortest_path_by_flow()
        direction = np.array(list(iter_arc.values())) - x0
        step_res = line_search(objective_step, gradient_step, x0, direction)
        step = step_res[0]
        update(step)
        x = np.array(list(arc_flow.values()))
        err = np.linalg.norm(x - x0, ord=2) / np.sum(x0)
        x0 = x
        if err <= eps:
            print('limited by eps. Iter =', i + 1)
            break
    return objective_step(x0)
```

&emsp;&emsp;结果如下：

```python
print(opt())
```

```shell
limited by eps. Iter = 387
4232582.202282087
```
