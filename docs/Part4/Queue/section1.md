# 导论


&emsp;&emsp;排队论通过建立数学模型分析服务需求与服务资源之间的动态关系，核心目标是资源配置、提升服务效率、减低成本并改善用户体验。一般情况下，我们关注这三类指标：顾客的等待时间，积累的顾客数量以及服务台的空闲时间。这将在下一个章节具体介绍。为了描述一个排队系统，我们需要明确一个排队系统的基本特征。

# 1.1 排队系统的特征

&emsp;&emsp;首先需要强调，大多数排队系统都含有随机变量，我们需要借助概率论相关方法求解问题，因此，在<u>大多数时候我们都假设排队系统是</u>​**<u>平稳的</u>**(stationary)。平稳在此处含义是指一个排队系统的长期统计特性(如平均队列长度、平均等待时间)<u>不随时间变化</u>。而**非平稳的**(nonstationary)排队系统就相对复杂，比如，餐厅在吃饭时间的顾客数会比其他时间多。

## 1.1.1 顾客的到达过程

&emsp;&emsp;顾客到达往往是随机的，因此有必要知道两个顾客到达的<u>时间间隔</u>的概率分布，最常见的是**泊松过程**(Poisson process)。但是还有以下因素需要单独考虑：

- 顾客是<u>单独</u>到来还是<u>成批</u>到来。

    严格来说，只要时间段的长度足够短，任意时间段都只有一个顾客到来，那么称顾客是单独到来的。这种情况是最常见的，直观上只要是陆陆续续到达的顾客都可视为独立到来。反之，成批到来就是同时会到来多个顾客。比如，旅游大巴会一次性把多个顾客送到旅游目的地。

- 顾客到来的<u>数量是否有限</u>。

&emsp;&emsp;顾客数量无限是大部分情况，举个有限的例子：工厂设备按照某概率出现故障，但其数量一定不会超过设备总数。

&emsp;&emsp;还有一种因素在现实中存在但我们几乎不考虑，我们<u>假设顾客达到是独立的</u>，而非非独立的。顾客到达非独立是指接受服务顾客对顾客达到数没有影响，也即回头客的情况。

## 1.1.2 服务过程

&emsp;&emsp;很多情况系统为顾客提供的服务时间也是随机的，因此需要一个概率分布来描述服务员为顾客服务的<u>时间序列</u>。同样，我们还有可能需要考虑如下因素：

- 服务是一对一的还是成批的。

    一对一也是常见的情况，这里举例成批服务的情况：导游同时带领多个游客，火车载有多个乘客。

- 是否是**状态相依服务**(state-dependent service)：服务时间依赖于等待的顾客数。

    如果队列越来越长，服务员可能会加速工作，效率提高；但也有可能变得慌乱，效率降低。

&emsp;&emsp;但是，为了简化排队模型，服务过程很多时候几乎不会考虑上述两类因素。

## 1.1.3 服务台与通道数量

&emsp;&emsp;服务台的数量是一个排队系统的重要特征，这涉及到关键的成本问题。通道数量也同样重要，下图(1.1)展示了几种典型的场景。

![service](./images/01-1.png){ width="600" }
/// caption
图 1.1 几种典型的排队模式
///

(a) 单台单队：一个服务台一个队列。

(b) 多对多台并联：超市收银台。

(c) 单队多台并联：机场的托运服务。

## 1.1.4 排队规则

&emsp;&emsp;排队规则是指当队列形成时，服务员选择顾客服务的方式。常见有以下几种：

- **先到先服务**(first come first served, **FCFS**)：最常见的规则。
- **后到先服务**(last come first served, **LCFS**)：库存系统中，后到的货物往往更容易被拿到。
- **随机服务**(random selection for service, **RSS**)：随机选择队列中的顾客，与顾客到达时间无关。
- **处理器共享**(processor sharing, **PS**)：同时服务所有顾客，但顾客越多，处理速度越慢。
- **轮询**(polling)：先服务第一个队列的顾客，让后服务第二列顾客。红绿灯就是一种轮询系统。
- **优先权服务**(priority service, **PR**)：比如，军人优先通道。

&emsp;&emsp;在优先规则中，还存在两种规则：(1) **抢占**(preemptive)规则会使优先级较高的用户打断优先级低的服务；(2) **非抢占**(nonpreemptive)规则只会使优先级更高的用户插队到最前面。

## 1.1.5 系统容量

&emsp;&emsp;在某些系统中，顾客等待的物理空间受限，因此当队列达到限制长度时，系统不允许其他顾客进入，这被称为有限排队场景。换句话说，当队列长度达到上限时，新到达的顾客将被迫放弃排队。

&emsp;&emsp;因系统容量而导致顾客离开的情况是很多经典模型会考虑的，但还有很多顾客选择离去的情况会大大增加模型的复杂程度：

- 如果队列太长，顾客会决定不排队，如果这发生在到达时，则称该顾客**止步**(balked)；
- 但如果顾客决定排队，但过段时间失去耐心，决定离开，则称该顾客**中途退出**(reneged)；
- 如果是多个队列，顾客从一个队列换去另一个队列，则称该顾客**换队**(jockey)。

&emsp;&emsp;虽然，有研究也会考虑这些因素，但本书经典模型中不会考虑到。

## 1.1.6 服务阶段的数量

&emsp;&emsp;排队系统可以只有一个服务阶段，也可以有多个服务阶段。比如，体检就是一个多阶段的排队系统。加之，排队通道的类型，又会形成下图<span data-type="text" style="color: var(--b3-font-color9);">(1.2)</span>两种典型的排队系统：

![multi-service](./images/01-2.png){ width="600" }
/// caption
图 1.2 几种多阶段服务排队模式
///

(a) 单队多台串联：医院挂号到看病到拿药。

(b) 多台混合：医院体检。

# 1.2 符号表示

&emsp;&emsp;现代常用的表示法是依据英国数学家肯德尔(Kendall)的研究发展而来的。排队过程用一组符号和斜线来描述：`A/B/X/Y/Z`。

$A$：到达时间间隔分布。

$B$：服务时间分布。

> $A$和$B$所代表的两种分布，常见有如下的情况：
>
> - $M$：指数分布。
> - $D$：确定性分布。
> - $E_k$：$k$阶埃尔朗分布($k=1,2,...$)。
> - $H_k$：$k$阶超指数分布($k=1,2,...$)。
> - $PH$：阶段型分布。
> - $G$：一般分布。
>
> 其中，$M$体现了指数分布的**马尔科夫性**(Markovian)或者**无记忆性**(memoryless)。同时，如果顾客到达是泊松过程，那么其到达时间间隔分布也呈指数分布。所以，若$A=M$，则说明顾客到达呈泊松流。这些性质都会在后续章节中证明。

$X$：并行服务台数量。取值为$1,2,...,\infty$。

$Y$：系统容量。取值为$1,2,...,\infty$。

$Z$：排队规则。取值有FCFS、LCFS、RSS、PR、GD(一般规则)。

&emsp;&emsp;在许多情况下，只使用前$3$个符号：如果系统容量没有限制，即$Y=\infty$，则可省略系统容量的符号。若排队规则是先到先服务，即$Z=FCFS$，则可省略排队规则符号。

!!! abstract "总结"

    &emsp;&emsp;可以看出这个表示法涵盖了排队系统大部分特征，除了默认不考虑的几个因素以外，还存在一些因素没有表示出来。如果模型没有<u>单独说明</u>，这些因素也是不考虑的：

    * 顾客是否单独到来，默认是单独到来。
    * 顾客是否有限，默认是无限的。
    * 服务通道的数量，在多服务台的情况下，这个表示法也没有体现。这需要在建模时说明队列情况。
    * 系统容量虽然体现了超出队列长度顾客会离开队列的情况，但没有说明其他离队情况，默认不考虑其他意外情况。

# 1.3 性能指标

## 1.3.1 利特尔法则

&emsp;&emsp;在排队论中，最广泛使用的一个基本关系式是由利特尔法则(Little's law)，也称Little定理，给出的。它给出了$3$个基本变量之间的关系：顾客到达系统的**平均速率**​$\lambda$、单个顾客在系统中花费的**平均时间**​$W$和系统中**顾客的平均数**​$L$。给定$3$个量中的两个就可以推算出第三个基本量。假定有一个排队系统，设$A^{(k)}$表示顾客$k$进入系统的时间，它是有序的，即$A^{(k+1)}\geq A^{(k)}$。设$A(t)$为$t$时刻前累计到达系统的顾客数；设$W^{(k)}$为顾客$k$在系统中花费的时间；设$N(t)$是$t$时刻系统中的顾客数。那么$\forall t\geq0$，满足$A^{(k)}\leq t$且$A^{(k)}+W^{(k)}\geq t$的$k$的个数与$N(t)$的值相等。当极限存在时，极限定义如下：

$$
\begin{equation}\tag{1.1}\label{f1.1}
\lambda\triangleq\lim\limits_{k\rightarrow\infty}\frac{A^{(k)}}{t},\quad W\triangleq\lim\limits_{k\rightarrow\infty}\frac{1}{k}\sum\limits_{i=1}^kW^{(i)},\quad L\triangleq\lim\limits_{T\rightarrow\infty}\frac{1}{T}\int_0^TN(t)\mathrm{d}t
\end{equation}
$$

其中，极限$\lambda$时长期**平均到达率**，极限$W$是每个顾客在系统中花费的长期**平均时间**，极限$L$是系统中的长期**平均顾客数**。

!!! danger "定理 1.1 (利特尔法则)"

    如果上式$(\ref{f1.1})$中的极限$\lambda$和$W$存在并且是有限的，则存在极限$L$，且：

    $$ L=\lambda W $$

!!! quote "证明 (不严谨)"

    &emsp;&emsp;令$S(t)$为系统在$t$时刻前所有顾客在系统中的逗留时间，定义为：

    $$
    S(t)\triangleq\int_0^tN(x)\mathrm{d}x
    $$

    &emsp;&emsp;设$R(t)$为在$t$时刻，仍还在系统中的顾客的剩余服务时间之和，定义为：

    $$
    R(t)\triangleq\sum\limits_{k=1}^{A(t)}W^{(k)}-S(t)
    $$

    &emsp;&emsp;按照上述定义，记$\lambda_t=\frac{A(t)}{t}$，则$\lambda=\lim\limits_{t\rightarrow\infty}\lambda_t$；记$W_t=\frac{1}{A(t)}\sum\limits_{k=1}^{A(t)}W^{(k)}$，则$W=\lim\limits_{t\rightarrow\infty}W_t$；记$L_t=\frac{1}{T}\int_0^tN(x)\mathrm{d}x=\frac{S(t)}{t}$，则$L=\lim\limits_{t\rightarrow\infty}L_t$。根据定义可作如下推导：

    $$
    \begin{align*}
    \lambda_tW_t=&\frac{A(t)}{t}\cdot\frac{1}{A(t)}\sum\limits_{k=1}^{A(t)}W^{(k)} \\
    =&\frac{1}{t}\sum\limits_{k=1}^{A(t)}W^{(k)} \\
    =&\frac{S(t)+R(t)}{t} \\
    =&L_t+\frac{R(t)}{t}
    \end{align*}
    $$

	&emsp;&emsp;由于，在稳态下服务时间一定是有限的，即$\lim\limits_{t\rightarrow\infty}\frac{R(t)}{t}=0$，那么：

    $$
    \begin{align*}
    \lim\limits_{t\rightarrow\infty}\lambda_tW_t=&\lim\limits_{t\rightarrow\infty}L_t+\lim\limits_{t\rightarrow\infty}\frac{R(t)}{t}
    \end{align*}
    $$

	根据他们的定义公式$(\ref{f1.1})$，即可得证。

## 1.3.2 一般结果

&emsp;&emsp;在介绍特定模型之前，首先看一下`G/G/1`​和`G/G/c`模型的一般结果。其中$G$表示的是一般的概率分布，也就是说这些结果适用于任何概率分布的情况。下面我们给出一些关键符号及其定义：

- $\lambda$：顾客到达系统的平均速率。
- $S$：表示服务时间的随机变量。
- $\mu$：每个服务台为顾客提供服务的平均速率，也即单位时间内完成服务的顾客数。定义为:

    $$
    \begin{equation}\tag{1.2}
    \mu\triangleq\frac{1}{E[S]}
    \end{equation}
    $$

- $c$：服务台的数量。
- $r$：排队系统的**输入负荷**(offered load)，定义为：

    $$
    \begin{equation}\tag{1.3}
    r\triangleq\frac{\lambda}{\mu}=\lambda E[S]
    \end{equation}
    $$

- $\rho$：衡量流量拥塞程度的指标，即**流量强度**(traffic intensity)；或称为**服务台利用率**(server utilization)，因它等于输入负荷除以服务台的数量，表示单位时间内每个服务台的平均工作量。其定义为：

    $$
    \begin{equation}\tag{1.4}
    \rho\triangleq\frac{\lambda}{c\mu}
    \end{equation}
    $$

- $T_s,T_q$：顾客在系统($s$)、队列($q$)中花费时间的随机变量。排队系统耗费的时间只与排队时间和服务时间有关，因此，有关时间的随机变量之间存在关系：

    $$
    \begin{equation}\tag{1.5}
    T_s=T_q+S
    \end{equation}
    $$

- $W_s,W_q$：顾客在系统($s$)、队列($q$)中花费的平均时间，它们的定义为：

    $$
    \begin{equation}\tag{1.6}
    W_q\triangleq E[T_q],\quad W_s\triangleq E[T_s]
    \end{equation}
    $$

- $N_s,N_q$：系统($s$)、队列($q$)中顾客数的随机变量。
- $L_s,L_q$：系统($s$)、队列($q$)中的平均顾客数。设$p_n=\mathrm{Pr}\{N=n\}$表示<u>系统中</u>存在$n$个顾客的<u>稳态概率</u>，那么对于有$c$个服务台的系统，$L_s$和$L_q$可以表示如下：

    $$
    \begin{equation}\tag{1.7}
    L_s\triangleq E[N_s]=\sum\limits_{n=0}^\infty np_n,\quad L_q\triangleq E[N_q]=\sum\limits_{n=c+1}^\infty(n-c)p_n
    \end{equation}
    $$

&emsp;&emsp;根据顾客花费的时间变量之间的关系式<span data-type="text" style="color: var(--b3-font-color9);">(1.5)</span>和顾客花费时间的指标的定义式<span data-type="text" style="color: var(--b3-font-color9);">(1.6)</span>，我们可以推导出如下结论：

$$
\begin{equation}\tag{1.8}
\begin{split}
W_s=E[T_s]=&E[T_q+S] \\
=&E[T_s]+E[S] \\
=& W_q+\frac{1}{\mu}
\end{split}
\end{equation}
$$

&emsp;&emsp;另外，分别对系统($s$)和队列($q$)应用利特尔法则，即可得到$L_s=\lambda W_s,L_q=\lambda W_q$。那么，结合上一个结论<span data-type="text" style="color: var(--b3-font-color9);">(1.8)</span>我们可以得到一个新的结论：

$$
\begin{equation}\tag{1.9}
\begin{split}
L_s=\lambda W_s=&\lambda\left(W_q+\frac{1}{\mu}\right) \\
=&\lambda W_q+\frac{\lambda}{\mu} \quad\quad\text{by(1.3)}\\
=&L_q+r
\end{split}
\end{equation}
$$

&emsp;&emsp;对于单个服务台的系统($c=1$)，此时$r=\rho$，上式(1.9)会有特定的形式：

$$
\begin{align*}
\rho=r=L_s-L_q=&\sum\limits_{n=1}^\infty np_n-\sum\limits_{n=1}^\infty(n-1)p_n \\
=&\sum\limits_{n=1}^\infty p_n \\
=&1-p_0
\end{align*}
$$

&emsp;&emsp;可以证明，要使稳态存在就必须有$\rho<1$或$\lambda<c\mu$。也就是说顾客进入系统的平均速率必须严格小于系统的最大平均服务速率。当$\rho>1$时，顾客到达的平均速率大于接受服务的平均速率，随着时间流逝，队列变得越来越长。因为队列一直增长，所以没有稳态。当$\rho=1$时，顾客到达速率恰好等于最大服务率。这种情况下，除非顾客到达时间和服务台服务时间都是固定且合理安排，否则将不会有稳态。综上，如果知道平均到达率和平均服务速率，则可以找到满足$\rho=\frac{\lambda}{c\mu}<1$的$c$的最小值，保证稳态所需的并行服务台数的最小值。

!!! note "总结"

    &emsp;&emsp;总结来说，我们需要知道$\lambda,\mu,c,p_n,n=0,1,...$，就可以得到$r,\rho,L_s,L_q,W_s,W_q$这些评价指标。

‍
