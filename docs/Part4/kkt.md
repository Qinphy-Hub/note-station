# KKT条件

&emsp;&emsp;Karush-Kuhn-Tucker(KKT)条件是非线性规划最优解的**必要条件**。它将拉格朗日乘子法所处理的等式约束优化问题拓展至不等式约束优化问题。在实际应用上，KKT条件一般不存在代数解。
首先，我们形式化地描述不等式约束优化问题。给定一个目标函数$f:\mathbb{R}^n\rightarrow\mathbb{R}$，我们希望找到$\boldsymbol{x}\in\mathbb{R}^n$，在满足约束条件$g(\boldsymbol{x})\leq0$的前提下，使得$f(\boldsymbol{x})$有最小值。这个优化问题记为：

$$
\begin{align}
	\min\quad&f(\boldsymbol{x}) \notag \\
	\text{s.t.}\quad&g(\boldsymbol{x})\leq0 \label{f:neq-constrain} \tag{1}
\end{align}
$$

称不等式($\ref{f:neq-constrain}$)为**原始可行性**(primal feasible)，据此我们定义可行域(feasible region)为$K=\{\boldsymbol{x}\in\mathbb{R}^n|g(\boldsymbol{x})\leq0\}$。

&emsp;&emsp;假设$\boldsymbol{x}^*$为满足条件的最优解，它可能有两种情况：

* $g(\boldsymbol{x}^*)<0$：此时最优解$\boldsymbol{x}^*$在可行域$K$的内部，称为內部解(interior solution)。这时候约束条件($\ref{f:neq-constrain}$)是无效的(inactive)。我们直接对目标函数求偏导就能得到驻点。
* $g(\boldsymbol{x}^*)=0$：此时最优解$\boldsymbol{x}^*$在可行域$K$的边界上，称为边界解(boundary solution)，此时约束条件是有效的(active)。单可以发现这时候只有等式约束，可以使用拉格朗日乘子法求解。

&emsp;&emsp;我们希望把上述两种情况统一起来，所以，首先我们希望把不等式约束也能写成拉格朗日函数的形式(见上一章节[拉格朗日乘子法](./lagrange.md))。

&emsp;&emsp;不论是内部解还是边界解，$\lambda*g(\boldsymbol{x})=0$一定成立，称为**互补松弛**(complementary slackness)：

1. 对于內部解，只需要对$f(\boldsymbol{x})$求偏导，只有$\lambda=0$，才能使得$L(\boldsymbol{x},\lambda)$求偏导等效于$f(\boldsymbol{x})$求偏导。
2. 对于边界解，不等式约束已经转化为了等式约束，所以拉格朗日函数成立。那么$g(\boldsymbol{x})=0$，则$\lambda g(\boldsymbol{x})=0$成立。除此之外，我们还能知道$\lambda\geq0$，这称为对偶可行性(dual feasible)：在上一小节(拉格朗日乘子法)中，分析到$\nabla f(\boldsymbol{x})$与$\nabla g(\boldsymbol{x})$方向要么相同，要么相反。而在这里，当我们确定了$\boldsymbol{x}^*$不在$g(\boldsymbol{x})<0$区域中，那么，意味着$\nabla f(\boldsymbol{x})$一定指向$g(x)>0$的区域。而梯度是指向函数值增长的方向，$g(\boldsymbol{x})$也是指向$g(\boldsymbol{x})>0$区域，这说明$\nabla f(\boldsymbol{x})$与$\nabla g(\boldsymbol{x})$反向。那么，根据$\nabla f(\boldsymbol{x})+\lambda g(\boldsymbol{x})=0$可知，$\lambda\geq0$。

&emsp;&emsp;根据上述信息，整理即可得到KKT条件：

$$
\begin{align*}
	\nabla_{\boldsymbol{x}}L=&\nabla f+\sum\limits_{i=1}^m\lambda_i\nabla g=0 \\
	g_i(\boldsymbol{x})\leq&0 \\
	\lambda_i\geq&0 \\
	\lambda_i g_i(\boldsymbol{x})=&0\quad i=1,2,...,m
\end{align*}
$$

&emsp;&emsp;但遗憾的是，如果最优点$\boldsymbol{x}^*$不满足**约束规范**(constraint qualification)或者**正则条件**(regularity condition)，KKT条件在$\boldsymbol{x}^*$处可能**不成立**！我们先思考可能的问题出在哪里？在上面单个不等式约束的分析中，$\nabla f$的方向与$\nabla g$的方向相反，但这一切的分析是基于该约束条件对可行域是有影响的。这样我们讨论$\nabla f(\boldsymbol{x})$与$\nabla g(\boldsymbol{x})$的方向关系才有意义。下面这个例子就说明了这一点

!!! example "考虑下面这个不等式约束的最优化问题"
    $$
	\begin{align}
			\min\quad&x_1 \notag \\
			\text{s.t.}\quad&x_2+(1-x_1)^3\leq0 \label{f:kkt-regular} \tag{2} \\
			&x_1\geq0,x_2\geq0 \notag
	\end{align}
    $$

	&emsp;&emsp;这个问题的最优解是$(1,0)$，它似乎不满足KKT条件？

!!! quote "Solution"
	&emsp;&emsp;先构造拉格朗日函数：

	$$ L(x_1,x_2,\lambda_1\lambda_2,\lambda_3)=x_1+\lambda_1[x_2-(1-x_1)^3]+\lambda_2(-x_1)+\lambda_3(-x_2) $$

    ![示例函数](./images/kkt-exa.png){ width="300" }
    /// caption
    图 (1). 示例函数
    ///
	
	&emsp;&emsp;只看对$x_1$的偏导就可以看出问题：

	$$ \frac{\partial L}{\partial x_1}=1-3\lambda_1(1-x_1)^2-\lambda_2=0 $$

	&emsp;&emsp;将最优值$(1,0)$带入上式可以得到$\lambda_2=-1$，然而根据对偶可行性$\lambda_2\geq0$矛盾。

	&emsp;&emsp;从图(1)可以看出，最优点在约束条件($\ref{f:kkt-regular}$)的边界上。观察到$x_1\geq0$对于可行域是没有作用的，而该约束的梯度方向正好与$\nabla f$同向。而$x_1\geq0$约束对应的$\lambda_2<0$，不仅是因为与$\nabla f$同向，也在于其他两个约束的梯度恰好与$\nabla f$正交(线性相关)。

&emsp;&emsp;下面介绍一个常用的约束规范。**线性独立约束规范**(Linear Independence Constraint Qualification, LICQ)：

&emsp;&emsp;记$\boldsymbol{x}^*$为最优解，如果所有$g_i(\boldsymbol{x}),i=1,2,...,m$连续，在$\boldsymbol{x}^*$处所有有效约束的梯度构成一组线性独立的向量，则KKT条件在$\boldsymbol{x}^*$处一定成立。

&emsp;&emsp;观察上述例子的约束条件($\ref{f:kkt-regular}$)，计算它们的梯度：

$$
\begin{equation*}
	\begin{split}
		&\nabla (x_2+(1-x_1)^3)=[0,1] \\
		&\nabla (-x_2) = [0, -1]
	\end{split}
\end{equation*}
$$

&emsp;&emsp;显然，$[0,1]+[0,-1]=[0, 0]$，这两个梯度线性相关，不满足LICQ约束规范。

&emsp;&emsp;最后，结合朗格朗日乘子法和KKT条件，我们把问题拓展至多个不等式和等式约束的情况。考虑标准约束优化问题：

$$
\begin{equation}\notag
	\begin{split}
		\min\quad&f(\boldsymbol{x}) \\
		\text{s.t.}\quad&g_i(\boldsymbol{x})=0\quad i=1,2,...,m \\
		&h_j(\boldsymbol{x})\leq0\quad j=1,2,...,p
	\end{split}
\end{equation}
$$

&emsp;&emsp;定义拉格朗日函数：

$$
\begin{equation}\notag
	L(\boldsymbol{x},\boldsymbol{\lambda},\boldsymbol{\mu})=f(\boldsymbol{x})+\sum\limits_{i=1}^m\lambda_ig_i(\boldsymbol{x})+\sum\limits_{j=1}^p\mu_jh_j(\boldsymbol{x})
\end{equation}
$$

其中，$\lambda_j$是对应$g_i(\boldsymbol{x})$的拉格朗日乘数；$\mu_j$是对应$h_j(\boldsymbol{x})$的KKT乘数。

&emsp;&emsp;KKT条件包括：

$$
\begin{equation}\notag
	\begin{split}
		\nabla_{\boldsymbol{x}}L=&0 \\
		g_i(\boldsymbol{x})=&0\quad i=1,2,...,m \\
		h_j(\boldsymbol{x})\leq&0 \\
		\mu_j\geq&0 \\
		\mu_jh_j(\boldsymbol{x})=&0\quad j=1,2,...,p
	\end{split}
\end{equation}
$$

&emsp;&emsp;注意在使用KKT条件时，需要检验是否满足约束规范。