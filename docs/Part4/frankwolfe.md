# Frank-Wolfe算法

## 基本理论

&emsp;&emsp;Frank和Wolfe于1956年提出了一种用于求解**线性约束的非线性规划**问题的线性化算法，通常称为Frank-Wolfe算法，也称为条件梯度算法。基本思想是从某个可行点开始，沿着令目标函数值下降的方向进行探索，求出新的可行迭代点，反复迭代直到找到最优解。核心在于Frank-Wolfe算法每次迭代会将目标函数**线性化**(linearization)，通过求解相应的线性规划获得可行的下降方向。

&emsp;&emsp;考虑如下的线性约束的非线性规划问题：

$$
\begin{equation}\label{f:fw}\tag{1}
	\begin{split}
		\min\quad&Z(\boldsymbol{x}) \\
		\text{s.t.}\quad&\boldsymbol{A}\boldsymbol{x}=\boldsymbol{b} \\
		&\boldsymbol{x}\geq0
	\end{split}
\end{equation}
$$

其中，$\boldsymbol{A}$是$m\times n$的系数矩阵，$\boldsymbol{b}$是$m$维常数列向量，$\boldsymbol{x}$是$n$维的列向量，$Z(\boldsymbol{x})$是连续可微函数，记可行域$D=\{\boldsymbol{x}|\boldsymbol{A}\boldsymbol{x}=\boldsymbol{b},\boldsymbol{x}\geq0\}$。

&emsp;&emsp;关键的线性化$Z(\boldsymbol{x})$的方法：记第$t$次迭代点为$\boldsymbol{x}_t$，对$Z(\boldsymbol{x})$在$\boldsymbol{x}_t$上进行一阶泰勒展开，得到$Z(\boldsymbol{x})$的近似线性函数$Z_L(\boldsymbol{x})$，即：

$$
\begin{equation}\notag
	\begin{split}
		Z_L^t(\boldsymbol{x})=&Z(\boldsymbol{x}_t)+\nabla Z(\boldsymbol{x}_t)^\top\cdot(\boldsymbol{x}-\boldsymbol{x}_t) \\
		=&[Z(\boldsymbol{x}_t)-\nabla Z(\boldsymbol{x}_t)^\top\boldsymbol{x}_t]+\nabla Z(\boldsymbol{x}_t)^\top\boldsymbol{x}
	\end{split}
\end{equation}
$$

&emsp;&emsp;可以注意到$Z(\boldsymbol{x}_t)-\nabla Z(\boldsymbol{x}_t)^\top\boldsymbol{x}_t$是一个常数，于是原线性约束的非线性规划问题(\ref{f:fw})可以转换为以下线性规划问题：

$$
\begin{equation}\label{f:fw-linear}\tag{2}
	\begin{split}
		\min\quad&\nabla Z(\boldsymbol{x}_t)^\top\cdot\boldsymbol{x} \\
		\text{s.t.}\quad&\boldsymbol{x}\in D
	\end{split}
\end{equation}
$$

&emsp;&emsp;上式($\ref{f:fw-linear}$)是一个线性规划问题，有很多效率很高的方法可以求解。设上式求解的结果为$\boldsymbol{y}_t$。因为$\boldsymbol{y}_t$是问题($\ref{f:fw-linear}$)的最优解，那么$\nabla Z(\boldsymbol{x}_t)^\top\cdot\boldsymbol{y}_t-\nabla Z(\boldsymbol{x}_t)^\top\cdot\boldsymbol{x}_t\leq0$。那么，其结果必有两种情况：

* 若$\nabla Z(\boldsymbol{x}_t)^\top\cdot(\boldsymbol{y}_t-\boldsymbol{x}_t)=0$，则停止迭代，此时的$\boldsymbol{x}_t$就是原问题($\ref{f:fw}$)的KKT点(满足KKT条件的点)。
* 若$\nabla Z(\boldsymbol{x}_t)^\top\cdot(\boldsymbol{y}_t-\boldsymbol{x}_t)<0$，因此，$(\boldsymbol{y}_t-\boldsymbol{x}_t)$为$Z(\boldsymbol{x})$在$\boldsymbol{x}_t$处的下降方向。

&emsp;&emsp;我们从迭代点$\boldsymbol{x}_t$开始，现在已知可行下降方向为$(\boldsymbol{y}_t-\boldsymbol{x}_t)$，接下来可以求得最优步长：

$$
\begin{equation}\label{f:fw-eta}\tag{3}
	\min\limits_{0\leq\lambda\leq1}\quad Z[\boldsymbol{x}_t+\lambda(\boldsymbol{y}_t-\boldsymbol{x}_t)]
\end{equation}
$$

&emsp;&emsp;得到最优步长$\lambda_t$之后，我们就可以更新迭代点：

$$
\begin{equation}\label{f:fw-update}\tag{4}
	\boldsymbol{x}_{t+1}=\boldsymbol{x}_t+\lambda_t(\boldsymbol{y}_t-\boldsymbol{x}_t)
\end{equation}
$$

&emsp;&emsp;由于$\boldsymbol{y}_t-\boldsymbol{x}_t\neq0$，且为下降方向，所以$Z(\boldsymbol{x}_{t+1})<Z(\boldsymbol{x}_t)$。得到$\boldsymbol{x}_{t+1}$之后，作为新的迭代点，反复迭代直至满足某个收敛准则(convergence criterion)。上式($\ref{f:fw-update}$)更新迭代点方法可以改写为$\boldsymbol{x}_{t+1}=(1-\lambda_t)\boldsymbol{x}_t+\lambda_t\boldsymbol{y}_t$，因此次新迭代点$\boldsymbol{x}_{t+1}$实际上是$\boldsymbol{x}_t$与$\boldsymbol{y}_t$的凸组合，故Frank-Wolfe算法又被称为**凸组合法**(convex combination method)。

!!! tip "Frank-Wolfe算法描述"
    1. 在可行域中，选择初始值$\boldsymbol{x}_0$，给定允许的误差$\epsilon$，迭代计数$t=0$。
    2. 求得$Z(\boldsymbol{x})$在$\boldsymbol{x}_t$处的梯度后，构造近似线性规划问题($\ref{f:fw-linear}$)。
    3. 求解近似线性规划问题($\ref{f:fw-linear}$)，得到$\boldsymbol{y}_t$。
    4. 若$\nabla Z(\boldsymbol{x}_t)^\top\cdot(\boldsymbol{y}_t-\boldsymbol{x}_t)=0$：停止迭代。否则按(\ref{f:fw-eta})求解最优步长$\lambda_t$。
    5. 按($\ref{f:fw-update}$)更新迭代点，得到$\boldsymbol{x}_{t+1}$。
    6. 判断迭代条件：若$|Z(\boldsymbol{x}_t)-Z(\boldsymbol{x}_{t+1})|<\epsilon$(也可以是其他方法)：停止迭代。否则，令$t=t+1$，返回第二步。

&emsp;&emsp;最后，从Frank-Wolfe算法的步骤也可以看出，该算法无法避免陷入局部最优，并且受到初始值的影响。

## 代码实现

&emsp;&emsp;下面是Frank-Wolfe算法的简单实现：
```python
import numpy as np
import scipy
from autograd import grad

class FrankWolfe:
	def __init__(self, func, A, b, grad_func=None):
		self.func = func
		self.A = A
		self.b = b
		self.grad_func = grad_func
		if grad_func is None:
			self.grad_func = grad(func)
		self.x = []
	
	def search_step(self, l, x, y):
		return self.func(x + l * (y - x))
	
	def iter(self, x):
		g = self.grad_func(x)
		lin = scipy.optimize.linprog(g.transpose(), A_eq=self.A, b_eq=self.b, method='highs')
		y = lin.x
		if np.inner(g, (y - x)) == 0:
			return x
		sca = scipy.optimize.minimize_scalar(self.search_step, bounds=(0.0, 1.0), args=(x, y))
		step = sca.x
		return x + step * (y - x)
	
	def optimize(self, x0, eps=1e-8, max_iter=400):
		self.x = [x0]
		for i in range(max_iter):
			self.x.append(self.iter(self.x[i]))
			if np.abs(self.func(self.x[i]) - self.func(self.x[i + 1])) < eps:
				return self.x[i + 1]
		return self.x[-1]
```