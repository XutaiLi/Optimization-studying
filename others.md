# 目标函数、损失函数、代价函数

**损失函数（Loss function）：**衡量预测值与真实值之间的距离

​	常见的损失函数

1. 均方误差MSE：$\frac{1}{n}\sum(y_i-\hat{y_i})^2$,对**较大的预测误差**有较高的惩罚,且容易求导
2. 交叉熵误差CrossEntropy：$-\sum y\cdot  log \hat{y}$,常用在分类算法中，y表示真实概率序列，$\hat{y}$表示预测概率序列

​	在多分类任务中，经常采用 softmax 激活函数+交叉熵损失函数，因为交叉熵描述了两个概率分布的差异，然而神经网络输出的是向量，并不是概率分布的形式。所以需要 softmax激活函数将一个向量进行“归一化”成概率分布的形式，再采用交叉熵损失函数计算 loss。

[^关于softmax函数]: Softmax函数可以将上一层的原始数据进行归一化，转化为一个(0,1)之间的数值，这些数值可以被当做概率分布，用来作为多分类的目标预测值。Softmax函数一般作为神经网络的最后一层，接受来自上一层网络的输入值，然后将其转化为概率。

​	3.

# conjugate function共轭函数

![sigmoid](./image/sigmoid.jpg)

# Legendre Transformation 勒让德变换

Legendre Transformation是为了实现一组独立变量到另一组独立变量的转化

$对于任何一个点，其切线为f(X)=\sum_i \frac{\partial f(X)}{\partial x_i}x_i+$

$记f(X)=f(x_1,x_2,...x_n),则df(X)=\sum_i \frac{\partial f(X)}{\partial x_i}dx_i①$

$由 d(ux)=udx+xdu \implies df(X)=\sum_i (d(u_ix_i)-x_idu_i) $

$f(X)=g(U)=g(u_1,u_2,...,u_n),dg(U)=\sum_i\frac{\partial g(U)}{\partial u_i}du_i =\sum_i \frac{\partial g(U)}{\partial u_i}\frac{du_i}{dx_i}dx_i②$

$①②对应相等，\frac{\partial g(U)}{\partial u_i}\frac{du_i}{dx_i}=\frac{\partial f(X)}{\partial x_i}$



# 传统的非线性规划方法为什么更依赖于初值？而神经网络似乎并不需要？

非线性规划指目标函数和约束条件中至少有一部分是非线性的，常常出现目标函数多峰，陷入局部最优解，搜索空间膨胀等。



传统优化算法主要包括梯度下降法、共轭梯度下降法、牛顿法等，（此处应当引用对各个方法的笔记）



如，下图为非凸目标函数$f(x)=x^4-4x^2+x$，使用不同初始值的梯度下降法寻优过程，在初值为-2.5时，能够很快找到最优值，而初值为0时虽然能找到最优值，但是速度相对较慢，而当初值为2.5时，则陷入局部最优解中

![Figure_1](E:\360MoveData\Users\李绪泰\OneDrive\桌面\opt\梯度下降失效.png)

神经网络为什么不依赖于初值？

神经网络有大量参数，随机初始化的权重提供了丰富的初始搜索点分布，尽管训练神经网络是非凸优化问题，但其目标函数通常表现出“平滑”的低维结构。



神经网络训练中常见的初始化策略：

$fan_{in}表示输入神经元的数量，fan_{out}表示输出神经元的数量$

**均匀随机初始化**：初始权重随机分布在某一范围内。通常为$[-\frac{1}{\sqrt { fan_{in}}},\frac{1}{\sqrt {fan_{out}}}]$

**Xavier初始化**:来自于**Xavier Glorot**在2010年发表的 *[[Understanding the difficulty of training deep feedforward neural networks* (mlr.press)](https://proceedings.mlr.press/v9/glorot10a.html)]，也称Glorot初始化

![sigmoid](.\image\sigmoid.jpg)

**如果初始化值很小**，那么随着层数的传递，方差就会趋于0，此时输入值也变得越来越小，在sigmoid上就是在0附近，接近于线性，失去了非线性。**如果初始值很大**，那么随着层数的传递，方差会迅速增加，此时输入值变得很大，而sigmoid在大输入值写倒数趋近于0，反向传播时会遇到梯度消失的问题。

在初始化权重时考虑了网络的大小（输入和输出单元的数量）。这种方法通过使权重与前一层中单元数的平方根成反比来确保权重保持在合理的值范围内。



​	其核心思想是，**对于每层网络来说保持输入和输出的方差一致，以避免所有输出值趋向于0，从而避免梯度消失现象。**



**He初始化**：通过控制权重的方差，确保神经网络层间的梯度分布均匀。



# 计算复杂性理论中的基本概念

## P类问题

​	P类问题（Polynomial time problems）是指那些能够在**多项式时间**内通过确定性算法求解的问题。换句话说，对于 P类问题，我们已经**有高效的算法**，能够在合理的时间范围内解决这些问题。

##  NP类问题

​	NP类问题（Nondeterministic Polynomial time problems）是那些**可以在多项式时间内验证解是否正确**的问题。注意，NP类问题不一定能在多项式时间内求解，但如果给出一个候选解，可以快速验证它是否为正确解。

## NP完全问题，NPC问题

​	NP完全问题（NP-Complete problems）是NP类问题中的一种特殊子集。这些问题不仅属于NP类问题（即解可以在多项式时间内验证），并且所有的NP类问题都可以通过某种方式归约为这些问题。换句话说，NP完全问题是最难的NP问题，它们之间存在某种等价性。如果你能够找到一种快速解决NP完全问题的方法，那么所有的NP类问题都可以在多项式时间内求解。

## NP-Hard Problem

​	NP 难问题（NP-Hard Problem），描述了那些非常难以求解的问题。这些问题的特性使得**求解它们的最佳算法目前尚未找到**，甚至可能不存在通用的快速求解方法。至少和NP完全问题一样难，但不一定是NP类问题



# 逆优化问题

​	传统优化问题是给定目标和约束，求解最优决策，而逆优化问题是给定一组决策，求解目标和约束，使该决策为最优决策。

## 经典逆优化问题算法

**还需进一步了解**

1. 线性模型变形利用对偶理论及KKT条件将问题线性化成单层线性规划
2. 整数规划模型采用切分算法逐步紧逼逆可行解空间
3. 序贯决策模型用动态规划方法求解
4. 网络流模型利用网络流对偶性条件解偶问题
5. 凸规划模型利用对偶理论和KKT条件的凸 relax技巧

# 无容量设施选址问题（Uncapacitated facility location，UFL）

该问题只有二元决策变量，常用于二元优化算法的性能比较和分析，属于NP-Hard problem

- [ ] 设施位置集M={1，2，3，...，m}，客户集N={1，2，3，...，n}
- [ ] $c_{ij}$表示客户i与设施j之间的单位运输费用，$f_j>0$表示设施j开放的费用，
- [ ] 要求每个客户必须选择一个且只能选择一个设施来满足其需求，并使得总费用最小



问题模型：

$\displaystyle min z= \sum_{i=1}^n \sum_{j=1}^m c_{ij}x_{ij}+\sum_{j=1}^n f_jy_j$

$\displaystyle s.t. \sum_{j=1}^n x_{ij}=1,\forall j \in N$

$\displaystyle x_{ij} \leq y_j,\forall i \in N,\forall j \in M$

$x_{ij}\in \{0,1\},\forall i \in N,\forall j \in M$

$y_j\in\{0，1\},\forall j \in M$

