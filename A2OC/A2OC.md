# A2OC

**When Waiting is not an Option: Learning Options with a Deliberation Cost**

## 1. Abstract

时间扩展的动作（options）可以完全端到端地学习，而不是预先指定。虽然如何学习 options 越来越清楚，但好的 options 应该是什么的问题仍然难以捉摸。本文通过 deliberation cost 的概念制定了有界理性框架中好的 options 应该是什么的答案。然后，本文推导出实用的基于梯度的学习算法来实现这一目标。

## 2. Introduction

时间抽象在人工智能中有着丰富的历史，并作为各种问题的有用机制，这些问题在可能的设置中影响人工智能系统，包括：生成更短的计划、加速规划、提高泛化、产生更好的探索、提高对模型错误规划或部分可观察性的鲁棒性。

在强化学习中， options 提供了一个框架来表示、学习和计划，并具有时间扩展的动作。近年来，由于从数据中自动构建此类抽象的成功案例越来越多，人们对强化学习中的时间抽象的兴趣大大增加。然而，定义构成良好选项集的内容仍然是一个悬而未决的问题。

在本文中，作者利用有界理性框架来解释是什么为 RL 系统提供良好的时间抽象。现有的许多强化学习工作都集中在马尔可夫决策过程上，其中可以在某些假设下获得最优策略。然而，最优性没有考虑代理可能的资源限制，这被认为是可以访问大量数据和计算时间。选项帮助代理克服这些限制，允许计算策略更快。然而，从绝对最优性的角度来看，时间抽象是不必要的：最优策略是通过原始动作实现的。因此，在精确的理论意义时间抽象动作有帮助方面，很难形式化。

有界合理性是理解自然系统和人工系统中合理性的一个非常重要的框架。在本文中，作者提出了有界理性作为一个镜头，我们可以描述构建时间抽象的需求，因为它们的目标是帮助在计算时间方面受到限制的智能体。这种观点帮助我们更精确地制定在 options 构建过程中应该满足哪些客观标准。作者建议好的 options 是那些允许智能体更快地学习和规划的 options ，并根据这个想法为学习 options 提供优化目标。

## 3. Preliminaries

一个有限折扣马尔科夫决策过程 $\mathcal{M}$ 是一个元组 $\mathcal{M} \doteq (\mathcal{S}, \mathcal{A}, \gamma, r, P)$ ，其中 $\mathcal{S}$ 和 $\mathcal{A}$ 分别状态和动作集， $\gamma \in [0,1)$ 是折扣因子。奖励函数 $r$ 通常被假定为状态和动作的确定性函数，但也可以映射到分布 $r:\mathcal{S} \times \mathcal{A} \to Dist(\mathbb{R})$ 。转移矩阵 $P: \mathcal{S} \times \mathcal{A} \to Dist(\mathcal{S})$ 是下一个状态的条件分布，因为动作 $a \in \mathcal{A}$ 是在某个状态 $s \in \mathcal{S}$ 下执行的。随机平稳策略 $\pi:\mathcal{S} \to Dist(\mathcal{A})$ 或确定性策略 $\pi: \mathcal{S} \to \mathcal{A}$ 与 MDP $\mathcal{M}$ 的相互作用在状态、动作和奖励上诱导马尔科夫过程。该马尔科夫过程定义了期望折扣回报 $V_{\pi}(s) \doteq \mathbb{E}_{\pi} [\sum_{t=0} \gamma^t r(S_t, A_t) | S_0 = s]$ 。策略 $\pi$ 的值函数 $V_{\pi}$ 满足 Bellman 方程：
$$
V_{\pi}(s) = \sum_a \pi(a|s)(r(s,a) + \gamma \sum_{s'}P(s'|s,a)V_{\pi}(s'))
$$
在控制问题中，我们有兴趣为给定的 MDP 找到最佳策略。如果对所有 $s$ 满足 $V_{\pi^*}(s) \doteq \max_{\pi} V_{\pi}(s)$ ，则策略 $\pi^*$ 是最优的。

强化学习中一类重要的控制方法基于 Actor-Critic 架构。同样，函数逼近可用于价值函数，策略也可以在搜索的参数化族中近似。在策略梯度定理中，期望折扣回报相对于策略参数的梯度形式为 $\mathbb{E}_{\alpha, \pi_{\theta}} [\sum_a \frac{\part \pi_{\theta}(a|s)}{\part \theta}Q_{\pi_\theta}(s,a)]$ ，其中 $\alpha$ 是初始状态分布。然后可以通过在策略参数上随机梯度上升来找到局部最优策略，同时学习动作值函数 $Q_{\pi_\theta}(s,a)$ 。

### 3.1 Options

Options 提供了一个框架，用于使用时间抽象动作表示、规划和学习。 Options 框架假设存在一个基础 MDP ，在该基础 MDP 上覆盖时间抽象的动作，称为 options 。一个 option 被定义为一个三元组 $(\mathcal{I}_0,\pi_0,\beta_0)$ ，其中 $\mathcal{I}_0 \subseteq \mathcal{S}$ 是一个初始集， $\pi_0:\mathcal{S} \to Dist(\mathcal{A})$ 是 option 的策略（也可以是确定性的）， $\beta_0:\mathcal{S} \to [0,1]$ 是终止条件。在 call-and-return 执行模型中， options 的策略 $\mu:\mathcal{S} \to Dist(\mathcal{O})$ 选择一个可以在给定状态下启动并执行该 option 的策略直到终止的 option 。一旦选择的 option 终止， options 的策略就会选择一个新的 option 并重复该过程，直到该 episode 结束。

一组 options 和基本的 MDP 的组合形成了半马尔可夫决策过程（SMDP），其中两个决策点之间的过渡时间是一个随机变量。当仅在 state-option 对级别考虑诱导过程时，通常在转换为等效 MDP 后重用通常的动态规划结果。

要理解这一点，我们需要为每个 option 定义定义两种模型：奖励模型 $b_0:\mathcal{S} \to \mathbb{R}$ 和转移模型 $F_0:\mathcal{S} \times \mathcal{S} \to \mathbb{R}$ 。如果一个 option 不依赖于自开始以来的历史，我们可以以封闭形式编写其模型或类似 Bellman 方程的解。与一组 options $\mathcal{O}$ 和它们上的策略相关的期望折扣回报是一组 Bellman 方程的解 $Q_{\theta}$ ：
$$
Q_{\theta}(s,o) = b_\theta(s,o) + \sum_{s'}F_{\theta}(s',s,o)V_{\theta}(s') \\
\doteq \mathbb{E}_\theta[\sum_{t=0}\gamma^t r(S_t,A_t)|S_0=s,O_0=0]
$$
其中 $\theta$ 是 options 的策略 $\mu$ ，options 策略和终止条件上的连接。

### 3.2 Intra-Option Bellman Equations

在马尔科夫 options 的情况下， Bellman 方程存在另一种形式，称为 option 内 Bellman 方程，这是推导基于梯度的学习算法学习 options 的关键。

令 $Z_t \doteq (S_t,O_t)$ 是 state-option 元组上的随机变量。我们将 state-option 对的空间称为增强状态空间。增广状态上马尔科夫过程的转移矩阵由下式给出：
$$
\tilde{P}_{\theta}(z'|z,a) = P(s'|s,a)((1-\beta_\theta(s',o)) 1_{o'=o} + \beta_{\theta}(s',o)\mu_{\theta}(o'|s'))
$$
使用这种链结构，我们可以定义  MDP $\tilde{\mathcal{M}} \doteq (\tilde{P}_\theta,\tilde{r},\gamma) $ ，其相关的值函数为 $\tilde{V}_\theta :(\mathcal{S} \times \mathcal{O}) \to \mathbb{R}$ ：
$$
\tilde{V}_{\theta}(z) = \mathbb{E}_{\theta} [\sum^{\infty}_{t=0} \gamma^t \tilde{r}(Z_t,A_t,Z_{t+1})|Z_0=z] \\
= \sum_{a,z'}\pi_{\theta}(a|z)\tilde{P}_{\theta}(z'|z,a)(\tilde{r}(z,a,z') + \gamma \tilde{V}_{\theta}(z'))
$$
由于奖励来自基本 MDP ，我们可以简单地写为 $\tilde{r}(z,a,z') = r(s,a)$ ，并且由于 $\sum_{z'}\tilde{P}_{\theta}(z'|z,a) = 1$ ，因此
$$
\sum_{z'}\tilde{P}_{\theta}(z'|z,a)\tilde{r}(z,a,z') = r(s,a)
$$
因此（4）式可以写为
$$
\tilde{V}_{\theta}(z) \doteq Q_\theta(s,o) = \sum_a \pi_{\theta}(a|s,o) ( r(s,a) + \gamma\sum_{s'}P(s'|s,a) [Q_{\theta}(s',o) - \beta_{\theta}(s',o)A_{\theta}(s',o)])
$$
其中 $A_{\theta}(s,o) = Q_{\theta}(s,o) - V_{\theta}(s)$ 是优势函数。（6）式的方程完全对应于 option 内的 Bellman 方程。然而，我们选择以另一种方式呈现它们：
$$
U(s',o) \doteq (1-\beta_{\theta}(s',o))Q_{\theta}(s',o) + \beta_{\theta}(s')V_{\theta}(s') \\
= Q_{\theta}(s',o) - \beta_{\theta}(s',o)A_{\theta}(s',o)
$$
其中 $U(s',o)$ 表示继续相同的 option 或切换到更好的 option 的效用。

### 3.3 Optimization

Option-critic 架构是一种基于梯度的 actor-critic 架构，用来端对端学习 options 。与 actor-critic 方法一样， option-critic 的想法是参数化 options 策略和终止条件，并通过对期望折扣回报的随机梯度上升联合学习它们的参数。

在假设 options 处处可以使用的情况下，为 options 策略与终止函数提供了梯度的形式。假设参数向量 $\theta = [\theta_\mu;\theta_\pi;\theta_\beta]$ 被划分为 options 上的策略， options 的策略和终止函数上的不相交参数集。

在 options 策略的梯度定理中，结果保持了与原始 MDP 策略梯度定理相同的形式，但在增强状态空间上。如果 $J_{\alpha}(\theta)$ 是 options 集及其策略的期望折扣回报，则 options 策略的梯度为：
$$
\frac{\part J_{\alpha}(\theta)}{\part \theta_{\pi}} = \gamma \mathbb{E}_{\alpha,\theta} [\sum_a \frac{\part \pi_{\theta}(a|z)}{\part \theta_{\pi}} \tilde{Q}_{\theta}(z,a)]
$$
其中， $\alpha$ 是状态和 options 的初始状态分布。

为了获得终止函数的梯度，先取 option 内 Bellman 方程的导数：
$$
\frac{\part Q_{\theta}(s,o)}{\part {\theta_{\beta}}} = \gamma \sum_a \pi_\theta(a|s,o) \sum_{s'} P(s'|s,a)[ \\ -\frac{\part \beta_{\theta}(s',o)}{\part \theta_{\beta}} A_{\theta}(s',o) + \frac{\part Q_{\theta}(s,o)}{\part \theta_{\beta}} - \beta_{\theta}(s',o)\frac{\part A_{\theta}(s',o)}{\part \theta_{\beta}}]
$$
通过观察（4）式和（9）式，我们可以很容易地求解导数的递归形式。事实上，很容易看出 $-\frac{\part \beta_{\theta}(s',o)}{\part \theta_{\beta}} A_{\theta}(s',o)$ 在通常的 Bellman 方程中起着”奖励“项的作用，并得出结论：
$$
\frac{\part J_{\alpha}(\theta)}{\part \theta_{\beta}} = \gamma \mathbb{E}_{\alpha,\theta} [ -\frac{\part \beta_{\theta}(s',o)}{\part \theta_{\beta}} A_{\theta}(s',o) ]
$$
因此终止梯度表明，如果一个 option 是有利的，终止的概率应该降低，使该 option 更长。相反，如果 option 的值小于在给定状态下通过不同选项所能达到的值， 则终止梯度将使它有可能在此状态下终止。

## 4. Deliberation Cost Model

从表示学习的角度来看，好的 options 应该允许智能体更快地学习和规划。由于其时间结构， options 提供了一种机制，智能体可以更好地利用其有限的计算资源并更快地行动。一旦选择了一个 option ，我们假设执行该 option 的计算成本可以忽略不计或恒定，直到终止。

将 options 上的策略呈现为较慢的基于模型的规划过程，而不是 option 中发生的快速和习惯性的学习。

拥有更长的 options 是提供更好的可解释性并通过压缩信息来简化通信和理解的一种方式。

考虑如下成本模型，在 option 中执行该 option 是无成本的，但在到达新状态时切换到另一个 option 会产生成本 $\eta$ 。进一步假设 option 的终止函数在所有状态是恒定的。如果 $\kappa$ 是该 option 的延续概率，则其期望折扣持续时间为$d = \frac{1}{1-\gamma\kappa}$ 。当终止产生固定成本 $\eta$ 时，该 option 每一步的平均成本为 $\eta / d = (1-\gamma\kappa)\eta$ 。因此，随着延续概率的增加和 option 变长，成本率降低。与仅使用原始动作相比，较长的 options 会导致更好的审议成本摊销。

### 4.1 Formulation

除了基本 MDP 的值函数 $\tilde{V}_{\theta}(z) \doteq Q_{\theta}(s,o)$ 和它们上的 options 外，我们还定义了一个直接代价函数 $\tilde{c}(z,a,z') \doteq c(s,o,a,s',o')$ 和相应的审议代价函数 $\tilde{D}_{\theta}(z) \doteq D_{\theta}(s,o)$ 。

![](.\1.png)

与一组 options 集和它们上的策略相关的折扣成本的期望总和为 $\tilde{D}_{\theta}:(\mathcal{S}\times\mathcal{O}) \to \mathbb{R}$ ：
$$
\tilde{D}_{\theta}(z) = \mathbb{E}_{\theta}[\sum^{\infty}_{t=0}\gamma^t\tilde{c}(Z_t,A_t,Z_{t+1})|Z_0=z]
$$
我们首先制定我们的目标，即最大化期望回报，同时保持审议成本低，将其作为约束优化问题：
$$
\underset{\theta}{\max} \sum_{s,o}\alpha(s,o)Q_{\theta}(s,o) \\
\text{subject to: } \sum_{s,o}\alpha(s,o)D_{\theta}(s,o) \le k
$$
其中 $\alpha$ 是 state-option 对的初始状态分布。我们考虑拉格朗日公式的无约束优化问题：
$$
\underset{\theta}{\max} J_{\alpha}(\theta), \\
\text{where } J_{\alpha}(\theta) \doteq \sum_{s,o}\alpha(s,o)(Q_{\theta}(s,o) - \eta D_{\theta}(s,o))
$$
$\eta \in \mathbb{R}$ 是正则化系数。虽然（13）式显示了 option 值函数和审议成本函数作为单独的实体，但它们实际上可以被视为单个 MDP ，其奖励函数是基础 MDP 奖励和成本函数的差异：
$$
J_{\alpha}(\theta) = \mathbb{E}_{\alpha,\theta}[\sum^{\infty}_{t=0} \gamma^t \tilde{r}(Z_t,A_t,Z_{t+1}) - \gamma^t \eta \tilde{c}(Z_t,A_t,Z_{t+1})]
$$
因此，有一组 Bellman 方程，变化后的奖励函数上的值函数为：
$$
\tilde{V}_{\theta}^{c}(z) = \sum_{a,z'}\pi_\theta(a|s,z)\tilde{P}_\theta(z'|z,a)( \tilde{r}(z,a,z') - \eta \tilde{c}(z,a,z') + \gamma \tilde{V}_\theta^c(z'))
$$
类似地，对于 options 上的策略的参数 $\theta_\mu$ 的意义上存在 Bellman 最优性方程：
$$
Q_{\mathcal{O}}^*(s,o) \doteq \underset{\theta_\mu \in \Pi(\mathcal{O})}{\max} (Q_{\theta_\mu}(s,o) - \eta D_{\theta_\mu}(s,o))
$$
这里的符号 $Q_{\theta_\mu}$ 表明 options 的参数保持固定，只允许 $\theta_\mu$ 变化。如果对于给定的 $\eta$ 在（16）式中达到最大值，则对于一组 options 的 option 上的策略 $\mu*$ 是 $\eta - \text{optimal}$ 。

### 4.2 Switching Cost and its Interpretation as a Margin

支持长 options 的一种方法是成本函数，它惩罚频繁的 options 的切换。以与 MDP 公式允许随机奖励函数相同的方式，我们还可以通过直接成本函数 $c$ 捕获切换的随机事件。由于 $\beta_{\theta}(s',o)$ 是伯努利随机变量对两种可能结果的平均值，切换或继续，因此切换事件对应的成本函数为 $c_{\theta}(s',o) = \gamma \beta_{\theta}(s',o)$ 。

在为 $c_{\theta}(s',o)$ 的这种选择扩展转换后的奖励上的价值函数时，我们得到：
$$
Q^c_{\theta}(s,o) = \sum_a\pi_{\theta}(a|s,o)(r(s,a) + \gamma \sum_{s'}P(s'|s,a)[
Q^c_{\theta}(s',o) - \beta_{\theta}(s',o)(A^c_{\theta}(s,o) + \eta)])
$$
因此，将切换成本函数添加到基本 MDP 奖励中，在转换后的奖励上对优势函数 $A_{\theta}^c$ 贡献了标量边距 $\eta$ 。在 option-critic 框架中学习终止函数时，无约束问题（（13）式）的终止梯度具有如下形式：
$$
\frac{\part J_{\alpha}(\theta)}{\part \theta_{\beta}} = \gamma \mathbb{E}_{\alpha,\theta} [-\frac{\part \beta_{\theta}(S_{t+1},O_t)}{\part \theta_\beta} (A^c_{\theta}(S_{t+1},O_t) + \eta)]
$$
因此， $\eta$ 为 option 应该有多好设置了边距或基线：这可能是由于近似误差或反映值估计中某种形式的不确定性的校正。通过增加其值，可以减少优势函数的差距，倾斜平衡以支持维护 option 而不是终止它。

### 4.3 Computational Horizon

由于公式的普遍性，审议成本函数的折扣因子可能与基础 MDP 奖励上的值函数的折扣因子不同。（13）式的无约束公式就变成了两个折扣因子的函数：基础 MDP 的 $\gamma$ 和审议成本函数的 $\lambda$ ：
$$
J^{\gamma,\lambda}_{\alpha}(\theta) = \sum_{s,o}\alpha(s,o)(Q^{\gamma}_{\theta}(s,o) - \eta D^{\lambda}_{\theta}(s,o))
$$
由于审议成本函数对终止参数的导数为：
$$
\frac{\part D^{\lambda}_{\theta}(s,o)}{\part \theta_{\beta}} = \frac{\part}{\part \theta_{\beta}} \sum_a \pi_{\theta}(a|s,o) \sum_{s'}P(s'|s,a)(c_{\theta}(s',o) + \\
\lambda[(1-\beta_{\theta}(s',o))D^{\lambda}_{\theta}(s',o) + \beta_{\theta}(s',o)\sum_{o'}\mu_{\theta}(o'|s')D^{\lambda}_{\theta}(s',o')])
$$
当代价函数为 $c_{\theta}(s',o) \doteq \gamma\beta_{\theta}(s',o)$ 时令 $\lambda = 0$ ，此时只留下一项： $\frac{\part \beta_{\theta}(s',o)}{\part \theta_{\beta}} \eta$ 。因此，混合目标的导数为
$$
\frac{\part J^{\gamma,\lambda=0}_{\alpha}(\theta)}{\part \theta_{\beta}} = \gamma \mathbb{E}_{\alpha,\theta} [- \frac{\part \beta_{\theta}(s',o)}{\part \theta_{\beta}} (A_{\theta}(s',o)+\eta)]
$$
虽然与（18）式类似，但（21）式依赖于 $A_{\theta}$ 而不是 $A^c_{\theta}$ 。我们还可以看到，当 $\gamma = \lambda$ 时，我们从（18）式中恢复转换后的 MDP 中期望回报的导数的相同形式：
$$
\frac{\part J^{\gamma=\lambda}_{\alpha}(\theta)}{\part \theta_{\beta}} = \gamma \mathbb{E}_{\alpha,\theta}[-\frac{\part \beta_{\theta}(s',o)}{\part \theta_{\beta}}(A^c_{\theta}(s',o) + \eta)] = \frac{\part J_{\alpha}(\theta)}{\part \theta_{\beta}}
$$
审议成本函数的折扣因子 $\lambda$ 提供了一种截断成本总和的机制。因此，它与仅缩放审议成本函数但不影响计算范围的正则化系数 $\eta$ 起着不同的作用。

## 5. Experiments

虽然端到端学习 options 是可能的，但除非使用正则化，否则频繁的终止可能会成为一个问题。本文基于 Advantage Asynchronous Actor-Critic (A3C) 架构，结合审议成本的想法，提出新的 option-critic 实现。本文的试验旨在评估：生成 options 的可解释性，是否可以控制频繁终止到单步 options ，审议成本是否可以为更快地学习提供归纳偏差。

### 5.1 Asynchronous Advantage Option-Critic (A2OC)

Option-Critic 架构是建立在 DQN 算法之上，其是一种离线算法，使用来自经验回放缓冲区的样本。另一方面， Option-critic 是一种在线算法，它使用每个新的采样转换进行更新。然而，在训练神经网络时，使用在线样本会出现问题。

A3C 方法解决了这个问题，并通过运行多个并行智能体获得稳定的在线学习方法。并行智能体允许深度网络从非常不同的状态中看到样本，这极大地稳定了学习过程。该算法与 option-critic 的思想一致，因为它们都使用在线策略梯度进行训练。我们引入 A2OC 方法，其以与 A3C 类似的方式学习 options ，但在 option-critic 架构中。

![](.\2.png)

我们使用相同大小的卷积神经网络，它输出一个在三个头之间共享的特征向量： option 策略，终止函数和 Q 值网络。 option 策略是线性 softmax 函数，终止函数使用 sigmoid 激活函数来表示终止的概率， Q 值只是线性层。在训练期间，所有梯度相加，并在单个线程实例中执行更新。 A3C 只需要为其策略学习一个值函数，而不是每个动作的 Q 值。相似地， A2OC 通过采样远离动作维度，但由于潜在的增强状态空间，需要保持 state-option 值。

对于超参数，在 options 上采用 $\epsilon-greedy$ 策略，其中 $\epsilon = 0.1$ 。预处理缩放与 A3C 相同， RGB 像素缩放到 $84 \times 84$ 灰度图像。智能体重复四个连续动作的动作，并接收 4 帧的堆栈作为输入。使用 0.01 的熵正则化，它推动 option 策略不会崩溃到确定性策略。所有试验采用 0.0007 的学习率，通常用 16 个并行线程训练智能体。

### 5.2 Empirical Effects of Deliberation Cost

## 6. Conclusion and Future Work

本文展示了使用审议成本作为激励创建持续较长时间 options 的一种方式。

审议成本不仅仅是惩罚冗长计算的想法。它还可用于在其环境中合并智能体固有的其他形式的边界。

未来工作的一个有趣的方向是在错过的机会（missed opportunity）方面考虑审议成本，并在与环境异步交互时为隐式形式的正则化开辟道路。

