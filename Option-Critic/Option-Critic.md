# Option-Critic

## 1. Abstract

时间抽象是强化学习中扩大学习和规划的关键。

我们推导了 options 的策略梯度定理，并提出了一种新的 option-critic 架构，能够与基于 options 的策略一起学习 options 的内部策略和终止条件，，而不需要提供任何额外的奖励或子目标。 

## 2. Introduction

时间抽象允许表示有关在不同时间尺度发生的动作过程的知识。在强化学习中， options 提供了一个框架，用于定义此类动作过程以及无缝地学习和规划它们。

本文提出一种观点，它模糊了发现 options 与学习 options 之间的界限。基于策略梯度定理，本文得出了新的结果，使得 options 内部的策略和终止函数的学习过程与对 options 的策略学习过程同时进行。在离散或连续的状态和动作空间下，这种方法自然适用于线性和非线性函数逼近器。

从单个任务中学习时，现有的学习 options 的方法要慢得多：大部分好处来自在类似任务中重复使用学到的 options 。相比之下，此方法能够在单个任务中成功学习 options ，而不会导致任何速度减慢，同时仍然为迁移学习提供好处。

与其他方法不同，我们只需要指定所需 options 的数量，不需要有子目标、额外奖励、演示、多个问题或任何其他特殊调整（但如果需要，该方法可以利用伪奖励函数）。

## 3. Preliminaries and Notion

马尔科夫决策过程由一组状态 $\mathcal{S}$ 、一组动作 $\mathcal{A}$ 、一个转移函数 $P:\mathcal{S} \times \mathcal{A} \to (\mathcal{S} \to [0,1])$ 和一个奖励函数 $r:\mathcal{S} \times \mathcal{A} \to \mathbb{R}$ 组成。马尔科夫平稳策略 $\pi : \mathcal{S} \times \mathcal{A} \to [0,1]$ 是基于状态的动作的概率分布。在 discounted 问题中，策略 $\pi$ 的值函数定义为期望收益： $V_{\pi}(s) = \mathbb{E}_{\pi} [\sum^{\infty}_{t=0} \gamma^t r_{t+1} | s_0=s]$ ，其动作值函数为 $Q_{\pi}(s,a) = \mathbb{E}_{\pi}[\sum^{\infty}_{t=0} \gamma^t r_{t+1} | s_0=s, a_0=a]$ ，其中 $\gamma \in [0,1)$ 为折扣因子 。如果 $\pi(s,a)>0 \ \text{iff} \ a = \arg\max_{a'}Q(s,a')$ ，则策略 $\pi$ 对于给定的动作值函数 $Q$ 是贪婪的。在离散 MDP 中，至少有一个对其自身的动作值函数是贪婪的最优策略。

### 3.1 Policy Gradient Methods

策略梯度方法通过执行随机梯度下降来优化给定参数化随机策略族 $\pi_{\theta}$ 上的性能指标。策略梯度定理提供了平均奖励和折扣奖励性能指标相对于 $\theta$ 的梯度表达式。

在折扣奖励设置中，性能指标是相对指定的起始状态（或分布） $s_0$ 来定义的：$\rho(\theta,s_0) = \mathbb{E}_{\pi_{\theta}}[\sum^{\infty}_{t=0}\gamma^t r_{t+1}|s_0]$ 。策略梯度定理表明： $\frac{\partial\rho(\theta,s_0)}{\partial\theta} = \sum_s \mu_{\pi_{\theta}}(s|s_0) \sum_a \frac{\partial\pi_{\theta}(a|s)}{\partial\theta}Q_{\pi_{\theta}}(s,a)$ ，其中 $\mu_{\pi_{\theta}}(s|s_0) = \sum^{\infty}_{t=0} \gamma^t P(s_t=s|s_0)$ 是沿着从 $s_0$ 开始的轨迹的状态的折扣权重。 在实践中，策略梯度是根据策略平稳分布的样本来估计的。忽略这种平稳分布中的折扣因子会使通常的策略梯度估计产生偏差，然而，纠正这种差异也会降低数据效率。

### 3.2 The Options Framework

马尔科夫 option $\omega \in \Omega$ 是一个三元组 $(\mathcal{I}_{\omega},\pi_{\omega},\beta_{\omega})$ ，其中 $\mathcal{I}_{\omega} \in \mathcal{S}$ 是初始集， $\pi_{\omega}$ 是 option 内策略， $\beta_{\omega}: \mathcal{S} \to [0,1]$ 是 终止函数。我们还假设 $\forall s \in \mathcal{S},\forall \omega \in \Omega:s \in \mathcal{I}_{\omega}$ ，即所有 option 在任何地方都可用。

被赋予一组 options 的 MDP 称为半马尔可夫决策过程（Semi-Markov Decision Process），它有 option 上的值函数 $V_{\Omega}(s)$ 和 option 值函数 $Q_{\Omega}(s,\omega)$ 。底层 MDP 的存在提供了并行学习许多不同 options 的可能性。

## 4. Learning Options

在任何时候，我们都希望将所有可用的经验提炼到我们系统的每个组件中：值函数， options 上的策略， option 内的策略和终止函数。为了实现这一目标，我们专注于学习 options 上的策略和终止函数。假设它们使用可微参数化函数逼近器表示。

考虑 call-and-return option 执行模型，其中智能体根据其对 options 的策略 $\pi_{\Omega}$ 选择 option $\omega$ ，然后遵循 option 内的策略 $\pi_{\omega}$ 直到终止（由 $\beta_{\omega}$ 指示）。令 $\pi_{\omega,\theta}$ 表示由 $\theta$ 参数化的 option $\omega$ 的 option 内的策略，$\beta_{\omega,\vartheta}$ 表示由 $\vartheta$ 参数化的终止函数。基于策略梯度定理，我们提出了学习 options 的两个新结果。这两个结果是在以下假设下得出的： the goal is to learn options that maximize the expected return in the current task 。

假设指标是直接优化从指定状态 $s_0$ 和 option $\omega_0$ 开始的所有轨迹的期望折扣回报，则：$\rho(\Omega,\theta,\vartheta,s_0,\omega_0) = \mathbb{E}_{\Omega,\theta,\omega}[\sum^{\infty}_{t=0}\gamma^t r_{t+1}|s_0,\omega_0]$ 。我们将获取该指标相对于 $\theta$ 和 $\vartheta$ 的梯度。

option-value 函数：
$$
Q_{\Omega}(s,\omega) = \sum_a \pi_{\omega,\theta}(a|s)Q_U(s,\omega,a)
$$
其中 $Q_U:\mathcal{S} \times \Omega \times \mathcal{A} \to \mathbb{R}$ 是在 state-option 对的上下文中执行 action 的值函数。
$$
Q_U(s,\omega,a) = r(s,a) + \gamma \sum_{s'}P(s'|s,a)U(\omega,s')
$$
函数 $U:\Omega \times \mathcal{S} \to \mathbb{R}$ 称为到达状态 $s'$ 时的 option-value 函数。
$$
U(\omega,s') = (1 - \beta_{\omega,\vartheta}(s'))Q_{\Omega}(s',\omega) + \beta_{\omega,\vartheta}(s')V_{\Omega}(s')
$$
$Q_U$ 和 $U$ 都取决于 $\theta$ 和 $\vartheta$ 。

导出策略梯度的最后一个要素是马尔科夫链，根据马尔科夫链来估计性能指标。自然的方法是考虑在增强状态空间中定义的链，因为 state-option 对现在在通常的马尔科夫链中扮演着常规状态的角色。如果 option $\omega_t$ 已在状态 $s_t$ 的时间 $t$ 启动或正在执行，则一步过渡到 $(s_{t+1},\omega_{t+1})$ 的概率为：
$$
P(s_{t+1},\omega_{t+1}|s_t,\omega_t) = \sum_a \pi_{\omega_t,\theta}(a|s_t)P(s_{t+1}|s_t,a)((1-\beta_{\omega_{t},\vartheta}(s_{t+1}))1_{\omega_t = \omega_{t+1}} + \beta_{\omega_t,\vartheta}(s_{t+1})\pi_{\Omega}(\omega_{t+1}|s_{t+1}))
$$
现在，我们将计算期望折扣收益相对于 option 内策略的参数 $\theta$ 的梯度，假设它们是随机且可微的。
$$
\frac{\partial Q_{\Omega}(s,\omega)}{\partial \theta} = (\sum_a \frac{\partial \pi_{\omega,\theta}(a|s)}{\partial\theta} Q_U(s,\omega,a)) + \sum_a \pi_{\omega,\theta}(a|s)\sum_{s'}\gamma P(s'|s,a) \frac{\partial U(\omega,s')}{\partial\theta}
$$
**Theorem 1** (Intra-Option Policy Gradient Theorem)：给定一组马尔科夫 Options ，其 option 内的策略在参数 $\theta$ 中可微，期望折扣回报相对于 $\theta$ 和初始条件 $(s_0,\omega_0)$ 的梯度为
$$
\sum_{s,\omega}\mu_{\Omega}(s,\omega|s_0,\omega_0) \sum_a \frac{\partial \pi_{\omega,\theta}(a|s)}{\partial\theta} Q_U(s,\omega,a)
$$
其中 $\mu_{\Omega}(s,\omega|s_0,\omega_0)$ 是沿着从 $(s_0,\omega_0)$ 开始的轨迹的 state-option 对的折扣权重。
$$
\mu_{\Omega}(s,\omega|s_0,\omega_0) = \sum^{\infty}_{t=0} \gamma^t P(s_t=s,\omega_t=\omega|s_0,\omega_0)
$$
该梯度描述了原始水平上的局部变化对全局期望折扣回报的影响。相反，子目标或伪奖励方法假设 options 的目标只是优化其自身的奖励函数，而忽略了其在总体性能指标中的传播。

接下来计算终止函数的梯度。
$$
\frac{\partial Q_{\Omega}(s,\omega)}{\partial \vartheta} = \sum_a \pi_{\omega,\theta}(a|s) \sum_{s'}\gamma P(s'|s,a) \frac{\partial U(\omega,s')}{\partial\vartheta}
$$
因此，关键量是 $U$ 的梯度。这是 call-and-return 执行的自然结果，其中终止函数的“优度”只能在进入下一个状态时进行评估。相关梯度可以进一步展开为：
$$
\frac{\partial U(\omega,s')}{\partial \vartheta} = - \frac{\part \beta_{\omega,\vartheta}(s')}{\part\vartheta} A_{\Omega}(s',\omega) + \gamma \sum_{\omega'}\sum_{s''}P(s'',\omega'|s',\omega) \frac{\part U(\omega',s'')}{\part\vartheta}
$$
其中 $A_{\Omega}$ 是相对于 options 的优势函数 $A_{\Omega}(s',\omega) = Q_{\Omega}(s',\omega) - V_{\Omega}(s')$ 。

**Theorem 2** (Termination Gradient Theorem)：给定一组马尔科夫 Options ，其随机终止函数在其参数 $\vartheta$ 中可微，期望折扣回报相对于 $\vartheta$ 和初始条件 $(s_1,\omega_0)$ 的梯度为
$$
- \sum_{s',\omega} \mu_{\Omega}(s',\omega|s_1,\omega_0) \frac{\part \beta_{\omega,\vartheta}(s')}{\part \vartheta} A_{\Omega}(s',\omega)
$$
其中 $\mu_{\Omega}(s',\omega|s_1,\omega_0)$ 是来自 $(s_1,\omega_0)$ 的 state-option 对的折扣权重： $\mu_{\Omega}(s,\omega|s_1,\omega_0) = \sum^{\infty}_{t=0}\gamma^t P(s_{t+1}=s,\omega_t=\omega|s_1,\omega_0)$ 。

终止梯度定理可以解释为提供基于梯度的中断贝尔曼算子。

## 5. Algorithms and Architecture

下图为 option-critic 架构图。

![](.\1.png)

option 内的策略、终止函数和 option 上的策略属于系统的 actor 部分，critic 部分由 $Q_U$ 和 $A_{\Omega}$ 组成。

Option-critic 架构没有规定如何获得 $\pi_{\Omega}$ ，因为可以应用各种现有方法：在 SMDP 级别使用策略梯度方法，在 options 模型上使用规划器，或者使用时间差分更新。如果 $\pi_{\Omega}$ 是对 options 的贪婪策略，则从（2）式可以得出，相应的一步 off-policy 更新目标 $g^{(1)}_t$ 为：
$$
g^{(1)}_t = r_{t+1} + \gamma ((1-\beta_{\omega_t,\vartheta}(s_{t+1}))\sum_a\pi_{\omega_t,\theta}(a|s_{t+1})Q_U(s_{t+1},\omega_t,a) + \\ \beta_{\omega_t,\vartheta}(s_{t+1}) \underset{\omega}{\max}\sum_a\pi_{\omega,\theta}(a|s_{t+1})Q_U(s_{t+1},\omega,a))
$$
这也是 option 内 Q-learning 算法的更新目标。算法 1 显示了使用 option 内 Q-learning 的 option-critic 架构实现。我们分别用 $\alpha$ ， $\alpha_{\theta}$ 和 $\alpha_{\vartheta}$ 表示 critic ， option 内策略和终止函数的学习率。

![](.\2.png)

除了 $Q_{\Omega}$ 之外，学习 $Q_U$ 在参数数量和样本数量方面都是计算上的浪费。一个实用的解决方案是仅学习 $Q_{\Omega}$ 并从中导出 $Q_U$ 的估计。因为 $Q_U$ 是对下一个状态的期望， $Q_U(s,\omega,a) = \mathbb{E}_{s' \sim P}[r(s,a)+\gamma U(\omega,s')|s,\omega,a]$ 。因此 $g^{(1)}_t$ 是一个合适的估计量。



