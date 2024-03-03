# FuN

**FeUdal Networks for Hierarchical Reinforcement Learning**

## 1. Abstract

本文引入 Feudal Networks ，一种用于分层强化学习的新颖架构。此方法受到封建强化学习提议的启发，并通过在多个级别上解耦端到端学习来获得有效性——使其能够利用不同分辨率的时间。本文的框架使用了一个 Manager 模块和一个 Worker 模块。 Manager 以更低的时间分辨率运行，并设置由 Worker 传达和制定的抽象目标。 Worker 在环境的每个时刻处生成原始动作。 FuN 的解耦结构传达了几个好处——除了促进非常长的时间尺度信用分配之外，它还鼓励出现与 Manager 设置的不同目标相关的子策略。

## 2. Introduction

长期信用分配仍然是深度强化学习方法的主要挑战，尤其是在奖励信号稀疏的环境中。

本文提议的框架灵感来自封建强化学习，其中代理内的层次结构级别通过显式目标进行通信。封建强化学习的一些关键见解是目标可以以自上而下的方式生成，并且目标设置可以与目标实现解耦；层次结构中的级别与必须实现的下一层级别通信，但不指定如何做。以较低的时间分辨率做出更高级别的推理自然地将智能体行为构建到时间扩展的子策略中。

在这项工作中探索的架构是一个完全可微的神经网络，具有两级层次结构（尽管对更深层次的层次结构有明显的概括）。顶层， Manger ，在其本身学习的潜在状态空间中以较低时间分辨率设置目标。较低级别的 Worker 以更高的时间分辨率运行并产生原始动作，条件是它从 Manager 接收的目标。 Worker 在内在奖励的驱动下遵循目标。值得注意的是， Worker 和 Manager 之间没有梯度传播； Manager 仅从环境中接收起学习信号。换句话说， Manager 学会选择最大化外部奖励的潜在目标。

本文的主要贡献：（1）一个一致的端到端可微模型，它体现和推广了封建强化学习的原理。（2）用于训练 Manager 的新型近似过渡策略梯度更新。它利用了它产生的目标的语义。（3）使用方向性的目标而不是本质上的绝对目标。（4） Manager 的新型 RNN 设计——扩展 LSTM ——它扩展了循环状态记忆的寿命，并允许梯度在时间上通过大跳流动，从而实现通过数百个步骤的有效反向传播。

## 3. Related Work

此方法与 the options framework 之间的主要区别在于，在此方法中，顶层为底层生成有意义且明确的目标来实现。子目标作为潜在状态空间的方向出现，自然多样。

## 4. The Model

FuN 是一个模块化神经网络，由两个模块组成—— Worker 和 Manager 。

Manager 内部计算潜在状态表示并输出目标向量 $g_t$ 。 Worker 产生以外部观察、自己的状态和 Manager 目标为条件的动作。 Manager 和 Worker 共享一个感知模块，该模块从环境 $x_t$ 中获取观察并计算共享的中间表示 $z_t$ 。 Manager 目标 $g_t$ 由一个近似过渡策略梯度训练得到。这是一个特别有效的策略梯度训练形式，它利用了 Worker 的行为最终将与其设定的目标方向对齐的知识。然后， Worker 通过内在奖励进行训练，以产生实现这些目标方向的动作。下图为 FuN 的整体设计框架，以下等式描述了网络的前向动力学：

![](.\1.png)
$$
z_t = f^{percept}(x_t)
$$

$$
s_t = f^{Mspace}(z_t)
$$

$$
h^M_t, \hat{g}_t = f^{Mrnn}(s_t,h^M_{t-1}); \ g_t=\hat{g}_t/||\hat{g}_t||
$$

$$
w_t = \phi(\sum^t_{i=t-c}g_i)
$$

$$
w_t = \phi(\sum^t_{i=t-c}g_i)
$$

$$
\pi_t = SoftMax(U_tw_t)
$$

其中 Manager 和 Worker 都是循环的。这里 $h^M$ 和 $h^W$ 分别对应于 Manager 和 Worker 的内部状态。线性变化 $\phi$ 将目标 $g_t$ 映射到嵌入向量 $w_t \in R^k$ ，然后通过与矩阵 $U_t$ （Worker 的输出）的乘积组合以产生策略 $\pi$ ——原始动作的概率向量。

### 4.1 Goal Embedding

目标 $g$ 通过低维目标嵌入空间 $R^k,k<<d$ 中的乘法交互来调节策略。 Worker 首先为每个动作生成一个嵌入向量，由矩阵 $U \in R^{|a| \times k}$ 的行表示。为了合并来自 Manager 的目标，最后 $c$ 个目标首先通过求和汇集，然后使用线性投影 $\phi$ 嵌入到向量 $w \in R^k$ 中。投影 $\phi$ 是线性的，没有偏差，并使用来自 Worker 动作的梯度来学习。嵌入矩阵 $U$ 通过矩阵向量乘积与目标嵌入 $w$ 相结合。由于 $\phi$ 没有偏差，它永远无法产生恒定的非零向量——这是设置可以忽略 Manager 输入的唯一方式。这确保了 Manager 输出的目标总是影响最终策略。由于在多个时间步长上汇集目标，来自 Manager 的条件变化平稳。

### 4.2 Learning

考虑标准的强化学习设置，在每一步 $t$ ，智能体从环境中接收观察 $x_t$ ，并从有限的可能动作集中选择一个动作。环境响应一个新的观察 $x_t$ 和一个标量奖励 $r_t$ 。该过程一直持续到达到终端状态，然后重新启动它。智能体的目标是在 $\gamma \in [0,1]$ 的情况下最大化折扣回报 $R_t = \sum^{\infty}_{t=0}\gamma^k r_{t+k+1}$ 。智能体的行为由其动作选择策略 $\pi$ 定义。 FuN 在等式（6）中定义的可能动作（随机策略）上产生动作分布。

传统的方法是直接通过策略梯度或者 TD 学习进行梯度下降来整体训练整个架构。由于 FuN 是完全可微的，我们可以使用对 Worker 采取的行动进行的策略梯度算法进行端到端训练。 Manager 的输出 $g$ 将通过来自 Worker 的梯度来训练。然而，这将剥夺 Manager 对任何语义含义的目标 $g$ ，使它们只是模型的内部潜在变量。相反，我们建议独立训练 Manager 来预测状态空间中的有利方向（转换），并内在地奖励 Worker 遵循这些方向。如果 Worker 能够满足在这些方向上移动的目标，那么我们应该最终通过状态空间采取有利的轨迹。我们在 Manager 的以下更新规则中形式化这一点：
$$
\nabla g_t = A^M_t \nabla_{\theta} d_{cos}(s_{t+c} - s_t, g_t(\theta))
$$
其中 $A^M_t = R_t - V^M_t(x_t,\theta)$ 是 Manager 的优势函数，使用内部 critic 的值函数估计 $V^M_t(x_t,\theta)$ 计算； $d_{cos}(\alpha,\beta) = \alpha^T \beta / (|\alpha||\beta|)$ 是两个向量之间的余弦相似度。在计算 $\nabla_{\theta}d_{cos}$ 时， $s$ 对 $\theta$ 的依赖性被忽略。现在 $g_t$ 在定义 Manager 时间分辨率的视界 $c$ 处将语义含义作为潜在状态空间的一个有利方向。

激励 Worker 遵循目标的内在奖励定义为：
$$
r^I_t = 1/c \sum^c_{i=1} d_{cos}(s_t-s_{t-i},g_{t-i})
$$
我们使用方向，因为 Worker 能够可靠地产生潜在状态的方向偏移比假设 Worker 可以让我们到达任意新的绝对位置更可行。它还为目标提供了一定程度的不变性，并允许结构泛化——相同的方向子目标 $g$ 可以调用在潜在状态空间的很大一部分中有效和有用的子策略。

原始的封建强化学习公式主张从较低级别的层次结构中完全隐藏奖励。在实践中，我们通过为遵循目标添加内在奖励来采用更柔和的方法，但也保留了环境奖励。然后以最大化加权和 $R_t + \alpha R^I_t$ 训练 Worker ，其中 $\alpha$ 是调节内在奖励影响的超参数。通过使用任何现成的深度强化学习算法，可以训练 Worker 策略 $\pi$ 以最大化内在奖励。在这里，我们使用 A2C 方法：
$$
\nabla \pi_t = A^D_t \nabla_{\theta} \log \pi(a_t|x_t;\theta)
$$
优势函数 $A^D_t = (R_t + \alpha R^I_t - V^D_t(x_t;\theta))$ 是使用内部 critic 计算的，它估计两个奖励的价值函数。

请注意， Worker 和 Manager 可能有不同的折扣因子 $\gamma$ 来计算回报。例如，这允许 Worker 更贪婪，并专注于即时奖励，而 Manager 可以考虑长期视角。

### 4.3 Transition Policy Gradients

我们现在激励提出的 Manager 更新规则，作为关于 Worker 行为模型的新型策略梯度形式。考虑一个高级策略 $o_t = \mu(s_t,\theta)$ ，它在子策略中进行选择，我们假设这些子策略是固定的持续时间行为（对于 $c$ 步持续）。对应于每个子策略是一个转换分布 $p(s_{t+c}|s_t,o_t)$ ，它描述了给定起始状态和子策略制定的情况下，我们最终在子策略结束时的状态分布。高级策略可以由转换分布组成，以给出描述给定起始状态结束状态的分布的“转换策略“ $\pi^{TP}(s_{t+c}|s_t) = p(s_{t+c}|s_t,\mu(s_t,\theta))$ 。将此称为策略是有效的，因为原始 MDP 与具有策略 $\pi^{TP}$ 和转换函数 $s_{t+c} = \pi^{TP}(s_t)$ 的新 MDP 同构。因此，我们可以将策略梯度定理应用于转换策略 $\pi^{TP}$ ，从而找到关于策略参数的性能梯度。
$$
\nabla_{\theta} \pi^{TP}_t = \mathbb{E}[(R_t-V(s_t)) \nabla_{\theta}\log p(s_{t+c}|s_t,\mu(s_t,\theta))]
$$
一般来说， Worker 可能遵循复杂的轨迹。策略梯度的一个简单应用要求智能体从这些轨迹的样本中学习。但是如果我们知道这些轨迹可能最终的位置，通过对转换进行建模，那么我们可以直接跳过 Worker 的行为，而是遵循预测转换的策略梯度。 FuN 假设转换模型的特定形式为：状态空间 $s_{t+c}-s_t$ 中的方向遵循 von Mises-Fisher 分布。具体来说，如果 von Mises-Fisher 分布的平均方向由 $g(o_t)$ 给出，我们将有 $p(s_{t+c}|s_t,o_t) \propto e^{d_{cos}(s_{t+c}-s_t,g_t)}$ 。如果这个函数形式确实是正确的，那么我们提出的 Manager 更新启发式（7）实际上是式（10）转换策略梯度的正确形式。

注意， Worker 的内在奖励基于状态轨迹的对数似然。通过这种方式， FuN 架构积极鼓励转换模型的功能形式保持正确。由于 Worker 正在学习实现 Manager 的方向，其转换应该随着时间的推移密切遵循这个方向的分布，因此我们对转换策略梯度的近似应该相当好。

## 5. Architecture Details

感知模块 $f^{percept}$ 是一个卷积神经网络，后跟一个全连接层。每个卷积层和全连接层后面都有一个整流器的非线性。 Manager 在制定目标时隐式建模的状态空间是通过 $f^{Mspace}$ 计算的， $f^{Mspace}$ 是另一个全连接层，后跟一个整流器非线性。嵌入向量 $w$ 的维度设置为 $k=16$ 。为了鼓励在转换策略中进行探索，在每一步我们都会以一个很小的概率 $\epsilon$ 发出从单变量高斯分布中采样的随机目标。

Worker 的循环神经网络 $f^{Wrnn}$ 是一个标准 LSTM 。对于 Manager 的循环神经网络 $f^{Mrnn}$ ，我们提出了一种新颖的设计——扩张 LSTM 。

### 5.1 Dilated LSTM

我们为 Manager 提出了一种新颖的 RNN 架构，该架构以比数据流更低的时间分辨率运行。我们定义了一个类似于扩张卷积神经网络的扩张 LSTM 。对于膨胀半径 $r$ ，设网络的完整状态为 $h= \{ \hat{h}^i \} ^r_{i=1}$ ，即它由 $r$ 个单独的子状态组或“核”组成。在时间 $t$ ，网络由以下方程控制： $\hat{h}^{t\%r}_t, g_t=LSTM(s_t,\hat{h}^{t\%r}_{t-1};\theta^{LSTM})$ ，其中 $\%$ 表示模运算，允许我们指示当前更新哪组核。我们明确使 LSTM 网络的参数 $\theta^{LSTM}$ 强调同一组参数支配 dLSTM 中 $r$ 个组中的每一个的更新。

在每个时间步，只更新状态的相应部分，并在之前的 $c$ 个输出中汇集输出。这允许 dLSTM 内的 $r$ 组内核来保留长时间的记忆，但整个 dLSTM 仍然能够从每个输入经验中处理和学习，并且还能够在每一步更新其输出。这个想法类似于 clockwork RNNs ，但是顶层 ticks 的速度是固定的、缓慢的，而 dLSTM 则观察所有可用的训练数据。

## 6. Experiments

**Baseline**：主要基线是 CNN 学习的表示之上的循环 LSTM 网络。 LSTM 架构是一种广泛使用的循环网络，它已被证明在一系列强化学习问题上表现得非常好。LSTM 输入是观察的特征表示和智能体的前一个动作，动作概率和价值函数估计从其隐藏状态回归。

**Optimisaion**：对于所有强化学习试验，使用 A3C 方法。

### 6.1 Montezuma's Revenge

### 6.2 ATARI

### 6.3 Memory in Labyrinth

### 6.4 Ablative Analysis

本节验证了本文的主要创新：用于训练 Manager 的转换策略梯度；相对目标而不是绝对目标； Manager 的低时间分辨率； Worker 的内在动机。

### 6.5 ATARI Action Repeat Transfer

## 7. Discussion and Future Work

如何创建可以学习将行为分解为有意义的原语的智能体，然后重用它们更有效地获取新行为是一个长期存在的研究问题。本文介绍了 Feudal Networks，这是一种新颖的架构，它将子目标制定为潜在状态空间的方向，之后，将其转换为有意义的行为原语。 FuN 清楚地将发现和设置子目标的模块与通过原始动作生成行为的模块分开。这创建了一个稳定的自然层次结构，并允许两个模块以互补的方式学习。实验清楚地表明，这使得长期信用分配和记忆更容易处理。这也为进一步研究开辟了许多途径，例如：可以通过在多个时间尺度设置目标来构建更深层次的层次结构，将代理扩展到真正具有稀疏奖励和部分可观察性的大型环境。 FuN 的模块化结构也适用于迁移和多任务学习——学习到的行为原语可以重新用于获取新的复杂技能，或者 Manager 的转换策略可以转移到具有不同体现的代理。