import torch
from torch import nn
from torch.nn.functional import cosine_similarity as d_cos, normalize
from utils import init_hidden, init_weight
from dilated_lstm import DilatedLSTM
from preprocess import Preprocessor


class FeudalNet(nn.Module):
    def __init__(self, num_workers, input_dim, hidden_dim_manager, hidden_dim_worker,
                 n_actions, time_horizon=10, dilation=10, device='cpu', args=None):
        super(FeudalNet, self).__init__()
        self.b = num_workers
        self.c = time_horizon
        self.d = hidden_dim_manager
        self.k = hidden_dim_worker
        self.r = dilation
        self.n_actions = n_actions
        self.device = device

        self.preprocessor = Preprocessor(input_dim, device)
        self.precept = Perception(input_dim[-1], self.d)
        self.manager = Manager(self.c, self.d, self.r, args, device)
        self.worker = Worker(self.b, self.c, self.d, self.k, n_actions, device)

        self.hidden_m = init_hidden(args.num_workers, self.r * self.d, device=device, grad=True)
        self.hidden_w = init_hidden(args.num_workers, self.k * n_actions, device=device, grad=True)

        self.args = args
        self.to(device)
        self.apply(init_weight)

    def forward(self, x, goals, states, mask, save=True):
        """
        A forward pass through the whole feudal network

        Order of operations:
        1. Input goes through a preprocessor to normalize and put on device
        2. Normalized input goes to the perception module resulting in a state
        3. State is input for manager which produces a goal
        4. State and goal is both input for worker which produces an action distribution

        Args:
            x(np.ndarray): observation from the environment
            goals(list): list of goal tensors, length = 2 * r + 1
            states(list): list of state tensors, length = 2 * r + 1
            mask(tensor): mask discribing for each worker if episode is done
            save(bool, optional): if we are calculating next_v, we do not store rnn states. Defaults to True.
        """

        x = torch.FloatTensor(x).unsqueeze(0)
        # x = torch.FloatTensor(x).unsqueeze(0).to(self.device)
        # print("x:", x)
        x = self.preprocessor(x)
        # print("preprocessed_x:", x)
        z = self.precept(x)
        # print("z:", z)

        goal, hidden_m, state, value_m = self.manager(z, self.hidden_m, mask)

        # Ensure that we only hava a list of size 2*c + 1, and we use FiLo
        if len(goals) > (2 * self.c + 1):
            goals.pop(0)
            states.pop(0)

        goals.append(goal)
        states.append(state.detach())  # state never have gradients active

        print("goals:", goals)

        # The manager is ahead at least c steps, so we feed only the first c + 1 states to worker
        action_dist, hidden_w, value_w = self.worker(z, goals[:self.c + 1], self.hidden_w, mask)

        # print("action_dist:", action_dist)

        if save:
            self.hidden_m = hidden_m
            self.hidden_w = hidden_w

        return action_dist, goals, states, value_m, value_w

    def intrinsic_reward(self, states, goals, masks):
        return self.worker.intrinsic_reward(states, goals, masks)

    def state_goal_cosine(self, states, goals, masks):
        return self.manager.state_goal_cosine(states, goals, masks)

    def repackage_hidden(self):
        def repackage_rnn(x):
            return [item.detach() for item in x]

        self.hidden_w = repackage_rnn(self.hidden_w)
        self.hidden_m = repackage_rnn(self.hidden_m)

    def init_obj(self):
        template = torch.zeros(self.b, self.d)
        goals = [torch.zeros_like(template).to(self.device) for _ in range(2 * self.c + 1)]
        states = [torch.zeros_like(template).to(self.device) for _ in range(2 * self.c + 1)]
        masks = [torch.ones(self.b, 1).to(self.device) for _ in range(2 * self.c + 1)]
        return goals, states, masks


class Perception(nn.Module):
    def __init__(self, input_dim, d):
        super(Perception, self).__init__()
        self.percept = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, d),
            nn.ReLU())

    def forward(self, x):
        return self.percept(x)


class Manager(nn.Module):
    def __init__(self, c, d, r, args, device):
        super(Manager, self).__init__()
        self.c = c  # Time horizon
        self.d = d  # Hidden dimension size
        self.r = r  # Dilation level
        self.eps = args.eps
        self.device = device

        self.Mspace = nn.Linear(self.d, self.d)
        self.Mrnn = DilatedLSTM(self.d, self.d, self.r)
        self.critic = nn.Linear(self.d, 1)

    def forward(self, z, hidden, mask):
        state = self.Mspace(z).relu()
        hidden = (mask * hidden[0], mask * hidden[1])
        goal_hat, hidden = self.Mrnn(state, hidden)
        value_est = self.critic(goal_hat)

        goal = normalize(goal_hat)
        state = state.detach()

        if self.eps > torch.rand(1)[0]:
            # To encourage exploration in transition policy,
            # at every step with a small probability ε
            # we emit a random goal sampled from a uni-variate Gaussian.
            goal = torch.randn_like(goal, requires_grad=False)

        return goal, hidden, state, value_est

    def state_goal_cosine(self, states, goals, masks):
        # Compute the cosine similarity between the current state and the goal
        # cos(S_{t+c} - S_{t}, G_{t})
        t = self.c
        mask = torch.stack(masks[t: t + self.c - 1]).prod(dim=0)
        cosine_dist = d_cos(states[t + self.c] - states[t], goals[t])
        cosine_dist = mask * cosine_dist.unsqueeze(-1)
        return cosine_dist


class Worker(nn.Module):
    def __init__(self, b, c, d, k, num_actions, device):
        super(Worker, self).__init__()
        self.b = b
        self.c = c
        self.k = k
        self.num_actions = num_actions
        self.device = device

        self.Wrnn = nn.LSTMCell(d, k * self.num_actions)
        self.phi = nn.Linear(d, k, bias=False)
        self.critic = nn.Sequential(
            nn.Linear(k * self.num_actions, 50),
            nn.ReLU(),
            nn.Linear(50, 1))

    def forward(self, z, goals, hidden, mask):
        hidden = (mask * hidden[0], mask * hidden[1])

        u, cx = self.Wrnn(z, hidden)
        hidden = (u, cx)

        # Detaching is vital, no end to end training
        goals = torch.stack(goals).detach().sum(dim=0)
        w = self.phi(goals)
        value_est = self.critic(u)

        u = u.reshape(u.shape[0], self.k, self.num_actions)

        # print("w:", w)
        # print("u:", u)

        a = torch.einsum("bk, bka -> ba", w, u).softmax(dim=-1)

        return a, hidden, value_est

    def intrinsic_reward(self, states, goals, masks):
        t = self.c
        r_i = torch.zeros(self.b, 1).to(self.device)
        mask = torch.ones(self.b, 1).to(self.device)

        for i in range(1, self.c + 1):
            r_i_t = d_cos(states[t] - states[t - i], goals[t - i]).unsqueeze(-1)
            r_i += (mask * r_i_t)
            mask = mask * masks[t - i]

        r_i = r_i.detach()
        return r_i / self.c


def feudal_loss(storage, next_v_m, next_v_w, args):
    # Calculate the loss for Worker and Manager
    # with timesteps T, batch size B and hidden dim D

    # Discount rewards, both of size B * T
    ret_m = next_v_m
    ret_w = next_v_w

    storage.placeholder()  # Fill ret_m, ret_w with empty vals
    for i in reversed(range(args.num_steps)):
        ret_m = storage.r[i] + args.gamma_m * ret_m * storage.m[i]
        ret_w = storage.r[i] + args.gamma_w * ret_w * storage.m[i]
        storage.ret_m[i] = ret_m
        storage.ret_w[i] = ret_w
    storage.normalize(['ret_w', 'ret_m'])

    rewards_intrinsic, value_m, value_w, ret_w, ret_m, logps, entropy, \
        state_goal_cosines = storage.stack(
        ['r_i', 'v_m', 'v_w', 'ret_w', 'ret_m', 'logp', 'entropy', 's_goal_cos'])

    # Calculate advantages, size B * T
    advantage_w = ret_w + args.alpha * rewards_intrinsic - value_w
    advantage_m = ret_m - value_m

    loss_worker = (logps * advantage_w.detach()).mean()
    loss_manager = (state_goal_cosines * advantage_m.detach()).mean()

    # Update the critics into the right direction
    value_w_loss = 0.5 * advantage_w.pow(2).mean()
    value_m_loss = 0.5 * advantage_m.pow(2).mean()

    entropy = entropy.mean()

    loss = - loss_worker - loss_manager + value_m_loss + value_w_loss - args.entropy_coef * entropy

    print("loss:", loss)

    return loss
