import random
import gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from Refuel_Env import *


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) # 队列,先进先出

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done)) # 将数据加入buffer

    def sample(self, batch_size): # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self): # 目前buffer中数据的数量
        return len(self.buffer)


class QNet(torch.nn.Module):
    # 只有一层隐藏层的网络
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x)) # 隐藏层使用ReLU激活函数
        return self.fc2(x)


class DQN:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, target_update, device):
        self.action_dim = action_dim
        self.q_net = QNet(state_dim, hidden_dim, action_dim).to(device) # Q网络
        self.target_q_net = QNet(state_dim, hidden_dim, action_dim).to(device) # 目标Q网络
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)

        self.gamma = gamma # 折扣因子
        self.epsilon = epsilon # epsilon-greedy
        self.target_update = target_update # 目标网络更新频率
        self.count = 0 # 计数器，记录更新次数
        self.device = device

    def take_action(self, state): # epsilon-greedy
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions) # Q值
        # 下个状态的最大Q值
        max_next_qvalues = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_qvalues * (1 - dones)
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets)) # 均方误差损失函数
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.count += 1


if __name__ == "__main__":
    lr = 2e-3
    num_episodes = 50
    hidden_dim = 128
    gamma = 0.98
    epsilon = 0.01
    target_update = 10
    buffer_size = 10000
    minimal_size = 500
    batch_size = 64
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    refuel_model = Refuel_Model()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    replay_buffer = ReplayBuffer(buffer_size)
    state_dim = refuel_model.state_dim
    action_dim = refuel_model.action_dim
    agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes)):
                episode_return = 0
                refuel_model = Refuel_Model()
                state = refuel_model.state
                done = False
                steps = 0
                while not done and steps < 30:
                    steps += 1
                    action = agent.take_action(state)
                    next_state, reward, done = refuel_model.transform(state, action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    # 当buffer数据的数量超过一定值后，才进行Q网络训练
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {
                            'states': b_s,
                            'actions': b_a,
                            'next_states': b_ns,
                            'rewards': b_r,
                            'dones': b_d
                        }
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)

    refuel_model = Refuel_Model()
    state = refuel_model.state
    done = False
    steps = 0
    while not done and steps < 30:
        steps += 1
        action = agent.take_action(state)
        print(refuel_model.actions[action])
        next_state, reward, done = refuel_model.transform(state, action)
        state = next_state

    plt.figure(1)
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on Refuel Model')
    plt.show()