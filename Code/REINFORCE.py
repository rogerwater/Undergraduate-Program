import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from Refuel_Env import *
from MAXQ import RefuelEnv

class PGNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim1, hidden_dim2, hidden_dim3, action_dim):
        super(PGNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim2)
        self.fc2 = torch.nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = torch.nn.Linear(hidden_dim2, hidden_dim3)
        self.fc4 = torch.nn.Linear(hidden_dim3, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return F.softmax(self.fc4(x), dim=1)


class REINFORCE:
    def __init__(self, state_dim, hidden_dim1, hidden_dim2, hidden_dim3,action_dim, learning_rate, gamma, device):
        self.pgnet = PGNet(state_dim, hidden_dim1, hidden_dim2, hidden_dim3,action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.pgnet.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.device = device

    def take_action(self, state): # 根据动作概率分布随机采样
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.pgnet(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']

        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(reward_list))): # 从最后一步算起
            reward = reward_list[i]
            state = torch.tensor([state_list[i]], dtype=torch.float).to(self.device)
            action = torch.tensor([action_list[i]]).view(-1, 1).to(self.device)
            log_probs = torch.log(self.pgnet(state).gather(1, action))
            G = self.gamma * G + reward
            loss = -log_probs * G # 每一步的损失函数
            loss.backward() # 反向传播计算梯度
        self.optimizer.step()


if __name__ == "__main__":
    learning_rate = 1e-3
    num_episodes = 50
    hidden_dim = 128
    gamma = 0.98
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    refuel_model = Refuel_Model()
    torch.manual_seed(0)
    state_dim = refuel_model.state_dim
    action_dim = refuel_model.action_dim
    agent = REINFORCE(state_dim, hidden_dim, hidden_dim, hidden_dim, action_dim, learning_rate, gamma, device)

    return_list = []
    for i in range(50):
        with tqdm(total=int(num_episodes), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes)):
                episode_return = 0
                transition_dict = {
                    'states': [],
                    'actions': [],
                    'next_states': [],
                    'rewards': [],
                    'dones': []
                }
                refuel_model = Refuel_Model()
                state = refuel_model.state
                done = False
                steps = 0
                while not done:
                    steps += 1
                    action = agent.take_action(state)
                    next_state, reward, done = refuel_model.transform(state, action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
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
        print(refuel_model.states[0], ':', state[0], ', ', refuel_model.states[1], ':', state[1], ', ', refuel_model.states[2], ':', state[2])
        state_tensor = torch.tensor([state], dtype=torch.float).to(agent.device)
        action = agent.pgnet(state_tensor).argmax()
        print(refuel_model.actions[action])
        next_state, reward, done = refuel_model.transform(state, action)
        state = next_state

    episode_list = list(range(len(return_list)))
    plt.plot(episode_list, return_list)
    plt.xlabel("Episodes")
    plt.ylabel("Returns")
    plt.title("REINFORCE on Refuel Model")
    plt.show()