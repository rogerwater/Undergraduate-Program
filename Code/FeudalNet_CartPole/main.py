import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from net import feudal_networks
from replay_buffer import replay_buffer
import gym
from copy import deepcopy
from option_critic import OptionCriticFeatures
from option_critic import critic_loss as critic_loss_fn
from option_critic import actor_loss as actor_loss_fn
from experience_replay import ReplayBuffer
from utils import to_tensor
import time
import matplotlib.pyplot as plt
import argparse


class feudal_model(object):
    def __init__(self, env, capacity, update_freq, episode, feature_dim, k_dim, dilation, horizon_c, learning_rate, alpha, gamma, entropy_weight, render):
        # * feature_dim >> k_dim
        # * dilation == horizon_c
        # * capacity <= update_freq
        self.env = env
        self.capacity = capacity
        self.update_freq = update_freq
        self.episode = episode
        self.feature_dim = feature_dim
        self.k_dim = k_dim
        self.dilation = dilation
        self.horizon_c = horizon_c
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.gamma = gamma
        self.entropy_weight = entropy_weight
        self.render = render

        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.net = feudal_networks(self.observation_dim, self.feature_dim, self.k_dim, self.action_dim, self.dilation, self.horizon_c)
        self.buffer = replay_buffer(self.capacity)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.h_m = torch.zeros([1, self.feature_dim])
        self.c_m = torch.zeros([1, self.feature_dim])
        self.h_w = torch.zeros([1, self.action_dim * self.k_dim])
        self.c_w = torch.zeros([1, self.action_dim * self.k_dim])
        self.count = 0
        self.weight_reward = None

    def get_returns(self, rewards, dones, values):
        returns = []
        run_return = values[-1]
        for i in reversed(range(rewards.size(0))):
            run_return = rewards[i] + self.gamma * run_return * (1. - dones[i])
            returns.append(run_return)
        returns = list(reversed(returns))
        returns = torch.cat(returns, dim=0).unsqueeze(1)
        return returns

    def train(self):
        # * Need to notice that the valid range of samples is [horizon_c: - horizon_c]
        observations, mstates, goals, m_values, policies, w_values_int, w_values_ext, rewards_ext, dones, actions = self.buffer.sample()

        actions = torch.LongTensor(actions)
        observations = torch.FloatTensor(np.vstack(observations))
        dones = torch.FloatTensor(dones)
        rewards_ext = torch.FloatTensor(rewards_ext)
        m_values = torch.cat(m_values, 0)
        policies = torch.cat(policies, 0)
        w_values_ext = torch.cat(w_values_ext, 0)
        w_values_int = torch.cat(w_values_int, 0)

        rewards_int = []
        for i in range(self.horizon_c, observations.size(0)):
            s = mstates[i]
            reward_int = 0
            for j in range(self.horizon_c):
                s_ = mstates[i - j - 1]
                g_ = goals[i - j - 1]
                reward_int += F.cosine_similarity(s - s_, g_)
            reward_int = reward_int / self.horizon_c
            rewards_int.append(reward_int)
        rewards_int = torch.cat(rewards_int, 0).unsqueeze(1)

        m_returns = self.get_returns(rewards_ext, dones, m_values)
        w_returns_ext = self.get_returns(rewards_ext, dones, w_values_ext)
        w_returns_int = self.get_returns(rewards_int, dones[self.horizon_c:], w_values_int[self.horizon_c:])

        m_adv = m_returns - m_values
        w_ext_adv = w_returns_ext - w_values_ext
        w_int_adv = w_returns_int[: -self.horizon_c, :] - w_values_int[self.horizon_c: -self.horizon_c, :]

        m_loss = []
        for i in range(0, observations.size(0) - self.horizon_c):
            s_ = mstates[i]
            s = mstates[i + self.horizon_c]
            g = goals[i]
            cos_sim = F.cosine_similarity(s - s_, g)
            m_loss.append(- m_adv[i].detach() * cos_sim)
        m_loss = torch.cat(m_loss, 0).unsqueeze(1)

        dists = torch.distributions.Categorical(policies)
        log_probs = dists.log_prob(actions)
        w_loss = - (w_ext_adv[self.horizon_c: -self.horizon_c, :] + self.alpha * w_int_adv).detach() * log_probs.unsqueeze(1)[self.horizon_c: -self.horizon_c, :] - self.entropy_weight * dists.entropy().unsqueeze(1)[self.horizon_c: -self.horizon_c]

        m_returns = m_returns[self.horizon_c: -self.horizon_c, :]
        m_values = m_values[self.horizon_c: -self.horizon_c, :]
        w_returns_ext = w_returns_ext[self.horizon_c: -self.horizon_c, :]
        w_values_ext = w_values_ext[self.horizon_c: -self.horizon_c, :]
        w_returns_int = w_returns_int[: -self.horizon_c, :]
        w_values_int = w_values_int[self.horizon_c: -self.horizon_c, :]
        m_loss = m_loss[self.horizon_c:, :]

        m_critic_loss = (m_returns.detach() - m_values).pow(2)
        w_critic_ext_loss = (w_returns_ext.detach() - w_values_ext).pow(2)
        w_critic_int_loss = (w_returns_int.detach() - w_values_int).pow(2)

        loss = m_loss + m_critic_loss + w_loss + w_critic_ext_loss + w_critic_int_loss
        loss = loss.mean()

        torch.autograd.set_detect_anomaly(True)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
        self.optimizer.step()

    def run(self):
        print("Running CartPole on FeudalNet...")
        episode_reward = []
        for i in range(self.episode):
            obs = self.env.reset()[0]
            steps_num = 0
            total_reward = 0
            if self.render:
                self.env.render()
            while True:
                # * manager change the goal every horizon_c steps
                if self.count % self.horizon_c == 0:
                    mstate, goal, m_hidden_new, m_value = self.net.get_goal(torch.FloatTensor(np.expand_dims(obs, 0)), (self.h_m, self.c_m), self.count)
                    self.net.store_goal(goal)
                policy, w_hidden_new, w_value_int, w_value_ext = self.net.get_policy(torch.FloatTensor(np.expand_dims(obs, 0)), (self.h_w, self.c_w))
                self.h_m, self.c_m = m_hidden_new
                self.h_w, self.c_w = w_hidden_new
                dist = torch.distributions.Categorical(policy)
                action = dist.sample().detach().item()
                next_obs, reward, done, info, _  = self.env.step(action)
                steps_num += 1
                total_reward += reward
                self.count += 1
                self.buffer.store(obs, mstate, goal, m_value, policy, w_value_int, w_value_ext, reward, done, action)
                obs = next_obs

                if self.count % self.update_freq == 0:
                    self.train()

                if done or steps_num > 18000:
                    if not self.weight_reward:
                        self.weight_reward = total_reward
                    else:
                        self.weight_reward = 0.99 * self.weight_reward + 0.01 * total_reward
                    print('episode: {}  reward: {}  weight_reward: {:.2f}'.format(i + 1, total_reward, self.weight_reward))
                    break
            episode_reward.append(total_reward)
        return episode_reward


parser = argparse.ArgumentParser(description="Option Critic")
parser.add_argument('--env', default='CartPole-v0', help='ROM to run')
parser.add_argument('--optimal-eps', type=float, default=0.05, help='Epsilon when playing optimally')
parser.add_argument('--learning-rate', type=float, default=.001, help='Learning rate')
parser.add_argument('--gamma', type=float, default=.99, help='Discount rate')
parser.add_argument('--epsilon-start', type=float, default=1.0, help=('Starting value for epsilon'))
parser.add_argument('--epsilon-min', type=float, default=.1, help='Minimum epsilon.')
parser.add_argument('--epsilon-decay', type=float, default=20000, help=('Number of steps to minimum epsilon'))
parser.add_argument('--max-history', type=int, default=10000, help=('Maximum number of steps stored in replay'))
parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
parser.add_argument('--freeze-interval', type=int, default=200, help=('Interval between target freezes'))
parser.add_argument('--update-frequency', type=int, default=30, help=('Number of actions before each SGD update'))
parser.add_argument('--termination-reg', type=float, default=0.01, help=('Regularization to decrease termination prob'))
parser.add_argument('--entropy-reg', type=float, default=0.01, help=('Regularization to increase policy entropy'))
parser.add_argument('--num-options', type=int, default=2, help=('Number of options to create'))
parser.add_argument('--temp', type=float, default=1, help='Action distribution softmax tempurature param')

parser.add_argument('--max_steps_ep', type=int, default=18000, help='Number of maximum steps per episode')
# parser.add_argument('--max_steps_total', type=int, default=50000, help='Number of maximum steps to take')
parser.add_argument('--cuda', type=bool, default=True, help='Enable CUDA training (recommended if possible).')
parser.add_argument('--seed', type=int, default=0, help='Random seed for numpy, torch, random.')
parser.add_argument('--logdir', type=str, default='runs', help='Directory for logging statistics')
parser.add_argument('--exp', type=str, default=None, help='Optional experiment name')
parser.add_argument('--max_episode', type=int, default=600, help='Number of maximum episodes')


def run_option_critic(args):
    print("Running CartPole on Option Critic...")
    env = gym.make(args.env)
    option_critic = OptionCriticFeatures
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    option_critic = option_critic(
        in_features=env.observation_space.shape[0],
        num_actions=env.action_space.n,
        device=device,
        num_options=args.num_options,
        temperature=args.temp,
        eps_start=args.epsilon_start,
        eps_min=args.epsilon_min,
        eps_decay=args.epsilon_decay,
        eps_test=args.optimal_eps
    )

    # Create a prime network for more stable Q values
    option_critic_prime = deepcopy(option_critic)

    optim = torch.optim.RMSprop(option_critic.parameters(), lr=args.learning_rate)

    buffer = ReplayBuffer(capacity=args.max_history, seed=args.seed)

    steps = 0
    episode_num = 0
    episode_reward = []
    while episode_num < args.max_episode:
        rewards = 0
        option_lengths = {opt: [] for opt in range(args.num_options)}
        obs = env.reset()[0]
        state = option_critic.get_state(to_tensor(obs))
        greedy_option = option_critic.greedy_option(state)
        current_option = 0

        done = False
        ep_steps = 0
        option_termination = True
        curr_op_len = 0
        while not done and ep_steps < args.max_steps_ep:
            epsilon = option_critic.epsilon

            if option_termination:
                option_lengths[current_option].append(curr_op_len)
                current_option = np.random.choice(args.num_options) if np.random.rand() < epsilon else greedy_option
                curr_op_len = 0

            action, logp, entropy = option_critic.get_action(state, current_option)

            next_obs, reward, done, info, _ = env.step(action)

            env.render()

            buffer.push(obs, current_option, reward, next_obs, done)
            rewards += reward

            if len(buffer) > args.batch_size:
                actor_loss = actor_loss_fn(obs, current_option, logp, entropy,
                                           reward, done, next_obs, option_critic, option_critic_prime, args)
                loss = actor_loss

                if steps % args.update_frequency == 0:
                    data_batch = buffer.sample(args.batch_size)
                    critic_loss = critic_loss_fn(option_critic, option_critic_prime, data_batch, args)
                    loss += critic_loss

                optim.zero_grad()
                loss.backward()
                optim.step()

                if steps % args.freeze_interval == 0:
                    option_critic_prime.load_state_dict(option_critic.state_dict())

            state = option_critic.get_state(to_tensor(next_obs))
            option_termination, greedy_option = option_critic.predict_option_termination(state, current_option)

            # update global steps etc
            steps += 1
            ep_steps += 1
            curr_op_len += 1
            obs = next_obs

        episode_reward.append(rewards)
        print('episode: {}  reward: {}'.format(episode_num + 1, rewards))
        episode_num += 1

    env.close()

    return episode_reward


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env = env.unwrapped
    args = parser.parse_args()
    test = feudal_model(
        env=env,
        capacity=30,
        update_freq=30,
        episode=600,
        feature_dim=128,
        k_dim=8,
        dilation=10,
        horizon_c=10,
        learning_rate=1e-3,
        alpha=0.5,
        gamma=0.99,
        entropy_weight=1e-4,
        render=False
    )
    feudalnet_reward = test.run()
    option_critic_reward = run_option_critic(args)
    feudalnet_reward = np.array(feudalnet_reward)
    option_critic_reward = np.array(option_critic_reward)

    plt.plot(feudalnet_reward, label='Feudal Networks')
    plt.plot(option_critic_reward, label='Option Critic')
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Performance on CartPole")
    plt.legend()
    plt.show()