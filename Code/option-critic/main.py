import numpy as np
import argparse
import torch
from copy import deepcopy

from option_critic import OptionCriticFeatures
from option_critic import critic_loss as critic_loss_fn
from option_critic import actor_loss as actor_loss_fn
from actor_critic import ActorCriticFeatures


from experience_replay import ReplayBuffer
from utils import to_tensor
from logger import Logger

import time
import gym
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Option Critic")
parser.add_argument('--env', default='CartPole-v0', help='ROM to run')
parser.add_argument('--optimal-eps', type=float, default=0.05, help='Epsilon when playing optimally')
parser.add_argument('--learning-rate', type=float, default=.0005, help='Learning rate')
parser.add_argument('--gamma', type=float, default=.99, help='Discount rate')
parser.add_argument('--epsilon-start', type=float, default=1.0, help=('Starting value for epsilon'))
parser.add_argument('--epsilon-min', type=float, default=.1, help='Minimum epsilon.')
parser.add_argument('--epsilon-decay', type=float, default=20000, help=('Number of steps to minimum epsilon'))
parser.add_argument('--max-history', type=int, default=10000, help=('Maximum number of steps stored in replay'))
parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
parser.add_argument('--freeze-interval', type=int, default=200, help=('Interval between target freezes'))
parser.add_argument('--update-frequency', type=int, default=4, help=('Number of actions before each SGD update'))
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
parser.add_argument('--max_episode', type=int, default=1000, help='Number of maximum episodes')


def run_option_critic(args):
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

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    buffer = ReplayBuffer(capacity=args.max_history, seed=args.seed)

    steps = 0
    episode_num = 0
    episode_reward = []
    while episode_num < args.max_episode:
        rewards = 0
        option_lengths = {opt: [] for opt in range(args.num_options)}
        obs = env.reset()
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

            next_obs, reward, done, _ = env.step(action)

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
        episode_num += 1

    env.close()

    return episode_reward


def run_actor_critic(args):
    env = gym.make(args.env)
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

    actor_critic = ActorCriticFeatures(
        env.observation_space.shape[0],
        128,
        env.action_space.n,
        1e-3,
        1e-2,
        args.gamma,
        device
    )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    episode_reward = []
    episode_num = 0
    while episode_num < args.max_episode:
        episode_rewards = 0
        transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
        state = env.reset()
        done = False
        while not done:
            action = actor_critic.take_action(state)
            next_state, reward, done, _ = env.step(action)
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)
            state = next_state
            episode_rewards += reward
        episode_reward.append(episode_rewards)
        actor_critic.update(transition_dict)
        episode_num += 1

    return episode_reward


if __name__ == "__main__":
    args = parser.parse_args()
    episode_reward_oc = run_option_critic(args)
    episode_reward_ac = run_actor_critic(args)
    episode_reward_oc = np.array(episode_reward_oc)
    episode_reward_ac = np.array(episode_reward_ac)

    np.save('episode_reward_oc_cartpole.npy', episode_reward_oc)
    np.save('episode_reward_ac_cartpole.npy', episode_reward_ac)


