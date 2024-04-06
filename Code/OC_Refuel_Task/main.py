import numpy as np
import argparse
import torch
from copy import deepcopy
import time
from option_critic import OptionCriticFeatures
from option_critic import critic_loss as critic_loss_fn
from option_critic import actor_loss as actor_loss_fn
from model import RefuelingEnv, UncoverEnv
from constraint_lib import ConstraintLibrary, PotentialBasedRewardShaping
from add_advice import add_advice
from experience_replay import ReplayBuffer
from utils import to_tensor
from datetime import datetime
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Option Critic")
parser.add_argument('--env', default='CartPole-v0', help='ROM to run')
parser.add_argument('--optimal-eps', type=float, default=0.05, help='Epsilon when playing optimally')
parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--gamma', type=float, default=.99, help='Discount rate')
parser.add_argument('--epsilon-start', type=float, default=1, help=('Starting value for epsilon'))
parser.add_argument('--epsilon-min', type=float, default=0.1, help='Minimum epsilon.')
parser.add_argument('--epsilon-decay', type=float, default=5000, help=('Number of steps to minimum epsilon'))
parser.add_argument('--max-history', type=int, default=500, help=('Maximum number of steps stored in replay'))
parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
parser.add_argument('--freeze-interval', type=int, default=200, help=('Interval between target freezes'))
parser.add_argument('--update-frequency', type=int, default=4, help=('Number of actions before each SGD update'))
parser.add_argument('--termination-reg', type=float, default=0.01, help=('Regularization to decrease termination prob'))
parser.add_argument('--entropy-reg', type=float, default=0.01, help=('Regularization to increase policy entropy'))
parser.add_argument('--num-options', type=int, default=4, help=('Number of options to create'))
parser.add_argument('--temp', type=float, default=1, help='Action distribution softmax tempurature param')

parser.add_argument('--max_steps_ep', type=int, default=50, help='Number of maximum steps per episode')
# parser.add_argument('--max_steps_total', type=int, default=50000, help='Number of maximum steps to take')
parser.add_argument('--cuda', type=bool, default=True, help='Enable CUDA training (recommended if possible).')
parser.add_argument('--seed', type=int, default=0, help='Random seed for numpy, torch, random.')
parser.add_argument('--logdir', type=str, default='runs', help='Directory for logging statistics')
parser.add_argument('--exp', type=str, default=None, help='Optional experiment name')
parser.add_argument('--max_episode', type=int, default=5000, help='Number of maximum episodes')
parser.add_argument('--advice_frequency', type=int, default=100, help='Frequency of add advice')


def run_option_critic(args):
    env = RefuelingEnv()

    # Potential based reward shaping
    pbrs = PotentialBasedRewardShaping(args.gamma, w=0.5)
    pbrs.add_constraint_automatically()
    print("Constraint Library: ")
    for key, value in pbrs.constraint_library.constraint_dict.items():
        print(key, value)

    option_critic = OptionCriticFeatures
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    option_critic = option_critic(
        in_features=env.state_space_shape,
        num_actions=env.action_space_shape,
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
    constraint_violations = []
    action_sequence = []

    while episode_num < args.max_episode:
        num_constraint_violations = 0
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
            action = env.get_action_by_index(action)

            potential, is_violated, suggested_action = pbrs.potential_function(pbrs.get_info_from_state(env.state), action)
            if is_violated:
                num_constraint_violations += 1
            action = suggested_action if is_violated else action

            next_obs, reward, done = env.step(action)
            reward = pbrs.shaped_reward(reward, potential)

            buffer.push(obs, current_option, reward, next_obs, done)
            rewards += reward

            if (episode_num + 1) % args.advice_frequency == 0:
                print("Step:", ep_steps + 1, "Action:", action, "Reward:", reward)
            if episode_num + 1 == args.max_episode:
                action_sequence.append(action)

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
        constraint_violations.append(num_constraint_violations)
        print("Episode: ", episode_num + 1, "Reward: ", rewards, "Constraint Violations: ", num_constraint_violations)

        # Add advice from natural language input
        '''
        if (episode_num + 1) % args.advice_frequency == 0:
            while True:
                advice = input("Please enter the advice: ")
                if not advice.strip():
                    print("End of adding advice, continue training...")
                    break
                generated_code = add_advice(advice)
                print(generated_code)
                try:
                    exec(generated_code)
                    print("Advice added successfully!")
                except Exception as e:
                    print(f"Advice added failed with error: {e}")
        '''
        episode_num += 1

    return episode_reward, constraint_violations


if __name__ == "__main__":
    start_time = time.time()
    current_time = datetime.now()
    current_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print("Running Option Critic on Refuel Task with Constraint Library at", current_time)
    args = parser.parse_args()
    episode_reward, constraint_violations = run_option_critic(args)
    end_time = time.time()
    current_time = datetime.now()
    current_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print("Finished running Option Critic on Refuel Task with Constraint Library at ", current_time)
    print("Time taken: ", (end_time - start_time)/60.0, "minutes.")
    episode_reward = np.array(episode_reward)
    constraint_violation = np.array(constraint_violations)

    plt.figure(1)
    plt.plot(episode_reward, label='option-critic')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Option Critic on Refuel Task with Constraint Library")

    plt.figure(2)
    plt.plot(constraint_violations)
    plt.xlabel("Episode")
    plt.ylabel("Number of Constraint Violations")
    plt.title("Constraint Violations")
    plt.show()


