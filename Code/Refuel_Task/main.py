import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from model import RefuelingEnv
from utils import take_action
from feudalnet import FeudalNet, feudal_loss
from storage import Storage

parser = argparse.ArgumentParser(description='Feudal Networks')
# Generic RL/Model Parameters
parser.add_argument('--lr', type=float, default=0.0005,
                    help='learning rate')
parser.add_argument('--env-name', type=str, default='BreakoutNoFrameskip-v4',
                    help='gym environment name')
parser.add_argument('--num-workers', type=int, default=1,
                    help='number of parallel environments to run')
parser.add_argument('--num-steps', type=int, default=50,
                    help='number of steps the agent takes before updating')
parser.add_argument('--max-steps', type=int, default=int(1e7),
                    help='maximum number of training steps in total')
parser.add_argument('--max-episodes', type=int, default=1e4,
                    help='maximum number of training episodes in total')
parser.add_argument('--cuda', type=bool, default=True,
                    help='Add cuda')
parser.add_argument('--grad-clip', type=float, default=1.0,
                    help='Gradient clipping (recommended).')
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='Entropy coefficient to encourage exploration.')
# Specific Feudal Networks Parameters
parser.add_argument('--time-horizon', type=int, default=5,
                    help='Manager horizon (c)')
parser.add_argument('--hidden-dim-manager', type=int, default=8,
                    help='Hidden dim (d)')
parser.add_argument('--hidden-dim-worker', type=int, default=2,
                    help='Hidden dim for worker (k)')
parser.add_argument('--gamma-w', type=float, default=0.99,
                    help="discount factor worker")
parser.add_argument('--gamma-m', type=float, default=0.999,
                    help="discount factor manager")
parser.add_argument('--alpha', type=float, default=0.5,
                    help='Intrinsic reward coefficient in [0, 1]')
parser.add_argument('--eps', type=float, default=int(1e-5),
                    help='Random Gausian goal for exploration')
parser.add_argument('--dilation', type=int, default=5,
                    help='Dilation parameter for manager LSTM.')
# Experiment Related Parameters
parser.add_argument('--run-name', type=str, default='baseline',
                    help='run name for the logger.')
parser.add_argument('--seed', type=int, default=0,
                    help='reproducibility seed.')


if __name__ == '__main__':
    args = parser.parse_args()
    cuda_is_available = torch.cuda.is_available() and args.cuda
    device = torch.device("cuda" if cuda_is_available else "cpu")
    torch.manual_seed(args.seed)
    if cuda_is_available:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    env = RefuelingEnv()

    feudalnet = FeudalNet(
        num_workers=args.num_workers,
        input_dim=env.state_space_shape,
        hidden_dim_manager=args.hidden_dim_manager,
        hidden_dim_worker=args.hidden_dim_worker,
        n_actions=env.action_space_shape,
        dilation=args.dilation,
        device=device,
        args=args
    )

    optimizer = torch.optim.RMSprop(feudalnet.parameters(), lr=args.lr,
                                    alpha=0.99, eps=1e-5)

    goals, states, masks = feudalnet.init_obj()
    x = env.reset()
    episode = 1
    episode_rewards = []
    while episode <= args.max_episodes:
        # Detaching LSTMs and goals
        feudalnet.repackage_hidden()
        goals = [g.detach() for g in goals]
        storage = Storage(size=args.num_steps,
                          keys=['r', 'r_i', 'v_w', 'v_m', 'logp', 'entropy',
                                's_goal_cos', 'mask', 'ret_w', 'ret_m',
                                'adv_m', 'adv_w'])
        rewards = 0
        for _ in range(args.num_steps):
            action_dist, goals, states, value_m, value_w = feudalnet(
                x, goals, states, masks[-1])
            # print(action_dist)
            action, logp, entropy = take_action(action_dist)
            # print(action)
            x, reward, done = env.step(env.get_action_by_index(action))

            # 测试训练效果
            if episode == args.max_episodes:
                print(f"Action: {action}, Reward: {reward}, Done: {done}, State:\n{x}")
                log_file_path = 'log.txt'
                log_entry = f"Action: {action}, Reward: {reward}, Done: {done}, State:\n{x}\n"
                with open(log_file_path, 'a') as log_file:
                    log_file.write(log_entry)

            rewards += reward
            mask = torch.FloatTensor(1 - done).unsqueeze(-1).to(device)
            masks.pop(0)
            masks.append(mask)

            storage.add({
                'r': torch.FloatTensor([reward]).to(device),
                'r_i': feudalnet.intrinsic_reward(states, goals, masks),
                'v_w': value_w,
                'v_m': value_m,
                'logp': logp,
                'entropy': entropy,
                's_goal_cos': feudalnet.state_goal_cosine(states, goals, masks),
                'm': mask
            })

        episode_rewards.append(np.mean(rewards))
        print(f"> episode = {episode} | episode_rewards = {rewards}")
        episode += 1

        with torch.no_grad():
            *_, next_v_m, next_v_w = feudalnet(x, goals, states, mask, save=False)
            next_v_m = next_v_m.detach()
            next_v_w = next_v_w.detach()

        optimizer.zero_grad()
        loss = feudal_loss(storage, next_v_m, next_v_w, args)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(feudalnet.parameters(), args.grad_clip)
        optimizer.step()

    plt.plot(episode_rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Total Rewards")
    plt.title("Feudal Networks on Refueling Task")
    plt.legend()
    plt.show()






