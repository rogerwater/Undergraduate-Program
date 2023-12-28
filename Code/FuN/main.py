import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import make_envs, take_action
from feudalnet import FeudalNet, feudal_loss
from storage import Storage
# from logger import Logger

parser = argparse.ArgumentParser(description='Feudal Networks')
# Generic RL/Model Parameters
parser.add_argument('--lr', type=float, default=0.0005,
                    help='learning rate')
parser.add_argument('--env-name', type=str, default='BreakoutNoFrameskip-v4',
                    help='gym environment name')
parser.add_argument('--num-workers', type=int, default=4,
                    help='number of parallel environments to run')
parser.add_argument('--num-steps', type=int, default=400,
                    help='number of steps the agent takes before updating')
parser.add_argument('--max-steps', type=int, default=int(1e7),
                    help='maximum number of training steps in total')
parser.add_argument('--cuda', type=bool, default=True,
                    help='Add cuda')
parser.add_argument('--grad-clip', type=float, default=5.,
                    help='Gradient clipping (recommended).')
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='Entropy coefficient to encourage exploration.')
parser.add_argument('--mlp', type=int, default=0,
                    help='toggle to feedforward ML architecture')
# Specific Feudalnet Parameters
parser.add_argument('--time-horizon', type=int, default=10,
                    help='Manager horizon (c)')
parser.add_argument('--hidden-dim-manager', type=int, default=256,
                    help='Hidden dim (d)')
parser.add_argument('--hidden-dim-worker', type=int, default=16,
                    help='Hidden dim for worker (k)')
parser.add_argument('--gamma-w', type=float, default=0.99,
                    help="discount factor worker")
parser.add_argument('--gamma-m', type=float, default=0.999,
                    help="discount factor manager")
parser.add_argument('--alpha', type=float, default=0.5,
                    help='Intrinsic reward coefficient in [0, 1]')
parser.add_argument('--eps', type=float, default=int(1e-5),
                    help='Random Gausian goal for exploration')
parser.add_argument('--dilation', type=int, default=10,
                    help='Dilation parameter for manager LSTM.')
# Experiment Related Parameters
parser.add_argument('--run-name', type=str, default='baseline',
                    help='run name for the logger.')
parser.add_argument('--seed', type=int, default=0,
                    help='reproducibility seed.')

if __name__ == "__main__":
    args = parser.parse_args()
    run_name = args.run_name

    save_steps = list(torch.arange(0, int(args.max_steps), int(args.max_steps) // 10).numpy())
    # logger = Logger(args.run_name, args)

    cuda_is_available = torch.cuda.is_available() and args.cuda
    device = torch.device("cuda" if cuda_is_available else "cpu")
    torch.manual_seed(args.seed)
    if cuda_is_available:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    envs = make_envs(args.env_name, args.num_workers, args.seed)
    feudalnet = FeudalNet(
        num_workers=args.num_workers,
        input_dim=envs.single_observation_space.shape,
        hidden_dim_manager=args.hidden_dim_manager,
        hidden_dim_worker=args.hidden_dim_worker,
        n_actions=envs.single_action_space.n,
        dilation=args.dilation,
        device=device,
        mlp=args.mlp,
        args=args
    )

    optimizer = torch.optim.RMSprop(feudalnet.parameters(), lr=args.lr,
                                    alpha=0.99, eps=1e-5)

    goals, states, masks = feudalnet.init_obj()
    x = envs.reset()
    step = 0
    episode = 1
    episode_rewards = []
    while step < args.max_steps:
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

            action, logp, entropy = take_action(action_dist)
            x, reward, done, info = envs.step(action)
            # logger.log_episode(info, step)
            rewards += reward
            mask = torch.FloatTensor(1 - done).unsqueeze(-1).to(device)
            masks.pop(0)
            masks.append(mask)

            storage.add({
                'r': torch.FloatTensor(reward).unsqueeze(-1).to(device),
                'r_i': feudalnet.intrinsic_reward(states, goals, masks),
                'v_w': value_w,
                'v_m': value_m,
                'logp': logp.unsqueeze(-1),
                'entropy': entropy.unsqueeze(-1),
                's_goal_cos': feudalnet.state_goal_cosine(states, goals, masks),
                'm': mask
            })

            step += args.num_workers
        episode_rewards.append(np.mean(rewards))
        print(f"> episode = {episode} | episode_rewards = {np.mean(rewards)} | num_steps = {step}")
        episode += 1

        with torch.no_grad():
            *_, next_v_m, next_v_w = feudalnet(x, goals, states, mask, save=False)
            next_v_m = next_v_m.detach()
            next_v_w = next_v_w.detach()

        optimizer.zero_grad()
        loss, loss_dict = feudal_loss(storage, next_v_m, next_v_w, args)
        loss.backward()
        torch.nn.utils.clip_grad_norm(feudalnet.parameters(), args.grad_clip)
        optimizer.step()
        # logger.log_scalars(loss_dict, step)

    plt.plot(episode_rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("FeUdal Networks on BreakoutNoFrameskip-v4")
    plt.legend()
    plt.show()
    envs.close()
    torch.save({
        'model': feudalnet.state_dict(),
        'args': args,
        'processor_mean': feudalnet.preprocessor.rms.mean,
        'optim': optimizer.state_dict()},
        f'models/{args.env_name}_{args.run_name}_steps={step}.pt')

