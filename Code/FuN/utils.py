import gym
from gym.wrappers import AtariPreprocessing, TransformReward
import numpy as np
import torch
from torch.distributions import Categorical


def atari_wrapper(env):
    env = AtariPreprocessing(env, grayscale_obs=False, scale_obs=True)
    env = TransformReward(env, lambda r: np.sign(r))
    return env


def make_envs(env_name, num_envs, seed=0):
    wrapper_fn = atari_wrapper
    envs = gym.vector.make(env_name, num_envs=num_envs, wrappers=wrapper_fn)
    envs.seed(seed)
    return envs


def take_action(a):
    dist = Categorical(a)
    action = dist.sample()
    logp = dist.log_prob(action)
    entropy = dist.entropy()
    return action.cpu().detach().numpy(), logp, entropy


def init_hidden(n_workers, h_dim, device, grad=False):
    return (torch.zeros(n_workers, h_dim, requires_grad=grad).to(device),
            torch.zeros(n_workers, h_dim, requires_grad=grad).to(device))


def init_weight(layer):
    if type(layer) == torch.nn.modules.conv.Conv2d or type(layer) == torch.nn.Linear:
        torch.nn.init.orthogonal_(layer.weight.data)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias.data, 0)

