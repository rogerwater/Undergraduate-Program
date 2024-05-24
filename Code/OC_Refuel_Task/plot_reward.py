import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import numpy as np
from scipy.signal import savgol_filter

x = np.arange(1, 1001)
reward = np.load('episode_reward.npy')
reward_human = np.load('episode_reward_human.npy')
reward_basic = np.load('episode_reward_basic.npy')
reward = savgol_filter(reward, window_length=51, polyorder=3)
reward_human = savgol_filter(reward_human, window_length=51, polyorder=3)
reward_basic = savgol_filter(reward_basic, window_length=51, polyorder=3)

reward_oc_uncover = np.load('episode_reward_oc_uncover.npy')
reward_oc_uncover = savgol_filter(reward_oc_uncover, window_length=20, polyorder=3)
reward_fn_uncover = np.load('episode_reward_fn_uncover.npy')
reward_fn_uncover = savgol_filter(reward_fn_uncover, window_length=20, polyorder=3)

reward_fn_cartpole = np.load('episode_reward_fn_cartpole.npy')
reward_fn_cartpole = savgol_filter(reward_fn_cartpole, window_length=20, polyorder=3)
reward_oc_cartpole = np.load('episode_reward_oc_cartpole.npy')
reward_oc_cartpole = savgol_filter(reward_oc_cartpole, window_length=20, polyorder=3)

plt.figure(1)
plt.plot(x, reward_oc_uncover, label='选项-评论家方法')
plt.fill_between(x, reward_oc_uncover - 3, reward_oc_uncover + 3, alpha=0.5)
plt.plot(x, reward_fn_uncover, label='封建层级网络方法')
plt.fill_between(x, reward_fn_uncover - 3, reward_fn_uncover + 3, alpha=0.5)
plt.xlabel("回合")
plt.ylabel("奖励值")
plt.title("两种任务规划方法在切割和移动保护层任务中的性能表现")
plt.legend()
plt.show()

plt.figure(2)
plt.plot(x, reward_oc_cartpole, label='选项-评论家方法')
plt.fill_between(x, reward_oc_cartpole - 10, reward_oc_cartpole + 10, alpha=0.5)
plt.plot(x, reward_fn_cartpole, label='封建层级网络方法')
plt.fill_between(x, reward_fn_cartpole - 10, reward_fn_cartpole + 10, alpha=0.5)
plt.xlabel("回合")
plt.ylabel("奖励值")
plt.title("两种任务规划方法在车杆环境中的性能表现")
plt.legend()
plt.show()