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
reward_oc_uncover = savgol_filter(reward_oc_uncover, window_length=15, polyorder=3)

plt.figure(1)
plt.plot(x, reward_oc_uncover, label='选项-评论家方法')
plt.fill_between(x, reward_oc_uncover - 4, reward_oc_uncover + 4, alpha=0.5)
plt.xlabel("回合")
plt.ylabel("奖励值")
plt.title("两种任务规划方法在切割和移动保护层任务中的性能表现")
plt.legend()
plt.show()