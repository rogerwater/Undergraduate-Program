import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import numpy as np
from scipy.signal import savgol_filter

reward = np.load('episode_reward.npy')
reward_human = np.load('episode_reward_human.npy')
reward_basic = np.load('episode_reward_basic.npy')
reward = savgol_filter(reward, window_length=51, polyorder=3)
reward_human = savgol_filter(reward_human, window_length=51, polyorder=3)
reward_basic = savgol_filter(reward_basic, window_length=51, polyorder=3)

plt.figure(1)
plt.plot(reward, label='基于规则库的选项-评论家方法')
# plt.plot(reward_human, label='人机协同任务规划方法')
plt.plot(reward_basic, label='原始的选项-评论家方法')
plt.xlabel("回合")
plt.ylabel("奖励值")
plt.title("在轨加注操作任务规划方法的性能表现")
plt.legend()
plt.show()