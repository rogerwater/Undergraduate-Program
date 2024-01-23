import numpy as np
import random


class RefuelingEnv():
    def __init__(self):
        # 初始化状态
        # 第一行表示机器人的位置，grab状态
        # 第二行至第五行表示四个工具的位置和grab状态，初始状态下都位于工具箱处
        # 第六行表示加注位置，姿态和加注状态，初始状态下加注状态为0，表示未加注
        self.state_space_shape = (6, 2)
        self.robot_init_state = (0, 0)
        self.tool_state = (3, 0)
        self.refueling_state = (5, 0)
        self.tool_actions = ['uncover', 'screw', 'insert', 'replenish']
        self.action_space = ['move_to_toolbox', 'move_to_refueling_position'] + \
                            ['grab_tool_' + tool for tool in self.tool_actions] + \
                            ['release_tool_' + tool for tool in self.tool_actions] + \
                            self.tool_actions  # 14 actions in total
        self.current_action_index = 0 # 判断加注任务执行到哪一步：0表示未开始，1表示uncover，2表示screw，3表示insert，4表示replenish
        self.state = None
        self.reset()

    def reset(self):
        # 重置环境
        robot_state = np.array([self.robot_init_state], dtype=int)
        tools_state = np.array([self.tool_state] * 4, dtype=int)
        refueling_state = np.array([self.refueling_state], dtype=int)
        self.state = np.concatenate((robot_state, tools_state, refueling_state), axis=0)
        return self.state

    def step(self, action):
        reward = -1
        done = False

        if action == 'move_to_toolbox':
            self.state[0][0] = self.tool_state[0]

        elif action == 'move_to_refueling_position':
            self.state[0][0] = self.refueling_state[0]

        elif action.startswith('grab_tool_'):
            tool_index = self.tool_actions.index(action[10:])
            # 前提条件：机器人在工具箱处，工具在工具箱处，工具未被抓取
            if self.state[0][0] == self.tool_state[0] and self.state[tool_index + 1][0] == self.tool_state[0] and \
                    self.state[tool_index + 1][1] == 0 and self.state[0][1] == 0:
                self.state[tool_index + 1][1] = 1
                self.state[0][1] = 1
            else:
                reward = -5

        elif action.startswith('release_tool_'):
            tool_index = self.tool_actions.index(action[13:])
            # 前提条件：机器人在工具箱处，工具在机器人处，工具被抓取
            if self.state[0][0] == self.tool_state[0] and self.state[tool_index + 1][0] == self.state[0][0] and \
                    self.state[tool_index + 1][1] == 1 and self.state[0][1] == 1:
                self.state[tool_index + 1][2] = 0
                self.state[0][1] = 0
            else:
                reward = -5

        elif action in self.tool_actions:
            if self.current_action_index < len(self.tool_actions) and action == self.tool_actions[
                self.current_action_index]:
                # 前提条件：机器人在加注位置，工具在机器人处，工具被抓取，加注位置未加注
                if self.state[0][0] == self.refueling_state[0] and self.state[self.current_action_index][0] == \
                        self.state[0][0] and self.state[self.current_action_index][1] == 1 and self.state[
                    self.current_action_index][1] == 0:
                    self.current_action_index += 1
                    self.state[5][1] += 1
                    reward = 30
                else:
                    reward = -5
        if self.current_action_index == len(self.tool_actions) - 1:
            done = True
        return self.state, reward, done

    def choose_action(self):
        return random.choice(self.action_space)


if __name__ == "__main__":
    refuel_env = RefuelingEnv()
    refuel_env.reset()
    print("Initial State:")
    print(refuel_env.state)

    for _ in range(10):
        action = refuel_env.choose_action()
        state, reward, done = refuel_env.step(action)
        print(f"Action: {action}, Reward: {reward}, Done: {done}, State:\n{state}")

    print("Final State:")
    print(refuel_env.state)

